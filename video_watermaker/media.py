"""FFmpeg probing and compression policies."""

import json
import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .process import MediaError, run

logger = logging.getLogger(__name__)


def number(value, default=0.0):
    try:
        result = float(Fraction(str(value).replace(":", "/")))
        return result if math.isfinite(result) else default
    except (ValueError, ZeroDivisionError):
        return default


def stream_duration(stream: dict) -> float:
    duration = number(stream.get("duration"))
    if duration > 0:
        return duration
    duration = number(stream.get("duration_ts")) * number(stream.get("time_base"))
    if duration > 0:
        return duration
    parts = str(stream.get("tags", {}).get("DURATION", "")).split(":")
    if len(parts) == 3:
        return number(parts[0]) * 3600 + number(parts[1]) * 60 + number(parts[2])
    return 0


@dataclass(frozen=True)
class VideoInfo:
    video: dict
    audio: dict | None
    duration: float
    width: int
    height: int
    display_width: float
    display_height: float
    fps: float

    @property
    def hdr(self):
        return self.video.get("color_transfer") in ("smpte2084", "arib-std-b67")


def probe(path: Path, stop: threading.Event) -> VideoInfo:
    raw = run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        stop,
        timeout=30,
    )
    try:
        data = json.loads(raw)
        streams = data.get("streams", [])
        video = next(
            s
            for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        )
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        width, height = int(video["width"]), int(video["height"])
        duration = stream_duration(video) or number(data.get("format", {}).get("duration"))
        if width < 2 or height < 2 or duration <= 0:
            raise ValueError("无有效尺寸或时长")
        sar = number(video.get("sample_aspect_ratio"), 1) or 1
        dw, dh = width * sar, float(height)
        rotation = number(video.get("tags", {}).get("rotate"))
        for side in video.get("side_data_list", []):
            if "rotation" in side:
                rotation = number(side["rotation"])
        # FFmpeg autorotates before our filters; use the same display orientation.
        if round(rotation) % 180 in (90,):
            dw, dh = dh, dw
        fps = number(video.get("avg_frame_rate")) or number(video.get("r_frame_rate"))
        return VideoInfo(video, audio, duration, width, height, dw, dh, fps)
    except (ValueError, KeyError, StopIteration, TypeError) as exc:
        raise MediaError(f"无法读取有效视频信息：{path} ({exc})") from exc


def dimensions(info: VideoInfo) -> tuple[int, int]:
    dw, dh = info.display_width, info.display_height
    mw, mh = (1280, 720) if dw >= dh else (720, 1280)
    scale = min(1.0, mw / dw, mh / dh)
    return max(2, int(dw * scale) // 2 * 2), max(2, int(dh * scale) // 2 * 2)


def tone_map(info: VideoInfo) -> list[str]:
    if not info.hdr:
        return []
    return [
        "zscale=t=linear:npl=100",
        "format=gbrpf32le",
        "zscale=p=bt709",
        "tonemap=tonemap=hable:desat=0",
        "zscale=t=bt709:m=bt709:r=tv",
    ]


class EncoderPolicy:
    def __init__(self, requested: str, stop: threading.Event):
        self.requested = requested
        self._lock = threading.Lock()
        self._selected = "cpu"
        if requested != "cpu":
            try:
                run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-f",
                        "lavfi",
                        "-i",
                        "color=size=256x256:rate=30",
                        "-t",
                        "0.1",
                        "-an",
                        "-c:v",
                        "h264_nvenc",
                        "-preset",
                        "p5",
                        "-f",
                        "null",
                        "-",
                    ],
                    stop,
                    timeout=20,
                )
                self._selected = "nvenc"
            except MediaError as exc:
                if requested == "nvenc":
                    raise MediaError(f"指定的 NVENC 不可用：{exc}") from exc
                logger.warning("NVENC 不可用，使用 CPU：%s", exc)
        logger.info("压缩编码器：%s", self._selected)

    @property
    def selected(self):
        with self._lock:
            return self._selected

    def fallback(self, error: MediaError) -> bool:
        detail = str(error).lower()
        hardware_error = any(
            token in detail
            for token in (
                "cuda",
                "nvenc",
                "nvidia",
                "no capable devices",
                "cannot load lib",
                "device lost",
            )
        )
        if self.requested != "auto" or not hardware_error:
            return False
        if "dimension less than the minimum" in detail:
            logger.info("本视频尺寸低于显卡编码要求，仅此任务使用 CPU")
            return True
        with self._lock:
            if self._selected != "cpu":
                logger.warning("硬件编码失败，本轮后续任务切换 CPU：%s", error)
                self._selected = "cpu"
        return True


def compress(
    source: Path,
    target: Path,
    info: VideoInfo,
    policy: EncoderPolicy,
    threads: int,
    stop: threading.Event,
    progress: Callable[[float], None],
):
    width, height = dimensions(info)
    filters = [*tone_map(info), f"scale={width}:{height}:flags=lanczos", "setsar=1"]

    def encode(encoder: str):
        codec = (
            ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
            if encoder == "nvenc"
            else ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]
        )
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-xerror",
            "-y",
            "-threads",
            str(threads),
            "-filter_threads",
            str(threads),
            "-i",
            str(source),
            "-map",
            f"0:{info.video['index']}",
        ]
        if info.audio is not None:
            channels = max(1, min(2, int(info.audio.get("channels", 2))))
            args += ["-map", f"0:{info.audio['index']}", "-c:a", "aac", "-b:a", "128k", "-ac", str(channels)]
        else:
            args += ["-an"]
        args += [
            "-sn",
            "-dn",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-vf",
            ",".join(filters),
            "-fpsmax:v",
            "30",
            *codec,
            "-threads:v",
            str(threads),
            "-pix_fmt",
            "yuv420p",
            "-maxrate:v",
            "1500k",
            "-bufsize:v",
            "3000k",
            "-metadata:s:v:0",
            "rotate=0",
        ]
        if info.hdr:
            args += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
        args += ["-movflags", "+faststart", "-progress", "pipe:1", "-nostats", str(target)]
        run(args, stop, timeout=None, progress=lambda seconds: progress(min(0.99, seconds / info.duration)))

    selected = policy.selected
    try:
        encode(selected)
    except MediaError as exc:
        if "no space left" in str(exc).lower():
            raise
        if selected != "nvenc" or not policy.fallback(exc):
            raise
        target.unlink(missing_ok=True)
        encode("cpu")
    result = probe(target, stop)
    if (
        result.video.get("codec_name") != "h264"
        or (result.width, result.height) != (width, height)
        or result.fps > 30.01
        or result.video.get("pix_fmt") != "yuv420p"
    ):
        raise MediaError("成品校验失败：编码、尺寸、帧率或像素格式不符合目标")
    if abs(result.duration - info.duration) > max(0.5, 2 / min(info.fps or 30, 30)):
        raise MediaError("成品校验失败：视频时长发生异常变化")
    if info.audio and (result.audio is None or result.audio.get("codec_name") != "aac"):
        raise MediaError("成品校验失败：音轨缺失或编码错误")
