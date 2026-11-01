"""Uniformly sampled contact sheets with aspect-correct cells."""

import io
import math
import threading
from collections.abc import Callable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .media import VideoInfo, tone_map
from .process import MediaError, check_cancelled, run

DEFAULT_FONT = Path(__file__).resolve().parent.parent / "fonts" / "Alibaba-PuHuiTi-Medium.ttf"
MAX_SHEET_PIXELS = 50_000_000


def timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _fit_text(draw, text, font, width):
    if draw.textbbox((0, 0), text, font=font)[2] <= width:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if draw.textbbox((0, 0), text[:mid] + "…", font=font)[2] <= width:
            low = mid
        else:
            high = mid - 1
    return text[:low] + "…"


def _frame(
    source: Path,
    info: VideoInfo,
    at: float,
    width: int,
    height: int,
    rotate: int,
    threads: int,
    stop: threading.Event,
) -> tuple[Image.Image, float]:
    rotations = {0: [], 90: ["transpose=2"], 180: ["hflip", "vflip"], 270: ["transpose=1"]}
    filters = [*tone_map(info), *rotations[rotate], f"scale={width}:{height}", "setsar=1"]
    # A seek near EOF can land after the last decodable frame, especially in tiny clips.
    candidates = dict.fromkeys((at, max(0, at - max(0.25, 1 / (info.fps or 25))), 0.0))
    for candidate in candidates:
        data = run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-threads",
                str(threads),
                "-filter_threads",
                str(threads),
                "-ss",
                f"{candidate:.6f}",
                "-i",
                str(source),
                "-map",
                f"0:{info.video['index']}",
                "-an",
                "-sn",
                "-dn",
                "-frames:v",
                "1",
                "-vf",
                ",".join(filters),
                "-c:v",
                "png",
                "-threads:v",
                "1",
                "-f",
                "image2pipe",
                "pipe:1",
            ],
            stop,
            timeout=60,
            capture_limit=64 * 1024 * 1024,
        )
        if data:
            with Image.open(io.BytesIO(data)) as frame:
                return frame.convert("RGB"), candidate
    raise MediaError(f"无法解码截图：{source} @ {at:.3f}s")


def create_thumbnail(
    source: Path,
    target: Path,
    info: VideoInfo,
    params: dict,
    threads: int,
    stop: threading.Event,
    progress: Callable[[float], None],
):
    rows, cols, width = params["rows"], params["cols"], params["width"]
    dw, dh = info.display_width, info.display_height
    if params["rotate"] in (90, 270):
        dw, dh = dh, dw
    cell_width = math.ceil(width / cols)
    cell_height = max(1, round(cell_width * dh / dw))
    font_size = max(12, width // 100)
    header_height = max(40, font_size * 3)
    height = rows * cell_height + header_height
    if width * height > MAX_SHEET_PIXELS:
        raise MediaError("缩略图超过 5000 万像素，请减小 --width 或调整网格")
    font_path = params.get("font")
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
        small_font = ImageFont.truetype(font_path, max(10, font_size - 2))
    else:
        font = ImageFont.load_default(size=font_size)
        small_font = ImageFont.load_default(size=max(10, font_size - 2))
    sheet = Image.new("RGB", (width, height), "black")
    try:
        draw = ImageDraw.Draw(sheet)
        draw.rectangle((0, 0, width, header_height), fill="white")
        size = source.stat().st_size / (1024 * 1024)
        label = f"{source.name} | {size:.1f} MiB | {timestamp(info.duration)}"
        label = _fit_text(draw, label, font, max(1, width - 16))
        draw.text((width // 2, header_height // 2), label, font=font, fill="black", anchor="mm")
        count = rows * cols
        for index in range(count):
            check_cancelled(stop)
            at = info.duration * (index + 0.5) / count
            frame, actual_time = _frame(
                source, info, at, cell_width, cell_height, params["rotate"], threads, stop
            )
            try:
                row, col = divmod(index, cols)
                left, right = col * width // cols, (col + 1) * width // cols
                top = header_height + row * cell_height
                contained = ImageOps.contain(frame, (right - left, cell_height), Image.Resampling.LANCZOS)
                sheet.paste(
                    contained,
                    (
                        left + (right - left - contained.width) // 2,
                        top + (cell_height - contained.height) // 2,
                    ),
                )
                contained.close()
                stamp = _fit_text(draw, timestamp(actual_time), small_font, max(1, right - left - 8))
                if cell_height >= font_size + 8 and right - left >= font_size * 3:
                    draw.text(
                        (right - 4, top + cell_height - 4),
                        stamp,
                        font=small_font,
                        fill="white",
                        stroke_width=1,
                        stroke_fill="black",
                        anchor="rs",
                    )
            finally:
                frame.close()
            progress((index + 1) / count * 0.95)
        watermark = params.get("watermark_text")
        if watermark:
            size = max(12, round(width * 0.04))
            wm_font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default(size=size)
            watermark = _fit_text(draw, watermark, wm_font, max(1, width - 20))
            # Small text-only overlay avoids a second full-size RGBA contact sheet.
            box = draw.textbbox((0, 0), watermark, font=wm_font, stroke_width=2)
            overlay = Image.new("RGBA", (box[2] - box[0] + 8, box[3] - box[1] + 8))
            try:
                ImageDraw.Draw(overlay).text(
                    (4 - box[0], 4 - box[1]),
                    watermark,
                    font=wm_font,
                    fill=(255, 255, 255, 204),
                    stroke_width=2,
                    stroke_fill=(0, 0, 0, 128),
                )
                sheet.paste(overlay, ((width - overlay.width) // 2, (height - overlay.height) // 2), overlay)
            finally:
                overlay.close()
        check_cancelled(stop)
        sheet.save(target, "JPEG", quality=params["quality"])
    finally:
        sheet.close()
    with Image.open(target) as result:
        result.verify()
