"""Command line interface; importing modules does not configure logging or start work."""

import argparse
import logging
import logging.handlers
import os
import shutil
import signal
import sqlite3
import sys
import threading
import uuid
from pathlib import Path

from PIL import ImageFont

from . import __version__
from .discovery import discover, output_path
from .media import EncoderPolicy
from .process import Cancelled, MediaError, check_cancelled
from .runner import RunnerLock, process_task, reset_scratch, run_queue, scratch_root
from .store import ConflictError, Store
from .thumbnail import DEFAULT_FONT

logger = logging.getLogger(__name__)


def _integer(low: int, high: int):
    def parse(value):
        try:
            result = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("必须是整数") from exc
        if not low <= result <= high:
            raise argparse.ArgumentTypeError(f"必须介于 {low} 和 {high} 之间")
        return result

    return parse


def parser():
    root = argparse.ArgumentParser(description="可续跑的视频目录队列：缩略图、水印与 720p 压缩")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    tui = commands.add_parser("tui", help="打开中文交互终端界面")
    tui.add_argument("input", type=Path, nargs="?", help="预填视频文件或目录")
    tui.add_argument("-o", "--output", type=Path, help="预填输出目录并查看队列")
    for name, help_text in (
        ("thumbnail", "生成网格缩略图"),
        ("compress", "压缩到最大 720p / 30fps / 视频 1.5Mbps"),
        ("transcribe", "提取视频对话，区分说话人并导出文字和字幕"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("input", type=Path, help="视频文件或目录；目录默认递归")
        command.add_argument(
            "-o", "--output", type=Path, default=Path("output"), help="输出目录，默认 output"
        )
        command.add_argument("--retry-failed", action="store_true", help="重新执行失败任务")
        if name in ("compress", "transcribe"):
            command.add_argument("--model", choices=("large-v3-turbo", "large-v3"), default="large-v3-turbo")
            command.add_argument("--language", default="zh", help="识别语言代码，默认 zh；auto 自动识别")
            command.add_argument(
                "--device",
                choices=("auto", "cuda", "cpu"),
                default="auto",
                help="语音模型设备；默认 auto，语音任务固定单并发",
            )
            command.add_argument("--speakers", type=_integer(1, 100), help="指定说话人数，默认自动判断")
        if name == "transcribe":
            command.set_defaults(workers=1)
            continue
        command.add_argument("--workers", type=_integer(1, 32), default=2, help="媒体任务并发数，默认 2")
        command.add_argument("--rows", type=_integer(1, 20), default=4)
        command.add_argument("--cols", type=_integer(1, 20), default=4)
        command.add_argument(
            "--width", type=_integer(64, 16384), default=3840, help="缩略图总宽度，默认 3840"
        )
        command.add_argument(
            "--rotate", type=int, choices=(0, 90, 180, 270), default=0, help="缩略图额外逆时针旋转角度"
        )
        command.add_argument("--quality", type=_integer(1, 100), default=80, help="缩略图 JPEG 质量")
        command.add_argument("--font", type=Path, help="缩略图字体文件")
        command.add_argument(
            "--watermark-text", "--watermark_text", dest="watermark_text", help="缩略图文字水印"
        )
        if name == "compress":
            command.add_argument("--transcribe", action="store_true", help="同时从原视频提取带说话人的对话")
            command.add_argument("--thumbnail", action="store_true", help="同时从原视频生成缩略图")
            command.add_argument(
                "--encoder",
                choices=("auto", "nvenc", "cpu"),
                default="auto",
                help="默认 auto：NVENC 优先，硬件不可用时回退 CPU",
            )
    status = commands.add_parser("status", help="查看输出目录的持久队列")
    status.add_argument("-o", "--output", type=Path, default=Path("output"))
    models = commands.add_parser("models", help="准备本地语音模型")
    prepare = models.add_subparsers(dest="models_command", required=True).add_parser("prepare")
    prepare.add_argument("--model", choices=("large-v3-turbo", "large-v3"), default="large-v3-turbo")
    return root


def _logging(state: Path):
    package = logging.getLogger("video_watermaker")
    package.setLevel(logging.INFO)
    package.propagate = False
    file = logging.handlers.RotatingFileHandler(
        state / "run.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(message)s"))
    for handler in (file, console):
        package.addHandler(handler)
    return package, (file, console)


def _summary(store: Store, batch: str | None = None):
    counts = store.counts(batch)
    print(
        f"成功 {counts['succeeded']} | 失败 {counts['failed']} | "
        f"运行 {counts['running']} | 待处理 {counts['pending']}"
    )
    for row in store.failures(limit=10, batch=batch):
        error = (row["error"] or "未知错误").replace("\n", " ")[-600:]
        print(f"  [{row['kind']}] {row['source']}: {error}")
    return counts


def _execute(args, state: Path, stop: threading.Event, interrupted: threading.Event, progress=None) -> int:
    source, output = args.input.resolve(), args.output.resolve()
    if not source.exists() or not (source.is_file() or source.is_dir()):
        raise MediaError(f"输入文件或目录不存在：{source}")
    if source.is_dir() and source == output:
        raise MediaError("输入目录和输出目录不能相同")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise MediaError(f"找不到 {executable}，请先安装系统 FFmpeg")
    with RunnerLock(state / "runner.lock"):
        store = Store(state / "queue.sqlite3")
        scratch = scratch_root(store.path)
        scratch_ready = False
        speech_client = None
        try:
            store.recover()
            reset_scratch(scratch)
            scratch_ready = True
            kinds = [args.command]
            if args.command == "compress" and args.thumbnail:
                kinds.append("thumbnail")
            if args.command == "compress" and args.transcribe:
                kinds.append("transcribe")
            font_arg = getattr(args, "font", None)
            font = font_arg.resolve() if font_arg else (DEFAULT_FONT if DEFAULT_FONT.is_file() else None)
            if "thumbnail" in kinds and font:
                try:
                    ImageFont.truetype(str(font), 16)
                except OSError as exc:
                    raise MediaError(f"字体不可用：{font}") from exc
            params = {}
            if "thumbnail" in kinds:
                params["thumbnail"] = {
                    "revision": 1,
                    "rows": args.rows,
                    "cols": args.cols,
                    "width": args.width,
                    "rotate": args.rotate,
                    "quality": args.quality,
                    "font": str(font) if font else None,
                    "font_mtime": font.stat().st_mtime_ns if font and "thumbnail" in kinds else None,
                    "watermark_text": args.watermark_text,
                }
            if "transcribe" in kinds:
                from .speech import speech_params

                params["transcribe"] = speech_params(args)
            if args.command == "compress":
                params["compress"] = {
                    "revision": 1,
                    "encoder": args.encoder,
                    "max_resolution": "720p",
                    "max_fps": 30,
                    "maxrate": 1500000,
                    "bufsize": 3000000,
                    "audio_bitrate": 128000,
                }
            batch = uuid.uuid4().hex
            total = skipped = errors = 0

            def scan_error(path, error):
                nonlocal errors
                errors += 1
                logger.error("导入失败：%s：%s", path, error)

            root = source if source.is_dir() else source.parent
            logger.info("扫描并导入：%s", source)
            try:
                for video in discover(source, output, scan_error):
                    check_cancelled(stop)
                    total += 1
                    for kind in kinds:
                        try:
                            result = store.enqueue(
                                video,
                                output_path(video, root, output, kind),
                                kind,
                                params[kind],
                                batch,
                                args.retry_failed,
                            )
                            skipped += result == "skipped"
                        except (OSError, ConflictError) as exc:
                            scan_error(video, exc)
                    if total % 100 == 0:
                        store.commit()
                    if total % 1000 == 0:
                        logger.info("已扫描 %s 个视频", total)
            finally:
                store.commit()
            if total == 0:
                raise MediaError("未发现视频文件；目录扫描只导入支持的视频扩展名")
            logger.info("导入 %s 个视频，跳过 %s 个已完成任务，导入错误 %s", total, skipped, errors)
            needs_speech = store.db.execute(
                """SELECT 1 FROM tasks t WHERE batch=? AND kind='transcribe' AND status='pending'
                   AND NOT EXISTS (SELECT 1 FROM speech_checkpoints c WHERE c.task_id=t.id AND c.name='result')
                   LIMIT 1""",
                (batch,),
            ).fetchone()
            if needs_speech:
                from .speech import read_manifest
                from .speech_client import SpeechClient

                speech_client = SpeechClient(stop)
                checked = speech_client.request(
                    "check",
                    {
                        "manifest": read_manifest(args.model),
                        "params": params["transcribe"],
                    },
                )
                logger.info("语音模型检查通过：%s；转写单进程排队", checked["device"])
            policy = None
            needs_compression = store.db.execute(
                "SELECT 1 FROM tasks WHERE batch=? AND kind='compress' AND status='pending' LIMIT 1", (batch,)
            ).fetchone()
            if needs_compression:
                policy = EncoderPolicy(args.encoder, stop)
            threads = max(1, min(4, (os.cpu_count() or 2) // args.workers))

            def work(task, progress):
                return process_task(task, store.path, scratch, policy, threads, stop, progress, speech_client)

            run_queue(
                store,
                batch,
                args.workers,
                stop,
                work,
                show_progress=progress is None,
                speech=False,
                on_progress=progress,
            )
            run_queue(
                store,
                batch,
                1,
                stop,
                work,
                show_progress=progress is None,
                speech=True,
                on_progress=progress,
            )
            counts = _summary(store, batch)
            if interrupted.is_set():
                logger.info("已停止；重复原命令即可续跑")
                return 130
            return 1 if errors or counts["failed"] or counts["pending"] else 0
        finally:
            if speech_client is not None:
                speech_client.close()
            store.close()
            if scratch_ready and scratch.exists() and not scratch.is_symlink():
                shutil.rmtree(scratch)


def main(argv=None, *, progress=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv and sys.stdin.isatty() and sys.stdout.isatty():
        argv = ["tui"]
    if argv and argv[0] not in (
        "tui",
        "compress",
        "thumbnail",
        "transcribe",
        "models",
        "status",
        "--help",
        "-h",
        "--version",
    ):
        argv.insert(0, "thumbnail")
    root = parser()
    args = root.parse_args(argv)
    if args.command == "tui":
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            print("TUI 需要交互终端；脚本中请使用 compress / thumbnail / transcribe 子命令", file=sys.stderr)
            return 1
        from .tui import VideoApp

        VideoApp(source=args.input, output=args.output).run()
        return 0
    if args.command == "compress" and not args.thumbnail and (args.watermark_text or args.font):
        root.error("--watermark-text 和 --font 用于缩略图，请同时指定 --thumbnail")
    state = args.output.resolve() / ".video-watermaker" if args.command != "models" else None
    if args.command == "status":
        database = state / "queue.sqlite3"
        if not database.is_file():
            print(f"尚无队列：{args.output}", file=sys.stderr)
            return 1
        store = None
        try:
            store = Store(database)
            _summary(store)
            print(f"队列：{database}\n日志：{state / 'run.log'}")
            return 0
        except (OSError, sqlite3.Error) as exc:
            print(f"无法读取队列：{exc}", file=sys.stderr)
            return 1
        finally:
            if store is not None:
                store.close()
    stop, interrupted = threading.Event(), threading.Event()

    def interrupt(signum, frame):
        interrupted.set()
        stop.set()

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    configured = None
    try:
        if args.command == "models":
            from .speech_client import SpeechClient

            with SpeechClient(stop, online=True) as client:
                client.request("prepare", {"model": args.model})
            print("语音模型准备完成，可离线转写")
            return 0
        state.mkdir(parents=True, exist_ok=True)
        configured = _logging(state)
        return _execute(args, state, stop, interrupted, progress)
    except (Cancelled, KeyboardInterrupt):
        print("已中断；重复原命令即可续跑", file=sys.stderr)
        return 130
    except (OSError, MediaError, sqlite3.Error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if configured:
            package, handlers = configured
            for handler in handlers:
                package.removeHandler(handler)
                handler.close()
