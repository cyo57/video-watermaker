"""Bounded pipe consumption, progress reporting and process-group cancellation."""

import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path


class MediaError(Exception):
    pass


class Cancelled(Exception):
    pass


def check_cancelled(stop: threading.Event):
    if stop.is_set():
        raise Cancelled("任务已中断，可重复原命令续跑")


def _terminate(process: subprocess.Popen):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        except ProcessLookupError:
            process.wait()


def run(
    args: list[str],
    stop: threading.Event,
    *,
    timeout: float | None = 60,
    progress: Callable[[float], None] | None = None,
    capture_limit: int = 4 * 1024 * 1024,
) -> bytes:
    check_cancelled(stop)
    launcher = str(Path(__file__).with_name("_child.py"))
    command = [sys.executable, launcher, str(os.getpid()), *map(str, args)]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    output = bytearray()
    errors = bytearray()
    pending = bytearray()
    started = last_advance = time.monotonic()
    last_seconds = -1.0
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                check_cancelled(stop)
                now = time.monotonic()
                if timeout is not None and now - started > timeout:
                    raise MediaError(f"媒体命令超过 {timeout:g} 秒：{args[0]}")
                if progress is not None and now - last_advance > 180:
                    raise MediaError("编码进度超过 180 秒未推进")
                for key, _ in selector.select(timeout=0.2):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stderr":
                        errors.extend(chunk)
                        del errors[:-65536]
                    elif progress is None:
                        output.extend(chunk)
                        if len(output) > capture_limit:
                            raise MediaError("媒体命令返回数据超过内存上限")
                    else:
                        pending.extend(chunk)
                        while b"\n" in pending:
                            line, _, remainder = pending.partition(b"\n")
                            pending[:] = remainder
                            if line.startswith(b"out_time_us="):
                                try:
                                    seconds = int(line.partition(b"=")[2]) / 1_000_000
                                except ValueError:
                                    continue
                                if seconds > last_seconds:
                                    last_advance = now
                                    last_seconds = seconds
                                progress(max(0, seconds))
                        if len(pending) > 65536:
                            raise MediaError("无效的 FFmpeg 进度输出")
            check_cancelled(stop)
            code = process.wait(timeout=5)
            if code:
                detail = errors.decode("utf-8", errors="replace").strip()
                raise MediaError(f"{Path(args[0]).name} 退出码 {code}：{detail[-6000:]}")
            return bytes(output)
    finally:
        _terminate(process)
        process.stdout.close()
        process.stderr.close()
