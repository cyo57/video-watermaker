"""Bounded scheduling and publication of verified outputs."""

import errno
import fcntl
import hashlib
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .media import EncoderPolicy, compress, probe
from .process import Cancelled, MediaError, check_cancelled
from .store import Store, Task
from .thumbnail import create_thumbnail

logger = logging.getLogger(__name__)


class RunnerLock:
    def __init__(self, path: Path):
        self.path = path
        self.file = None

    def __enter__(self):
        self.file = self.path.open("a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.file.close()
            raise MediaError("已有进程正在处理此输出队列；可用 status 查看状态") from exc
        return self

    def __exit__(self, *args):
        self.file.close()


def scratch_root(database: Path) -> Path:
    key = hashlib.sha256(str(database.resolve()).encode()).hexdigest()[:20]
    return Path("/tmp") / f"video-watermaker-{os.getuid()}-{key}"


def reset_scratch(path: Path):
    if path.is_symlink():
        raise MediaError(f"临时工作区不能是符号链接：{path}")
    if path.exists():
        if path.stat().st_uid != os.getuid():
            raise MediaError(f"临时工作区不属于当前用户：{path}")
        shutil.rmtree(path)
    path.mkdir(mode=0o700)


def _unchanged(task: Task):
    snapshot = task.source.stat()
    if (snapshot.st_size, snapshot.st_mtime_ns) != (task.source_size, task.source_mtime):
        raise MediaError("输入文件在排队或处理期间改变，请重新导入")


def publish(artifact: Path, task: Task, store: Store, stop: threading.Event, *, target: Path | None = None):
    speech = target is not None
    target = target or task.target
    check_cancelled(stop)
    _unchanged(task)
    target.parent.mkdir(parents=True, exist_ok=True)
    output_root = store.path.parent.parent.resolve()
    if not target.parent.resolve().is_relative_to(output_root):
        raise MediaError("输出子目录符号链接指向了输出目录之外")
    # /tmp may be another filesystem. An exclusive create prevents clobbering,
    # while the inode journal makes interrupted copies identifiable on recovery.
    with artifact.open("rb") as source, target.open("xb") as destination:
        identity = os.fstat(destination.fileno())
        try:
            if speech:
                store.artifact_publishing(task.id, target, identity)
            else:
                store.publishing(task, identity)
        except Exception:
            target.unlink()
            raise
        while chunk := source.read(2 * 1024 * 1024):
            check_cancelled(stop)
            destination.write(chunk)
        destination.flush()
        os.fsync(destination.fileno())
    directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    check_cancelled(stop)
    _unchanged(task)
    if target.stat().st_size != artifact.stat().st_size:
        raise MediaError("成品发布失败：大小不一致")
    if speech:
        store.artifact_complete(task.id, target)
    else:
        store.succeed(task)


def process_task(
    task: Task,
    database: Path,
    scratch: Path,
    policy: EncoderPolicy | None,
    threads: int,
    stop: threading.Event,
    progress,
    speech_client=None,
) -> str:
    store = Store(database)
    try:
        check_cancelled(stop)
        _unchanged(task)
        with tempfile.TemporaryDirectory(prefix=f"task-{task.id}-", dir=scratch) as work:
            if task.kind == "transcribe":
                from .speech import transcribe_task

                transcribe_task(task, store, Path(work), stop, progress, speech_client)
                progress(1)
                return "succeeded"
            info = probe(task.source, stop)
            artifact = Path(work) / ("video.mp4" if task.kind == "compress" else "thumbnail.jpg")
            if task.kind == "compress":
                compress(task.source, artifact, info, policy, threads, stop, progress)
            else:
                create_thumbnail(task.source, artifact, info, task.params, threads, stop, progress)
            publish(artifact, task, store, stop)
        progress(1)
        return "succeeded"
    except Cancelled as exc:
        store.finish_error(task, str(exc), cancelled=True)
        return "pending"
    except Exception as exc:  # noqa: BLE001 -- isolate each queued job, persist unexpected failures too.
        store.finish_error(task, str(exc))
        logger.error("%s 失败：%s：%s", task.kind, task.source, exc)
        if (isinstance(exc, OSError) and exc.errno == errno.ENOSPC) or "no space left" in str(exc).lower():
            logger.error("空间不足，停止派发新任务；释放空间后用 --retry-failed 续跑")
            stop.set()
        return "failed"
    finally:
        store.close()


def run_queue(
    store: Store,
    batch: str,
    workers: int,
    stop: threading.Event,
    worker,
    show_progress: bool = True,
    *,
    speech: bool | None = None,
    on_progress=None,
):
    """At most `workers` futures exist; SQLite holds the rest of the queue."""
    active = {}
    fractions = {}
    lock = threading.Lock()
    last_display = 0.0
    last_event = 0.0
    tty = show_progress and sys.stderr.isatty()

    def update(task_id, fraction):
        with lock:
            fractions[task_id] = fraction

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video")
    try:
        while not stop.is_set():
            for task in store.claim(batch, workers - len(active), speech=speech):
                callback = lambda fraction, task_id=task.id: update(task_id, fraction)
                active[executor.submit(worker, task, callback)] = task
            if not active:
                break
            done, _ = wait(active, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                future.result()
                with lock:
                    fractions.pop(task.id, None)
            now = time.monotonic()
            if on_progress is not None and now - last_event >= 0.5:
                with lock:
                    current = [
                        {
                            "id": task.id,
                            "name": task.source.name,
                            "kind": task.kind,
                            "fraction": max(0.0, min(1.0, fractions.get(task.id, 0.0))),
                        }
                        for task in active.values()
                    ]
                on_progress({"active": current, "counts": store.counts(batch)})
                last_event = now
            if show_progress and now - last_display >= (0.5 if tty else 10):
                counts = store.counts(batch)
                line = (
                    f"完成 {counts['succeeded']} | 失败 {counts['failed']} | "
                    f"运行 {counts['running']} | 等待 {counts['pending']}"
                )
                if tty:
                    with lock:
                        current = " | ".join(
                            f"{t.source.name[:30]} {fractions.get(t.id, 0):.0%}"
                            for t in list(active.values())[:3]
                        )
                    print("\r\033[K" + line + " | " + current, end="", file=sys.stderr, flush=True)
                else:
                    logger.info(line)
                last_display = now
    except BaseException:
        stop.set()
        raise
    finally:
        executor.shutdown(wait=True)
        if tty:
            print("\r\033[K", end="", file=sys.stderr, flush=True)
    # Surface worker infrastructure errors even when cancellation ended the loop.
    for future in active:
        future.result()
