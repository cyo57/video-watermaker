import errno
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from video_watermaker import runner
from video_watermaker.discovery import discover, output_path
from video_watermaker.process import MediaError
from video_watermaker.store import ConflictError, Store


@pytest.fixture
def store(tmp_path):
    state = tmp_path / "out" / ".video-watermaker"
    state.mkdir(parents=True)
    result = Store(state / "queue.sqlite3")
    yield result
    result.close()


def enqueue(store, tmp_path, *, name="input.mp4", params=None, batch="batch"):
    source = tmp_path / name
    source.parent.mkdir(exist_ok=True, parents=True)
    source.write_bytes(b"source")
    target = store.path.parent.parent / (name + ".compressed.mp4")
    store.enqueue(source, target, "compress", params or {}, batch)
    store.commit()
    return source, target


def test_discovery_excludes_output_symlinks_and_nonvideo(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    output = source / "generated"
    output.mkdir()
    (source / "movie.MP4").touch()
    (source / "notes.txt").touch()
    (output / "recursive.mp4").touch()
    (source / "nested").mkdir()
    (source / "nested" / "movie.mkv").touch()
    (source / "loop").symlink_to(source, target_is_directory=True)
    (source / "alias.mp4").symlink_to(source / "movie.MP4")
    errors = []
    found = list(discover(source, output, lambda *args: errors.append(args)))
    assert {p.relative_to(source).as_posix() for p in found} == {"movie.MP4", "nested/movie.mkv"}
    assert errors == []
    assert output_path(source / "movie.MP4", source, output, "compress").name == "movie.MP4.compressed.mp4"


def test_ten_thousand_tasks_bounded_dispatch_resume_and_retry(tmp_path, store, monkeypatch):
    source = tmp_path / "library"
    source.mkdir()
    output = store.path.parent.parent
    for index in range(10_000):
        directory = source / str(index // 100)
        directory.mkdir(exist_ok=True)
        (directory / f"{index}.mp4").touch()

    def scan(batch, retry=False):
        count = 0
        for path in discover(source, output, lambda *args: pytest.fail(str(args))):
            store.enqueue(path, output_path(path, source, output, "compress"), "compress", {}, batch, retry)
            count += 1
            if count % 100 == 0:
                store.commit()
        store.commit()
        return count

    assert scan("first") == 10_000
    lock = threading.Lock()
    outstanding = peak = 0

    class MeasuredExecutor(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            nonlocal outstanding, peak
            with lock:
                outstanding += 1
                peak = max(peak, outstanding)
            future = super().submit(*args, **kwargs)

            def complete(_):
                nonlocal outstanding
                with lock:
                    outstanding -= 1

            future.add_done_callback(complete)
            return future

    monkeypatch.setattr(runner, "ThreadPoolExecutor", MeasuredExecutor)
    inject_failure = True

    def fake_work(task, progress):
        worker_store = Store(store.path)
        try:
            if inject_failure and task.id % 1000 == 0:
                worker_store.finish_error(task, "injected damaged video")
                return "failed"
            task.target.parent.mkdir(parents=True, exist_ok=True)
            task.target.write_bytes(b"synthetic verified artifact")
            worker_store.succeed(task)
            return "succeeded"
        finally:
            worker_store.close()

    runner.run_queue(store, "first", 4, threading.Event(), fake_work, show_progress=False)
    assert peak <= 4
    assert outstanding == 0
    assert store.counts("first") == {"pending": 0, "running": 0, "succeeded": 9990, "failed": 10}
    assert scan("resume") == 10_000
    assert store.counts("resume")["pending"] == 0
    assert scan("retry", retry=True) == 10_000
    assert store.counts("retry")["pending"] == 10
    inject_failure = False
    runner.run_queue(store, "retry", 4, threading.Event(), fake_work, show_progress=False)
    assert store.counts("retry")["succeeded"] == 10_000
    assert store.db.execute("SELECT SUM(attempts) FROM tasks").fetchone()[0] == 10_010


def test_recovery_removes_only_owned_partial_output(store, tmp_path):
    source, target = enqueue(store, tmp_path)
    task = store.claim("batch", 1)[0]
    target.write_bytes(b"partial")
    store.publishing(task, target.stat())
    store.recover()
    assert not target.exists()
    assert store.counts()["pending"] == 1
    task = store.claim("batch", 1)[0]
    target.write_bytes(b"someone else's file")
    store.recover()
    assert target.read_bytes() == b"someone else's file"
    with pytest.raises(ConflictError):
        store.enqueue(source, target, "compress", {}, "next")


def test_success_skip_missing_output_requeue_and_parameter_conflict(store, tmp_path):
    source, target = enqueue(store, tmp_path)
    task = store.claim("batch", 1)[0]
    target.write_bytes(b"complete")
    store.succeed(task)
    assert store.enqueue(source, target, "compress", {}, "next") == "skipped"
    store.commit()
    with pytest.raises(ConflictError):
        store.enqueue(source, target, "compress", {"changed": True}, "third")
    assert target.read_bytes() == b"complete"
    target.unlink()
    assert store.enqueue(source, target, "compress", {}, "third") == "pending"


def test_other_input_with_same_output_is_conflict(store, tmp_path):
    _, target = enqueue(store, tmp_path)
    other = tmp_path / "other.mp4"
    other.touch()
    with pytest.raises(ConflictError):
        store.enqueue(other, target, "compress", {}, "next")


def test_runner_lock_rejects_second_runner(tmp_path):
    with (
        runner.RunnerLock(tmp_path / "runner.lock"),
        pytest.raises(MediaError, match="已有进程"),
        runner.RunnerLock(tmp_path / "runner.lock"),
    ):
        pytest.fail("Second runner acquired lock")


@pytest.mark.parametrize("failure", ["disk_full", "cancel", "source_changed", "target_exists"])
def test_publication_failures_do_not_become_success(store, tmp_path, monkeypatch, failure):
    _source, target = enqueue(store, tmp_path)
    task = store.claim("batch", 1)[0]
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    stop = threading.Event()
    monkeypatch.setattr(runner, "probe", lambda *args: object())

    def encode(source, artifact, *args):
        artifact.write_bytes(b"encoded bytes")
        if failure == "source_changed":
            source.write_bytes(b"source was changed while encoding")
        if failure == "target_exists":
            target.write_bytes(b"external output")

    monkeypatch.setattr(runner, "compress", encode)
    if failure == "disk_full":

        def disk_full(_):
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(os, "fsync", disk_full)
    if failure == "cancel":
        original = Store.publishing

        def cancel(self, task, identity):
            original(self, task, identity)
            stop.set()

        monkeypatch.setattr(Store, "publishing", cancel)
    result = runner.process_task(task, store.path, scratch, None, 1, stop, lambda _: None)
    assert result == ("pending" if failure == "cancel" else "failed")
    assert store.counts()["succeeded"] == 0
    if failure == "target_exists":
        assert target.read_bytes() == b"external output"
    else:
        assert not target.exists()
    assert not list(scratch.iterdir())
    if failure == "disk_full":
        assert stop.is_set()


def test_no_dispatch_after_cancel(store, tmp_path):
    enqueue(store, tmp_path)
    stop = threading.Event()
    stop.set()
    runner.run_queue(store, "batch", 2, stop, lambda *args: pytest.fail("Dispatched after cancel"), False)
    assert store.counts()["pending"] == 1


def test_changed_input_requeues_failed_job_without_retry(store, tmp_path):
    source, target = enqueue(store, tmp_path)
    task = store.claim("batch", 1)[0]
    store.finish_error(task, "bad input")
    source.write_bytes(b"replaced with valid video")
    assert store.enqueue(source, target, "compress", {}, "next") == "pending"
