import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from video_watermaker.process import Cancelled, MediaError, run

ROOT = Path(__file__).resolve().parents[1]


def test_pipe_consumption_does_not_deadlock_and_is_bounded():
    output = run(
        [sys.executable, "-c", "import sys;sys.stderr.write('x'*200000);print('done')"],
        threading.Event(),
        timeout=10,
    )
    assert output == b"done\n"
    with pytest.raises(MediaError, match="内存上限"):
        run([sys.executable, "-c", "print('x'*100000)"], threading.Event(), capture_limit=100)


def test_timeout_and_cancellation_reap_process():
    with pytest.raises(MediaError, match="超过"):
        run([sys.executable, "-c", "import time;time.sleep(30)"], threading.Event(), timeout=0.2)
    stop = threading.Event()
    timer = threading.Timer(0.2, stop.set)
    timer.start()
    try:
        with pytest.raises(Cancelled):
            run([sys.executable, "-c", "import time;time.sleep(30)"], stop, timeout=5)
    finally:
        timer.join()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGKILL])
def test_cli_interrupt_and_hard_kill_resume(make_clip, cli, tmp_path, signum):
    source = make_clip(size="1920x1080", rate="60", duration="4")
    output = tmp_path / "out"
    args = ["compress", str(source), "-o", str(output), "--encoder", "cpu", "--workers", "1"]
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "main.py"), *args],
        cwd="/tmp",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    child = None
    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline and process.poll() is None:
            for children in Path(f"/proc/{process.pid}/task").glob("*/children"):
                try:
                    child_pids = children.read_text().split()
                except FileNotFoundError:
                    continue
                for pid in child_pids:
                    try:
                        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
                    except FileNotFoundError:
                        continue
                    if cmdline.startswith(b"ffmpeg\x00") and b"libx264" in cmdline:
                        child = int(pid)
                        break
            if child:
                break
            time.sleep(0.02)
        assert child is not None, process.communicate(timeout=5)
        process.send_signal(signum)
        process.communicate(timeout=10)
        assert process.returncode == (130 if signum == signal.SIGINT else -signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = Path(f"/proc/{child}/status")
            try:
                state = status.read_text()
            except FileNotFoundError:
                break
            if "\nState:\tZ" in state:
                break
            time.sleep(0.02)
        else:
            pytest.fail("Encoder survived parent termination")
        result = cli(*args)
        assert result.returncode == 0, result.stderr
        db = sqlite3.connect(output / ".video-watermaker" / "queue.sqlite3")
        try:
            assert db.execute("SELECT status,attempts FROM tasks").fetchone() == ("succeeded", 2)
        finally:
            db.close()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
