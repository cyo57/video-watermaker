import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def make_clip(tmp_path):
    def make(name="sample.mp4", size="320x180", rate="24", duration="1", audio=False, extra=()):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={size}:rate={rate}",
        ]
        if audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
        command += [
            "-t",
            duration,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
        ]
        if audio:
            command += ["-c:a", "aac", "-b:a", "96k"]
        command += [*extra, str(target)]
        subprocess.run(command, check=True, capture_output=True, timeout=30)
        return target

    return make


@pytest.fixture
def cli():
    def invoke(*args, timeout=60):
        return subprocess.run(
            [sys.executable, str(ROOT / "main.py"), *map(str, args)],
            cwd="/tmp",
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    return invoke
