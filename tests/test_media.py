import os
import sqlite3
import subprocess
import sys
import threading
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from video_watermaker import media
from video_watermaker.media import EncoderPolicy, compress, probe
from video_watermaker.process import MediaError


@pytest.mark.parametrize(
    "size,rate,expected",
    [
        ("1920x1080", "60", (1280, 720)),
        ("1080x1920", "60", (720, 1280)),
        ("640x360", "24000/1001", (640, 360)),
    ],
)
def test_cpu_compression_real_dimensions_fps_audio(make_clip, tmp_path, size, rate, expected):
    source = make_clip(size=size, rate=rate, audio=True, duration="2")
    target = tmp_path / "compressed.mp4"
    stop = threading.Event()
    info = probe(source, stop)
    progress = []
    compress(source, target, info, EncoderPolicy("cpu", stop), 2, stop, progress.append)
    result = probe(target, stop)
    assert (result.width, result.height) == expected
    assert result.fps == pytest.approx(min(float(Fraction(rate)), 30), abs=0.01)
    assert result.video["codec_name"] == "h264"
    assert result.video["pix_fmt"] == "yuv420p"
    assert result.audio["codec_name"] == "aac"
    assert result.audio["channels"] <= 2
    assert result.video["sample_aspect_ratio"] == "1:1"
    assert progress and max(progress) > 0
    assert abs(result.duration - info.duration) < 0.15
    # VBV permits an initial buffer; check its duration-dependent envelope, not a per-frame CBR.
    assert int(result.video["bit_rate"]) < 1_500_000 + 3_000_000 / result.duration
    data = target.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")


def test_nvenc_real_and_no_audio(make_clip, tmp_path):
    stop = threading.Event()
    try:
        policy = EncoderPolicy("nvenc", stop)
    except MediaError as exc:
        if "invalid param" in str(exc).lower():
            raise
        pytest.skip("NVENC device unavailable")
    source = make_clip(size="1920x1080", rate="60", duration="3")
    output = tmp_path / "nvenc.mp4"
    compress(source, output, probe(source, stop), policy, 2, stop, lambda _: None)
    result = probe(output, stop)
    assert (result.width, result.height, result.fps) == (1280, 720, 30)
    assert result.audio is None


def test_rotation_metadata_and_non_square_pixels(make_clip, tmp_path):
    source = make_clip(size="640x360", extra=("-vf", "setsar=4/3"))
    rotated = tmp_path / "rotated.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(source), "-c", "copy", str(rotated)],
        check=True,
        capture_output=True,
    )
    stop = threading.Event()
    info = probe(rotated, stop)
    assert info.display_height > info.display_width
    output = tmp_path / "normal.mp4"
    compress(rotated, output, info, EncoderPolicy("cpu", stop), 2, stop, lambda _: None)
    result = probe(output, stop)
    assert result.width <= 720 and result.height <= 1280
    assert result.display_width / result.display_height == pytest.approx(
        info.display_width / info.display_height, abs=0.005
    )
    assert result.video["sample_aspect_ratio"] == "1:1"


def test_variable_frame_rate_capped(make_clip, tmp_path):
    source = make_clip(
        rate="60", duration="2", extra=("-vf", "select='if(lt(t,1),1,not(mod(n,4)))'", "-fps_mode", "vfr")
    )
    stop = threading.Event()
    output = tmp_path / "vfr.mp4"
    compress(source, output, probe(source, stop), EncoderPolicy("cpu", stop), 2, stop, lambda _: None)
    result = probe(output, stop)
    assert result.fps <= 30.01
    assert float(Fraction(result.video["r_frame_rate"])) <= 30.01


def test_hdr_tagged_input_tone_maps_to_sdr(make_clip, tmp_path):
    source = make_clip(
        extra=("-vf", "setparams=color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc")
    )
    stop = threading.Event()
    info = probe(source, stop)
    assert info.hdr
    output = tmp_path / "sdr.mp4"
    compress(source, output, info, EncoderPolicy("cpu", stop), 2, stop, lambda _: None)
    result = probe(output, stop)
    assert result.video["color_transfer"] == "bt709"
    assert result.video["color_primaries"] == "bt709"
    assert result.video["pix_fmt"] == "yuv420p"


def test_tiny_video_auto_uses_cpu_without_disabling_gpu(make_clip, tmp_path):
    stop = threading.Event()
    policy = EncoderPolicy("auto", stop)
    before = policy.selected
    source = make_clip(size="64x64")
    output = tmp_path / "tiny.mp4"
    compress(source, output, probe(source, stop), policy, 2, stop, lambda _: None)
    result = probe(output, stop)
    assert (result.width, result.height) == (64, 64)
    assert policy.selected == before


def test_auto_falls_back_but_forced_nvenc_fails(monkeypatch):
    def unavailable(*args, **kwargs):
        raise MediaError("Cannot load libcuda.so.1")

    monkeypatch.setattr(media, "run", unavailable)
    stop = threading.Event()
    assert EncoderPolicy("auto", stop).selected == "cpu"
    with pytest.raises(MediaError, match="NVENC"):
        EncoderPolicy("nvenc", stop)


def test_runtime_hardware_failure_falls_back_to_cpu(make_clip, tmp_path, monkeypatch):
    source = make_clip()
    stop = threading.Event()
    real_run = media.run
    attempted = []

    def simulated_hardware_loss(args, *a, **kw):
        if "h264_nvenc" in args:
            if "lavfi" in args:
                return b""  # Startup probe succeeds, device fails when encoding the job.
            attempted.append("nvenc")
            raise MediaError("NVENC: CUDA device lost")
        if "libx264" in args:
            attempted.append("cpu")
        return real_run(args, *a, **kw)

    monkeypatch.setattr(media, "run", simulated_hardware_loss)
    policy = EncoderPolicy("auto", stop)
    output = tmp_path / "fallback.mp4"
    compress(source, output, probe(source, stop), policy, 2, stop, lambda _: None)
    assert attempted == ["nvenc", "cpu"]
    assert policy.selected == "cpu"
    assert probe(output, stop).video["codec_name"] == "h264"


@pytest.mark.parametrize("rotate,rows,cols", [(0, 2, 3), (90, 3, 2)])
def test_short_contact_sheet_geometry_and_watermark(make_clip, cli, tmp_path, rotate, rows, cols):
    source = make_clip(name="中文 很长的文件名字 短视频.mp4", size="160x90", duration="0.2")
    original = source.read_bytes()
    plain, marked = tmp_path / "plain", tmp_path / "marked"
    args = ["thumbnail", source, "--rows", rows, "--cols", cols, "--width", 600, "--rotate", rotate]
    first = cli(*args, "-o", plain)
    assert first.returncode == 0, first.stderr
    second = cli(*args, "-o", marked, "--watermark_text", "中文水印")
    assert second.returncode == 0, second.stderr
    with (
        Image.open(plain / (source.name + ".thumbnail.jpg")) as a,
        Image.open(marked / (source.name + ".thumbnail.jpg")) as b,
    ):
        ratio = 90 / 160 if rotate == 0 else 160 / 90
        assert a.size == (600, rows * round((600 // cols) * ratio) + 40)
        assert ImageChops.difference(a, b).getbbox() is not None
        # Every cell contains a decoded image, including very short clips.
        cell_h = (a.height - 40) // rows
        for row in range(rows):
            for col in range(cols):
                patch = a.crop(
                    (col * 600 // cols, 40 + row * cell_h, (col + 1) * 600 // cols, 40 + (row + 1) * cell_h)
                )
                assert any(high - low > 50 for low, high in patch.getextrema())
                patch.close()
    assert source.read_bytes() == original


def test_combined_cli_resume_bad_video_and_conflicting_parameters(make_clip, cli, tmp_path):
    source = make_clip(name="library/子目录/movie.mp4", duration="0.4")
    # Same stem, different extension: distinct artifacts, no overwrite.
    same_stem = source.with_suffix(".mkv")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(source), "-c", "copy", str(same_stem)],
        check=True,
        capture_output=True,
    )
    bad = source.parent / "broken.mp4"
    bad.write_bytes(b"this is not a video")
    output = tmp_path / "output"
    args = [
        "compress",
        source.parent.parent,
        "-o",
        output,
        "--encoder",
        "cpu",
        "--thumbnail",
        "--rows",
        1,
        "--cols",
        2,
        "--width",
        400,
        "--watermark-text",
        "测试",
    ]
    first = cli(*args)
    assert first.returncode == 1
    db = sqlite3.connect(output / ".video-watermaker" / "queue.sqlite3")
    try:
        assert dict(db.execute("SELECT status,COUNT(*) FROM tasks GROUP BY status")) == {
            "succeeded": 4,
            "failed": 2,
        }
        attempts = db.execute("SELECT SUM(attempts) FROM tasks").fetchone()[0]
        second = cli(*args)
        assert second.returncode == 1  # Known failed jobs remain visible until explicitly retried.
        assert db.execute("SELECT SUM(attempts) FROM tasks").fetchone()[0] == attempts
        third = cli(*args, "--retry-failed")
        assert third.returncode == 1
        assert db.execute("SELECT SUM(attempts) FROM tasks").fetchone()[0] == attempts + 2
        assert cli("status", "-o", output).returncode == 0
        before = (output / "子目录/movie.mp4.thumbnail.jpg").read_bytes()
        conflict = cli(*args, "--width", 600)
        assert conflict.returncode == 1 and "冲突" in conflict.stderr
        assert (output / "子目录/movie.mp4.thumbnail.jpg").read_bytes() == before
    finally:
        db.close()


def test_legacy_cli_and_no_import_side_effects(make_clip, cli, tmp_path):
    source = make_clip(duration="0.2")
    result = cli(source, "-o", tmp_path / "old", "--rows", 1, "--cols", 1, "--width", 200)
    assert result.returncode == 0, result.stderr
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-c", "import main"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
        capture_output=True,
    )
    assert not (tmp_path / "video_watermaker.log").exists()


def test_compression_conflict_does_not_block_thumbnail(make_clip, cli, tmp_path):
    source = make_clip(duration="0.3")
    output = tmp_path / "combined"
    output.mkdir()
    existing = output / (source.name + ".compressed.mp4")
    existing.write_bytes(b"external result must be preserved")
    result = cli(
        "compress",
        source,
        "-o",
        output,
        "--thumbnail",
        "--rows",
        1,
        "--cols",
        1,
        "--width",
        200,
        "--encoder",
        "cpu",
    )
    assert result.returncode == 1
    assert "冲突" in result.stderr
    assert existing.read_bytes() == b"external result must be preserved"
    with Image.open(output / (source.name + ".thumbnail.jpg")) as image:
        image.verify()


def test_status_reports_corrupt_database_without_traceback(cli, tmp_path):
    state = tmp_path / ".video-watermaker"
    state.mkdir()
    (state / "queue.sqlite3").write_bytes(b"not a database")
    result = cli("status", "-o", tmp_path)
    assert result.returncode == 1
    assert "无法读取队列" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "args",
    [
        ("thumbnail", "missing.mp4", "--rows", "0"),
        ("compress", "missing.mp4", "--workers", "0"),
        ("thumbnail", "missing.mp4", "--quality", "101"),
        ("compress", "missing.mp4", "--watermark-text", "text"),
    ],
)
def test_invalid_options_return_usage_error(cli, args):
    result = cli(*args)
    assert result.returncode == 2
