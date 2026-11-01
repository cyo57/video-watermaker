import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_watermaker import speech_client, speech_worker
from video_watermaker.process import Cancelled, MediaError
from video_watermaker.speech_client import InferenceError, SpeechClient, worker_environment


def test_ipc_reuses_process_and_redacts_tokens():
    with SpeechClient(threading.Event()) as client:
        with pytest.raises(InferenceError, match="无效"):
            client.request("invalid", {})
        process = client.process
        with pytest.raises(InferenceError):
            client.request("invalid", {})
        assert client.process is process
        assert process.poll() is None
    assert process.poll() is not None
    assert speech_client.redact("secret hf_abc123 rest") == "secret [HF_TOKEN] rest"


@pytest.mark.parametrize("mode", ["cancel", "crash", "oversized"])
def test_ipc_bounded_stderr_cancel_and_crash_stop_dispatch(monkeypatch, mode):
    original = subprocess.Popen
    spawned = []
    behavior = {
        "cancel": "import time; time.sleep(30)",
        "crash": "raise SystemExit(5)",
        "oversized": "sys.stdout.write('x'*2000000);sys.stdout.flush()",
    }[mode]

    def fake_worker(command, **kwargs):
        program = (
            "import sys;sys.stdin.readline();sys.stderr.write('e'*200000);sys.stderr.flush();" + behavior
        )
        child = original([*command[:4], "-c", program], **kwargs)
        spawned.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", fake_worker)
    stop = threading.Event()
    timer = threading.Timer(0.5, stop.set) if mode == "cancel" else None
    if timer:
        timer.start()
    try:
        with SpeechClient(stop) as client, pytest.raises(Cancelled if timer else MediaError):
            client.request("check", {})
    finally:
        if timer:
            timer.join()
    assert stop.is_set()
    assert spawned[0].poll() is not None


def test_speech_worker_dies_when_parent_is_killed():
    script = """
import threading,time
from video_watermaker.speech_client import SpeechClient,InferenceError
client=SpeechClient(threading.Event())
try:
    client.request('invalid',{})
except InferenceError:
    pass
print(client.process.pid,flush=True)
time.sleep(30)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd="/tmp",
        env=worker_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child = None
    try:
        # communicate cannot be used: the parent must remain alive while we capture the worker PID.
        import selectors

        with selectors.DefaultSelector() as selector:
            selector.register(parent.stdout, selectors.EVENT_READ)
            assert selector.select(10), "Parent never reported speech worker PID"
        child = int(parent.stdout.readline())
        parent.kill()
        parent.communicate(timeout=10)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                status = Path(f"/proc/{child}/status").read_text()
            except FileNotFoundError:
                break
            if "\nState:\tZ" in status:
                break
            time.sleep(0.02)
        else:
            pytest.fail("Speech worker survived SIGKILL of its parent")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.communicate()
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_models_reused_with_nonoverlapping_gpu_residency(monkeypatch):
    events = []

    class Whisper:
        supported_languages = ("zh", "en")

        def __init__(self, *args, **kwargs):
            events.append("asr_init")
            self.model = SimpleNamespace(
                unload_model=lambda **kwargs: events.append("asr_off"),
                load_model=lambda: events.append("asr_on"),
            )

        def detect_language(self, *args):
            events.append("asr_probe")

    class Pipeline:
        @classmethod
        def from_pretrained(cls, path):
            events.append("diar_init")
            return cls()

        def to(self, device):
            events.append("diar_" + device)

    torch = SimpleNamespace(
        set_num_threads=lambda _: None,
        device=lambda x: x,
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
    )
    np = SimpleNamespace(zeros=lambda *args, **kwargs: [], float32=None)
    monkeypatch.setattr(speech_worker, "dependencies", lambda: (np, torch, Whisper, Pipeline))
    engine = speech_worker.Engine(
        {"asr": {"path": "asr"}, "diarization": {"path": "diar"}},
        {"device": "auto", "language": "zh"},
        lambda _: None,
    )
    engine.activate_asr()
    engine.activate_diarizer()
    engine.activate_asr()
    assert events.count("asr_init") == events.count("diar_init") == 1
    assert events.index("asr_off") < events.index("diar_cuda")
    assert events[-2:] == ["diar_cpu", "asr_on"]
    assert engine.diarizer.segmentation_batch_size == engine.diarizer.embedding_batch_size == 4


@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_gpu_oom_fallback_only_in_auto_mode(requested):
    engine = speech_worker.Engine.__new__(speech_worker.Engine)
    engine.device = "cuda"
    engine.params = {"device": requested}
    engine.asr = None
    engine.asr_loaded = False
    engine.diarizer = None
    engine.torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
    notices = []
    engine.emit = notices.append
    calls = []

    def operation():
        calls.append(engine.device)
        if engine.device == "cuda":
            raise RuntimeError("CUDA out of memory")
        return "ok"

    if requested == "auto":
        assert engine.perform(operation) == "ok"
        assert calls == ["cuda", "cpu"]
        assert len(notices) == 1
    else:
        with pytest.raises(RuntimeError, match="out of memory"):
            engine.perform(operation)
        assert calls == ["cuda"] and not notices


def test_base_commands_import_no_optional_ml_packages():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys,json
from video_watermaker.cli import parser
parser().parse_args(['compress','A'])
from video_watermaker.speech import exports
print(json.dumps(sorted(set(sys.modules) & {'numpy','torch','faster_whisper','pyannote'})))
""",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []


def test_asr_preserves_long_silence_timeline(monkeypatch):
    engine = speech_worker.Engine.__new__(speech_worker.Engine)
    engine.device = "cuda"
    engine.activate_asr = lambda: None
    samples = list(range(302))
    monkeypatch.setattr(speech_worker, "SAMPLE_RATE", 1)
    monkeypatch.setattr(speech_worker, "speech_windows", lambda _: [(0, 159), (159, 302)])
    calls = []

    def transcribe(audio, **options):
        calls.append(audio)
        if audio[0] == 0:
            return iter([]), SimpleNamespace(language="zh")
        word = SimpleNamespace(start=139.64, end=139.84, word="也")
        return iter([SimpleNamespace(words=[word])]), SimpleNamespace(language="zh")

    engine.asr = SimpleNamespace(transcribe=transcribe)
    result = engine.transcribe_chunk(samples, "zh")
    assert calls == [samples[:159], samples[159:]]
    assert result["words"][0]["start"] == pytest.approx(298.64)
    assert result["words"][0]["end"] == pytest.approx(298.84)
    assert result["words"][0]["text"] == "也"


def test_asr_skips_silent_chunk_without_loading_model(monkeypatch):
    engine = speech_worker.Engine.__new__(speech_worker.Engine)
    engine.device = "cpu"
    engine.activate_asr = lambda: pytest.fail("Silent chunk should not run Whisper")
    monkeypatch.setattr(speech_worker, "speech_windows", lambda _: [])
    assert engine.transcribe_chunk(object(), "auto") == {"words": [], "language": "auto", "device": "cpu"}


def test_long_pause_windows_preserve_all_original_samples(monkeypatch):
    clips = [{"start": 10, "end": 20}, {"start": 21, "end": 30}, {"start": 200, "end": 210}]
    vad = SimpleNamespace(VadOptions=lambda **kwargs: kwargs, get_speech_timestamps=lambda *args: clips)
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    monkeypatch.setattr(speech_worker, "SAMPLE_RATE", 1)
    assert speech_worker.speech_windows(list(range(300))) == [(0, 115), (115, 300)]
    clips[:] = clips[:2]
    assert speech_worker.speech_windows(list(range(300))) == [(0, 300)]
    clips.clear()
    assert speech_worker.speech_windows(list(range(300))) == []
