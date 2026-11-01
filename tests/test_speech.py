import json
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from video_watermaker import runner, speech
from video_watermaker.process import Cancelled, MediaError
from video_watermaker.store import ConflictError, Store


@pytest.fixture
def speech_job(tmp_path):
    source = tmp_path / "library" / "lesson.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")
    state = tmp_path / "out" / ".video-watermaker"
    state.mkdir(parents=True)
    store = Store(state / "queue.sqlite3")
    target = state.parent / "lesson.mp4.transcript.json"
    params = {
        "model": "large-v3-turbo",
        "asr_revision": "asr123",
        "diarization_revision": "diar123",
        "language": "zh",
        "device": "auto",
        "speakers": None,
        "revision": 1,
    }
    store.enqueue(source, target, "transcribe", params, "batch")
    store.commit()
    task = store.claim("batch", 1)[0]
    work = tmp_path / "scratch"
    work.mkdir()
    yield store, task, work
    store.close()


def example(params):
    return speech.document(
        params,
        5,
        [
            {"start": 0.1, "end": 1.2, "text": "你好。", "speaker": "A", "overlap": False},
            {"start": 2, "end": 4, "text": "你好，今天聊什么？", "speaker": "B", "overlap": True},
        ],
    )


def test_global_labels_returning_speaker_overlap_unknown_and_sentences():
    words = [
        {"start": 0, "end": 1, "text": "你好。"},
        {"start": 301, "end": 302, "text": "换我说。"},
        {"start": 602, "end": 603, "text": "又是我。"},
        {"start": 700, "end": 701, "text": "未知。"},
    ]
    turns = [
        {"start": 0, "end": 2, "speaker": "speaker_8"},
        {"start": 300, "end": 304, "speaker": "speaker_0"},
        {"start": 601, "end": 604, "speaker": "speaker_8"},
    ]
    diarization = {
        "exclusive": turns,
        "turns": [
            *turns,
            {"start": 302, "end": 302.5, "speaker": "speaker_8"},
            {"start": 602.5, "end": 603, "speaker": "speaker_0"},
        ],
    }
    segments = speech.align_speakers(words, diarization)
    assert [segment["speaker"] for segment in segments] == ["A", "B", "A", None]
    assert [segment["overlap"] for segment in segments] == [False, False, True, False]
    assert speech.speaker_name(26) == "AA"
    assert speech.speaker_name(51) == "AZ"


def test_chunk_join_and_audio_offset():
    windows = list(speech.chunk_windows(601))
    assert windows == [(0, 0, 300, 0, 302), (1, 300, 600, 298, 601), (2, 600, 601, 598, 601)]
    first = speech.core_words(
        [
            {"start": 298, "end": 299, "text": "前"},
            {"start": 299.5, "end": 300.5, "text": "边界"},
        ],
        0,
        0,
        300,
        3,
        604,
    )
    second = speech.core_words(
        [
            {"start": 0, "end": 1, "text": "前"},
            {"start": 1.5, "end": 2.5, "text": "边界"},
            {"start": 2.5, "end": 3.5, "text": "后"},
        ],
        298,
        300,
        600,
        3,
        604,
    )
    assert [word["text"] for word in first + second] == ["前", "边界", "后"]
    assert first[0]["start"] == 301
    assert second[0]["start"] == 302.5
    assert speech.core_words([{"start": 0, "end": 1, "text": "片头之前"}], 0, 0, 2, -3, 10) == []


@pytest.mark.parametrize("split_word", [False, True])
def test_overlap_join_keeps_boundary_word_despite_timestamp_drift(split_word):
    previous = [
        {"start": 299.4, "end": 299.722, "text": "地方"},
        {"start": 299.722, "end": 300.322, "text": "政府"},
        {"start": 300.322, "end": 300.5, "text": "的"},
    ]
    current = [
        {"start": 299.7, "end": 300.24, "text": "政府"},
        {"start": 300.24, "end": 300.48, "text": "的"},
        {"start": 300.48, "end": 301.1, "text": "眼中钉"},
    ]
    if split_word:
        current[:1] = [
            {"start": 299.7, "end": 299.97, "text": "政"},
            {"start": 299.97, "end": 300.24, "text": "府"},
        ]
    merged = speech.merge_chunk_words(previous, current, 300)
    assert "".join(word["text"] for word in merged) == "地方政府的眼中钉"


def test_overlap_join_does_not_confuse_repeated_words_or_silent_gap():
    previous = [
        {"start": 298, "end": 298.2, "text": "好"},
        {"start": 299.9, "end": 300.2, "text": "好"},
        {"start": 300.3, "end": 300.6, "text": "我们"},
    ]
    current = [
        {"start": 299.8, "end": 300.1, "text": "好"},
        {"start": 300.3, "end": 300.6, "text": "我们"},
        {"start": 300.7, "end": 301.2, "text": "继续"},
    ]
    assert "".join(w["text"] for w in speech.merge_chunk_words(previous, current, 300)) == "好好我们继续"
    first = [{"start": 10, "end": 12, "text": "前句"}]
    last = [{"start": 310, "end": 312, "text": "后句"}]
    assert speech.merge_chunk_words(first, last, 300) == first + last


def test_exports_unicode_timestamps_and_empty():
    result = example({"model": "test", "asr_revision": "1", "diarization_revision": "2", "language": "zh"})
    rendered = speech.exports(result)
    assert json.loads(rendered[".json"]) == result
    assert "说话人 A：你好。" in rendered[".txt"]
    assert "00:00:00,100 --> 00:00:01,200" in rendered[".srt"]
    assert "说话人 B（重叠发言）" in rendered[".srt"]
    assert speech.timestamp(3599.9996) == "01:00:00,000"
    result["segments"] = []
    assert speech.exports(result)[".srt"] == speech.exports(result)[".txt"] == ""


def test_missing_export_reuses_canonical_result_without_model_or_source_probe(speech_job, monkeypatch):
    store, task, work = speech_job
    store.save_checkpoint(task.id, "result", example(task.params))
    monkeypatch.setattr(speech, "probe", lambda *args: pytest.fail("Re-probed completed transcription"))
    assert (
        runner.process_task(task, store.path, work, None, 1, threading.Event(), lambda _: None) == "succeeded"
    )
    original = task.target.read_bytes()
    assert store.enqueue(task.source, task.target, task.kind, task.params, "next") == "skipped"
    store.commit()
    task.target.with_suffix(".srt").unlink()
    assert store.enqueue(task.source, task.target, task.kind, task.params, "repair") == "pending"
    store.commit()
    pending = store.claim("repair", 1)[0]
    assert (
        runner.process_task(pending, store.path, work, None, 1, threading.Event(), lambda _: None)
        == "succeeded"
    )
    assert task.target.read_bytes() == original
    assert task.target.with_suffix(".srt").exists()
    task.target.with_suffix(".txt").write_text("用户修改")
    with pytest.raises(ConflictError):
        store.enqueue(task.source, task.target, task.kind, task.params, "conflict")
    assert task.target.with_suffix(".txt").read_text() == "用户修改"


def test_partial_exports_recover_and_unknown_replacement_is_preserved(speech_job):
    store, task, work = speech_job
    artifact = work / "ready.json"
    artifact.write_text("{}")
    runner.publish(artifact, task, store, threading.Event(), target=task.target)
    text = task.target.with_suffix(".txt")
    text.write_text("partial")
    store.artifact_publishing(task.id, text, text.stat())
    store.save_checkpoint(task.id, "asr:0", {"words": ["已完成"]})
    store.recover()
    assert task.target.exists() and not text.exists()
    assert store.checkpoint(task.id, "asr:0") == {"words": ["已完成"]}
    assert store.enqueue(task.source, task.target, task.kind, task.params, "next") == "pending"
    store.commit()
    task = store.claim("next", 1)[0]
    text.write_text("partial")
    store.artifact_publishing(task.id, text, text.stat())
    text.rename(work / "old-inode")
    text.write_text("external replacement")
    store.recover()
    assert text.read_text() == "external replacement"
    with pytest.raises(ConflictError):
        store.enqueue(task.source, task.target, task.kind, task.params, "next")


@pytest.mark.parametrize("change", ["source", "params"])
def test_checkpoint_invalidated_when_inputs_change(speech_job, change):
    store, task, _ = speech_job
    store.save_checkpoint(task.id, "asr:0", {"words": []})
    store.finish_error(task, "interrupted", cancelled=True)
    params = dict(task.params)
    if change == "source":
        task.source.write_bytes(b"changed source")
    else:
        params["language"] = "en"
    store.enqueue(task.source, task.target, task.kind, params, "new")
    store.commit()
    assert store.checkpoint(task.id, "asr:0") is None


def test_no_audio_is_success_with_empty_exports(speech_job, make_clip):
    store, task, work = speech_job
    task.source.write_bytes(make_clip().read_bytes())
    store.enqueue(task.source, task.target, task.kind, task.params, "real")
    store.commit()
    task = store.claim("real", 1)[0]
    assert (
        runner.process_task(task, store.path, work, None, 1, threading.Event(), lambda _: None) == "succeeded"
    )
    assert json.loads(task.target.read_text())["reason"] == "no_audio"
    assert task.target.with_suffix(".srt").read_bytes() == b""
    assert task.target.with_suffix(".txt").read_bytes() == b""


def test_real_pcm_decode_and_space_check(make_clip, tmp_path, monkeypatch):
    source = make_clip(audio=True)
    info = speech.probe(source, threading.Event())
    audio = tmp_path / "audio.f32"
    offset = speech.extract_audio(source, audio, info, threading.Event(), lambda _: None)
    assert abs(offset) < 0.1
    assert abs(audio.stat().st_size / (16000 * 4) - 1) < 0.1
    monkeypatch.setattr(speech.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match="空间不足"):
        speech.extract_audio(source, audio, info, threading.Event(), lambda _: None)


def test_cancel_during_export_preserves_checkpoint(speech_job, monkeypatch):
    store, task, work = speech_job
    store.save_checkpoint(task.id, "result", example(task.params))
    stop = threading.Event()
    original = Store.artifact_publishing

    def cancel(self, *args):
        original(self, *args)
        stop.set()

    monkeypatch.setattr(Store, "artifact_publishing", cancel)
    assert runner.process_task(task, store.path, work, None, 1, stop, lambda _: None) == "pending"
    assert not task.target.exists()
    assert store.checkpoint(task.id, "result") is not None


def test_old_database_gets_additive_tables_without_changing_tasks(tmp_path):
    path = tmp_path / "old.sqlite3"
    store = Store(path)
    source = tmp_path / "old.mp4"
    source.touch()
    store.enqueue(source, tmp_path / "old.compressed.mp4", "compress", {}, "old")
    store.commit()
    original = tuple(store.db.execute("SELECT * FROM tasks").fetchone())
    store.close()
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE speech_artifacts")
        db.execute("DROP TABLE speech_checkpoints")
    store = Store(path)
    try:
        assert tuple(store.db.execute("SELECT * FROM tasks").fetchone()) == original
        assert store.db.execute("SELECT COUNT(*) FROM speech_checkpoints").fetchone()[0] == 0
    finally:
        store.close()


def test_ten_thousand_speech_tasks_dispatch_one_at_a_time(tmp_path):
    path = tmp_path / "queue.sqlite3"
    store = Store(path)
    source = tmp_path / "source.mp4"
    source.touch()
    try:
        for index in range(10_000):
            store.enqueue(source, tmp_path / f"{index}.transcript.json", "transcribe", {}, "large")
            if index % 100 == 0:
                store.commit()
        store.enqueue(source, tmp_path / "video.mp4", "compress", {}, "large")
        store.commit()
        active = peak = completed = 0
        lock = threading.Lock()

        def fake_work(task, progress):
            nonlocal active, peak, completed
            assert task.kind == "transcribe"
            with lock:
                active += 1
                peak = max(peak, active)
            worker_store = Store(path)
            try:
                # Inference and publication are tested separately; this stresses durable dispatch.
                worker_store.save_checkpoint(task.id, "result", {"segments": []})
                with worker_store.db:
                    worker_store.db.execute("UPDATE tasks SET status='succeeded' WHERE id=?", (task.id,))
            finally:
                worker_store.close()
                with lock:
                    active -= 1
                    completed += 1

        runner.run_queue(store, "large", 1, threading.Event(), fake_work, False, speech=True)
        assert peak == 1 and completed == 10_000
        assert store.counts() == {"pending": 1, "running": 0, "succeeded": 10_000, "failed": 0}
    finally:
        store.close()


def test_cli_defaults_and_missing_manifest(cli, tmp_path, monkeypatch):
    from video_watermaker.cli import parser

    args = parser().parse_args(["transcribe", "A"])
    assert (args.model, args.language, args.device, args.speakers, args.workers) == (
        "large-v3-turbo",
        "zh",
        "auto",
        None,
        1,
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "empty-cache"))
    source = tmp_path / "input.mp4"
    source.touch()
    result = cli("transcribe", source, "-o", tmp_path / "out")
    assert result.returncode == 1
    assert "models prepare" in result.stderr


def test_cli_compression_then_speech_and_preflight_failure(tmp_path, monkeypatch):
    from video_watermaker import cli, speech_client

    source = tmp_path / "input.mp4"
    source.touch()
    params = {"model": "test"}
    monkeypatch.setattr(speech, "speech_params", lambda _: params)
    monkeypatch.setattr(speech, "read_manifest", lambda _: {})
    monkeypatch.setattr(cli, "EncoderPolicy", lambda *args: None)
    events = []

    class Client:
        fail = False

        def __init__(self, stop):
            pass

        def request(self, *args):
            events.append("check")
            if self.fail:
                raise MediaError("missing model")
            return {"device": "cpu"}

        def close(self):
            events.append("close")

    def work(task, database, *args):
        events.append(task.kind)
        store = Store(database)
        store.finish_error(task, "simulated video failure")
        store.close()

    monkeypatch.setattr(speech_client, "SpeechClient", Client)
    monkeypatch.setattr(cli, "process_task", work)
    assert cli.main(["compress", str(source), "-o", str(tmp_path / "out"), "--transcribe"]) == 1
    assert events == ["check", "compress", "transcribe", "close"]
    events.clear()
    Client.fail = True
    assert cli.main(["transcribe", str(source), "-o", str(tmp_path / "fail")]) == 1
    store = Store(tmp_path / "fail" / ".video-watermaker" / "queue.sqlite3")
    try:
        assert store.counts() == {"pending": 1, "running": 0, "succeeded": 0, "failed": 0}
    finally:
        store.close()
    assert events == ["check", "close"]


def test_resumes_asr_chunks_and_diarization_is_global(speech_job, monkeypatch):
    from video_watermaker import speech_worker

    store, task, _ = speech_job
    engine = speech_worker.Engine.__new__(speech_worker.Engine)
    engine.params = task.params
    engine.emit = lambda _: None
    engine.np = SimpleNamespace(memmap=lambda *args, **kwargs: [0] * 601, asarray=lambda x: x)
    engine.perform = lambda operation: operation()
    monkeypatch.setattr(speech_worker, "SAMPLE_RATE", 1)
    calls = []

    def asr(samples, language):
        calls.append(len(samples))
        if len(calls) == 2:
            raise Cancelled("interrupt after first saved chunk")
        return {"words": [{"start": 2, "end": 2.5, "text": "你好。"}], "language": "zh", "device": "cpu"}

    diarization_calls = []

    def diarize(samples, offset, duration):
        diarization_calls.append(len(samples))
        turns = [{"start": 0, "end": 601, "speaker": "person"}]
        return {"turns": turns, "exclusive": turns, "device": "cpu"}

    engine.transcribe_chunk = asr
    engine.diarize = diarize
    request = {
        "database": str(store.path),
        "task_id": task.id,
        "audio": "unused",
        "offset": 0,
        "duration": 601,
    }
    with pytest.raises(Cancelled):
        engine.infer(request)
    assert store.checkpoint(task.id, "asr:0") is not None
    assert store.checkpoint(task.id, "asr:1") is None
    engine.infer(request)
    assert calls == [302, 303, 303, 3]
    assert diarization_calls == [601]
    result = store.checkpoint(task.id, "result")
    assert [segment["start"] for segment in result["segments"]] == [2, 300, 600]
    assert {segment["speaker"] for segment in result["segments"]} == {"A"}
    engine.infer(request)
    assert len(calls) == 4
