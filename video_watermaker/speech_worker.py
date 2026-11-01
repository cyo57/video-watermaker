"""Optional ML runtime. Only imported inside the isolated speech process."""

import gc
import json
import os
import sys
import threading
import warnings
from collections import Counter
from itertools import pairwise
from pathlib import Path

from .speech import (
    DIARIZATION_MODEL,
    MODELS,
    SAMPLE_RATE,
    align_speakers,
    chunk_windows,
    core_words,
    document,
    merge_chunk_words,
    model_manifest_path,
)
from .store import Store


def dependencies():
    try:
        import numpy as np
        import torch
        from faster_whisper import WhisperModel

        # Waveform input deliberately bypasses this optional decoder on FFmpeg 9.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"\s*torchcodec is not installed correctly.*", category=UserWarning
            )
            from pyannote.audio import Pipeline
        return np, torch, WhisperModel, Pipeline
    except (ImportError, OSError) as exc:
        raise RuntimeError("语音依赖不可用，请执行 uv sync --extra speech；" + str(exc)) from exc


def cuda_error(error):
    return any(
        value in str(error).lower()
        for value in (
            "cuda",
            "cudnn",
            "cublas",
            "out of memory",
            "driver",
            "compute capability",
        )
    )


def speech_windows(samples):
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    clips = get_speech_timestamps(samples, VadOptions(min_silence_duration_ms=500))
    if not clips:
        return []
    boundaries = [0]
    for previous, current in pairwise(clips):
        if current["start"] - previous["end"] > 2 * SAMPLE_RATE:
            boundaries.append((previous["end"] + current["start"]) // 2)
    boundaries.append(len(samples))
    return list(pairwise(boundaries))


class Engine:
    def __init__(self, manifest, params, emit):
        self.np, self.torch, self.WhisperModel, self.Pipeline = dependencies()
        self.torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.manifest, self.params, self.emit = manifest, params, emit
        self.asr = self.diarizer = None
        self.asr_loaded = False
        self.device = "cpu"
        if params["device"] != "cpu":
            if self.torch.cuda.is_available():
                self.device = "cuda"
            elif params["device"] == "cuda":
                raise RuntimeError("指定了 CUDA，但 NVIDIA GPU 不可用")
            else:
                emit({"type": "notice", "message": "CUDA 不可用，语音处理使用 CPU"})
        self.perform(self._check)

    def notice(self, message):
        self.emit({"type": "notice", "message": message})

    def perform(self, operation):
        try:
            return operation()
        except Exception as exc:
            if self.device != "cuda" or self.params["device"] != "auto" or not cuda_error(exc):
                raise
            self.notice("语音 GPU 不可用或显存不足，切换 CPU 并重试当前阶段")
        # Release exception traceback (which can own GPU tensors) before reclaiming memory.
        if self.diarizer is not None:
            self.diarizer.to(self.torch.device("cpu"))
        self.asr = None
        self.asr_loaded = False
        gc.collect()
        self.torch.cuda.empty_cache()
        self.device = "cpu"
        return operation()

    def activate_asr(self):
        if self.diarizer is not None:
            self.diarizer.to(self.torch.device("cpu"))
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
        if self.asr is None:
            self.asr = self.WhisperModel(
                self.manifest["asr"]["path"],
                device=self.device,
                compute_type="int8_float16" if self.device == "cuda" else "int8",
                cpu_threads=min(4, os.cpu_count() or 1),
                num_workers=1,
                local_files_only=True,
            )
        elif not self.asr_loaded:
            self.asr.model.load_model()
        self.asr_loaded = True

    def activate_diarizer(self):
        if self.asr is not None and self.asr_loaded and self.device == "cuda":
            self.asr.model.unload_model(to_cpu=True)
            self.asr_loaded = False
        if self.diarizer is None:
            self.diarizer = self.Pipeline.from_pretrained(self.manifest["diarization"]["path"])
            if self.diarizer is None:
                raise RuntimeError("说话人模型加载失败，请重新运行 models prepare")
            self.diarizer.segmentation_batch_size = 4
            self.diarizer.embedding_batch_size = 4
        self.diarizer.to(self.torch.device(self.device))

    def _check(self):
        self.activate_asr()
        language = self.params["language"]
        if language != "auto" and language not in self.asr.supported_languages:
            raise ValueError(f"不支持的语言代码：{language}")
        # Force an actual CT2 operation so missing CUDA shared libraries fail before dispatch.
        self.asr.detect_language(self.np.zeros(SAMPLE_RATE, dtype=self.np.float32))
        self.activate_diarizer()
        self.diarizer.to(self.torch.device("cpu"))
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def transcribe_chunk(self, samples, language):
        windows = speech_windows(samples)
        if not windows:
            return {"words": [], "language": language, "device": self.device}
        self.activate_asr()
        words, languages = [], []
        for start, end in windows:
            # Decode long-separated speech independently so a word after a long pause
            # cannot be aligned to the previous utterance when VAD removes the silence.
            offset = start / SAMPLE_RATE
            segments, info = self.asr.transcribe(
                samples[start:end],
                language=None if language == "auto" else language,
                beam_size=1,
                best_of=1,
                temperature=0,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,
            )
            for segment in segments:
                if segment.words:
                    words.extend(
                        {"start": word.start + offset, "end": word.end + offset, "text": word.word}
                        for word in segment.words
                    )
                    languages.extend([info.language] * len(segment.words))
                elif segment.text.strip():
                    words.append(
                        {"start": segment.start + offset, "end": segment.end + offset, "text": segment.text}
                    )
                    languages.append(info.language)
        detected = Counter(languages).most_common(1)[0][0] if languages else language
        return {"words": words, "language": detected, "device": self.device}

    def diarize(self, samples, offset, duration):
        self.activate_diarizer()

        def hook(step_name, step_artifact, file=None, total=None, completed=None):
            if total and completed is not None:
                self.emit({"type": "progress", "value": 0.7 + 0.2 * completed / total})

        kwargs = {"num_speakers": self.params["speakers"]} if self.params["speakers"] else {}
        output = self.diarizer(
            {"waveform": self.torch.from_numpy(samples).unsqueeze(0), "sample_rate": SAMPLE_RATE},
            hook=hook,
            **kwargs,
        )

        def turns(annotation):
            result = []
            for turn, _, label in annotation.itertracks(yield_label=True):
                start, end = max(0, turn.start + offset), min(duration, turn.end + offset)
                if end > start:
                    result.append({"start": float(start), "end": float(end), "speaker": str(label)})
            return result

        return {
            "turns": turns(output.speaker_diarization),
            "exclusive": turns(output.exclusive_speaker_diarization),
            "device": self.device,
        }

    def infer(self, request):
        store = Store(Path(request["database"]))
        samples = None
        try:
            task_id = request["task_id"]
            if store.checkpoint(task_id, "result") is not None:
                return
            # Read/write mapping enables zero-copy Torch tensors; models receive waveform, not paths.
            samples = self.np.memmap(request["audio"], dtype="<f4", mode="r+")
            duration = len(samples) / SAMPLE_RATE
            words, languages, devices = [], [], set()
            for index, start, end, left, right in chunk_windows(duration):
                name = f"asr:{index}"
                chunk = store.checkpoint(task_id, name)
                if chunk is None:
                    waveform = self.np.asarray(
                        samples[round(left * SAMPLE_RATE) : round(right * SAMPLE_RATE)]
                    )
                    raw = self.perform(
                        lambda waveform=waveform: self.transcribe_chunk(waveform, self.params["language"])
                    )
                    chunk = {
                        **raw,
                        "words": core_words(
                            raw["words"],
                            left,
                            left,
                            right,
                            request["offset"],
                            request["duration"],
                        ),
                    }
                    store.save_checkpoint(task_id, name, chunk)
                words = merge_chunk_words(words, chunk["words"], start + request["offset"])
                devices.add(chunk["device"])
                if chunk["words"]:
                    languages.append(chunk["language"])
                self.emit({"type": "progress", "value": 0.1 + 0.6 * end / duration})
            diarization = store.checkpoint(task_id, "diarization")
            if words and diarization is None:
                diarization = self.perform(
                    lambda: self.diarize(samples, request["offset"], request["duration"])
                )
                store.save_checkpoint(task_id, "diarization", diarization)
            segments = align_speakers(words, diarization) if words else []
            language = Counter(languages).most_common(1)[0][0] if languages else self.params["language"]
            result = document(
                self.params,
                request["duration"],
                segments,
                reason=None if words else "no_speech",
                language=language,
                devices={
                    "asr": sorted(devices),
                    "diarization": diarization["device"] if diarization else None,
                },
            )
            store.save_checkpoint(task_id, "result", result)
        finally:
            if samples is not None:
                # Drop all NumPy views before the map; no model retains the waveform between requests.
                del samples
            store.close()


def prepare(model, emit):
    dependencies()
    from huggingface_hub import get_token, hf_hub_download, snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    # Check access first, before downloading the much larger ASR model.
    token = get_token()
    try:
        hf_hub_download(DIARIZATION_MODEL, "config.yaml", token=token)
    except HfHubHTTPError as exc:
        if exc.response is None or exc.response.status_code not in (401, 403):
            raise
        raise RuntimeError(
            "无法下载说话人模型；请在 https://huggingface.co/pyannote/speaker-diarization-community-1 "
            "接受使用条款，再通过 HF_TOKEN 或 hf auth login 配置本机令牌。"
        ) from exc
    emit({"type": "notice", "message": "下载本地语音模型；视频和音频不会上传"})
    diarization = Path(snapshot_download(DIARIZATION_MODEL, token=token))
    asr = Path(
        snapshot_download(
            MODELS[model],
            allow_patterns=[
                "config.json",
                "preprocessor_config.json",
                "model.bin",
                "tokenizer.json",
                "vocabulary.*",
            ],
        )
    )
    manifest = {
        "asr": {"path": str(asr), "revision": asr.name},
        "diarization": {"path": str(diarization), "revision": diarization.name},
    }
    Engine(manifest, {"device": "auto", "language": "zh", "speakers": None}, emit)
    path = model_manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    from .runner import RunnerLock

    with RunnerLock(path.with_suffix(".lock")):
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            saved = {}
        saved.update({model: manifest["asr"], "diarization": manifest["diarization"]})
        with path.open("w", encoding="utf-8") as file:
            json.dump(saved, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
    return manifest


def main():
    protocol = sys.stdout
    sys.stdout = sys.stderr
    lock = threading.Lock()

    def emit(event):
        with lock:
            protocol.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            protocol.flush()

    engine = None
    for line in sys.stdin:
        done = threading.Event()

        def heartbeat(done=done):
            while not done.wait(5):
                emit({"type": "heartbeat"})

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        result = None
        try:
            request = json.loads(line)
            if request["op"] == "prepare":
                result = prepare(request["model"], emit)
            elif request["op"] == "check":
                engine = Engine(request["manifest"], request["params"], emit)
                result = {"device": engine.device}
            elif request["op"] == "infer" and engine is not None:
                engine.infer(request)
            else:
                raise ValueError("无效的语音进程请求")
            response = {"type": "result", "ok": True, "value": result}
        except Exception as exc:  # noqa: BLE001 -- isolate job errors and return them over IPC.
            response = {"type": "result", "ok": False, "error": str(exc)}
        finally:
            done.set()
            thread.join()
        emit(response)


if __name__ == "__main__":
    main()
