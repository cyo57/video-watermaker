"""Speech orchestration and exports; deliberately usable without ML dependencies."""

import json
import math
import os
import shutil
from pathlib import Path

from .media import number, probe
from .process import MediaError, check_cancelled, run

SAMPLE_RATE = 16000
CHUNK_SECONDS = 300
CONTEXT_SECONDS = 2
MODELS = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large-v3": "Systran/faster-whisper-large-v3",
}
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def model_manifest_path():
    root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return root / "video-watermaker" / "models.json"


def read_manifest(model):
    try:
        data = json.loads(model_manifest_path().read_text())
        result = {"asr": data[model], "diarization": data["diarization"]}
        for item in result.values():
            if not all(isinstance(item[key], str) and item[key] for key in ("path", "revision")):
                raise ValueError("无效的模型记录")
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise MediaError(
            f"模型尚未准备好；先执行 uv run --extra speech main.py models prepare --model {model}"
        ) from exc


def speech_params(args):
    manifest = read_manifest(args.model)
    return {
        "revision": 4,
        "model": args.model,
        "language": args.language,
        "device": args.device,
        "speakers": args.speakers,
        "asr_revision": manifest["asr"]["revision"],
        "diarization_revision": manifest["diarization"]["revision"],
        "chunk_seconds": CHUNK_SECONDS,
        "context_seconds": CONTEXT_SECONDS,
    }


def chunk_windows(duration):
    for index in range(math.ceil(duration / CHUNK_SECONDS)):
        start = index * CHUNK_SECONDS
        end = min(duration, start + CHUNK_SECONDS)
        yield index, start, end, max(0, start - CONTEXT_SECONDS), min(duration, end + CONTEXT_SECONDS)


def core_words(words, window_start, core_start, core_end, offset, duration):
    """Assign each word to one non-overlapping core, preserving video timestamps."""
    result = []
    for word in words:
        start = float(word["start"]) + window_start
        end = float(word["end"]) + window_start
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start:
            continue
        if not core_start <= (start + end) / 2 < core_end:
            continue
        start, end = max(0, start + offset), min(duration, end + offset)
        if end > start and word["text"].strip():
            result.append({"start": round(start, 3), "end": round(end, 3), "text": word["text"]})
    return result


def merge_chunk_words(previous, current, boundary):
    """Join overlapping hypotheses at shared text, despite small timestamp drift."""
    if not previous:
        return list(current)
    if not current:
        return previous

    def spans(words):
        for index in range(len(words) - 1, -1, -1):
            word = words[index]
            if word["end"] < boundary - CONTEXT_SECONDS - 1:
                break
            if word["start"] > boundary + CONTEXT_SECONDS + 1:
                continue
            text = ""
            # Chinese words can be emitted as one token in one window and multiple
            # tokens in the next. Compare short word sequences as well as single words.
            for end in range(index, min(index + 3, len(words))):
                text += "".join(
                    character.casefold() for character in words[end]["text"] if character.isalnum()
                )
                if text:
                    center = (word["start"] + words[end]["end"]) / 2
                    yield text, center, end + 1

    right_spans = {}
    for text, center, end in spans(current):
        right_spans.setdefault(text, []).append((center, end))
    best = None
    for text, center, end in spans(previous):
        for right_center, right_end in right_spans.get(text, []):
            drift = abs(center - right_center)
            if drift > 0.75:
                continue
            score = abs((center + right_center) / 2 - boundary) + 2 * drift
            if best is None or score < best[0]:
                best = score, end, right_end
    if best is not None:
        return previous[: best[1]] + current[best[2] :]

    # With no common text (e.g. a silent seam), one hypothesis determines the cut.
    # Never let two independently rounded midpoint tests both discard a boundary word.
    left = [word for word in previous if word["start"] < boundary]
    cut = max(boundary, left[-1]["end"] if left else boundary)
    return left + [word for word in current if (word["start"] + word["end"]) / 2 >= cut]


def speaker_name(index):
    value = ""
    while index >= 0:
        value = chr(65 + index % 26) + value
        index = index // 26 - 1
    return value


def overlapping_intervals(turns):
    events = []
    for turn in turns:
        events.extend([(turn["start"], 1, turn["speaker"]), (turn["end"], -1, turn["speaker"])])
    active, result, previous = {}, [], None
    for time, delta, label in sorted(events):
        if previous is not None and time > previous and len(active) > 1:
            result.append((previous, time))
        active[label] = active.get(label, 0) + delta
        if active[label] == 0:
            del active[label]
        previous = time
    return result


def align_speakers(words, diarization):
    turns = sorted(diarization["exclusive"], key=lambda x: (x["start"], x["end"]))
    names = {}
    for turn in sorted(diarization["turns"], key=lambda x: (x["start"], x["speaker"])):
        names.setdefault(turn["speaker"], speaker_name(len(names)))
    overlap = overlapping_intervals(diarization["turns"])
    cursor = overlap_cursor = 0
    segments = []
    for word in sorted(words, key=lambda x: (x["start"], x["end"])):
        while cursor < len(turns) and turns[cursor]["end"] <= word["start"]:
            cursor += 1
        scores = {}
        for index in range(cursor, len(turns)):
            turn = turns[index]
            if turn["start"] >= word["end"]:
                break
            amount = min(turn["end"], word["end"]) - max(turn["start"], word["start"])
            if amount > 0:
                scores[turn["speaker"]] = scores.get(turn["speaker"], 0) + amount
        ranked = sorted(scores, key=lambda key: (-scores[key], key))
        label = None
        if ranked and (len(ranked) == 1 or scores[ranked[0]] > scores[ranked[1]] + 1e-6):
            label = names.get(ranked[0])
        while overlap_cursor < len(overlap) and overlap[overlap_cursor][1] <= word["start"]:
            overlap_cursor += 1
        simultaneous = overlap_cursor < len(overlap) and overlap[overlap_cursor][0] < word["end"]
        previous = segments[-1] if segments else None
        if previous and (
            previous["speaker"] == label
            and previous["overlap"] == simultaneous
            and word["start"] - previous["end"] <= 1
            and word["end"] - previous["start"] <= 8
            and len(previous["text"]) + len(word["text"]) <= 80
            and not previous["text"].rstrip().endswith(tuple("。！？.!?"))
        ):
            previous["text"] += word["text"]
            previous["end"] = max(previous["end"], word["end"])
        else:
            segments.append({**word, "speaker": label, "overlap": simultaneous})
    for segment in segments:
        segment["text"] = segment["text"].strip()
    return segments


def document(params, duration, segments, *, reason=None, language=None, devices=None):
    return {
        "schema_version": 1,
        "model": params["model"],
        "model_revision": params["asr_revision"],
        "diarization_model": DIARIZATION_MODEL,
        "diarization_revision": params["diarization_revision"],
        "language": language or params["language"],
        "duration": duration,
        "reason": reason,
        "devices": devices or {},
        "segments": segments,
    }


def timestamp(seconds, separator=","):
    milliseconds = max(0, round(seconds * 1000))
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{milliseconds:03}"


def exports(result):
    text, srt = [], []
    for index, segment in enumerate(result["segments"], 1):
        label = f"说话人 {segment['speaker']}" if segment["speaker"] else "说话人未知"
        if segment["overlap"]:
            label += "（重叠发言）"
        content = " ".join(segment["text"].splitlines())
        start, end = segment["start"], segment["end"]
        end = max(start + 0.001, end)
        text.append(f"[{timestamp(start, '.')} → {timestamp(end, '.')}] {label}：{content}\n")
        srt.append(f"{index}\n{timestamp(start)} --> {timestamp(end)}\n{label}：{content}\n\n")
    return {
        ".json": json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        ".txt": "".join(text),
        ".srt": "".join(srt),
    }


def extract_audio(source, path, info, stop, progress):
    offset = number(info.audio.get("start_time")) - number(info.video.get("start_time"))
    duration = max(0, info.duration - offset)
    required = int(duration * SAMPLE_RATE * 4) + 256 * 1024 * 1024
    if shutil.disk_usage(path.parent).free < required:
        raise OSError(28, "临时音频空间不足；请释放 /tmp 空间后重试")

    def advance(seconds):
        if shutil.disk_usage(path.parent).free < 64 * 1024 * 1024:
            raise OSError(28, "临时音频空间不足")
        progress(min(0.1, 0.1 * seconds / max(duration, 0.001)))

    if duration <= 0:
        path.write_bytes(b"")
        return offset
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-map",
            f"0:{info.audio['index']}",
            "-vn",
            "-t",
            str(duration),
            "-af",
            "asetpts=PTS-STARTPTS,aresample=16000:async=1:first_pts=0",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "-progress",
            "pipe:1",
            "-nostats",
            str(path),
        ],
        stop,
        timeout=None,
        progress=advance,
    )
    return offset


def transcribe_task(task, store, work, stop, progress, client):
    from .runner import publish

    result = store.checkpoint(task.id, "result")
    if result is None:
        info = probe(task.source, stop)
        if info.audio is None:
            result = document(task.params, info.duration, [], reason="no_audio")
        else:
            audio = work / "audio.f32"
            offset = extract_audio(task.source, audio, info, stop, progress)
            if audio.stat().st_size == 0:
                result = document(task.params, info.duration, [], reason="no_speech")
            else:
                client.request(
                    "infer",
                    {
                        "database": str(store.path),
                        "task_id": task.id,
                        "params": task.params,
                        "audio": str(audio),
                        "offset": offset,
                        "duration": info.duration,
                    },
                    progress,
                )
                result = store.checkpoint(task.id, "result")
                if result is None:
                    raise MediaError("语音进程未保存转写结果")
        store.save_checkpoint(task.id, "result", result)
    rendered = exports(result)
    for target in store.speech_targets(task.target):
        check_cancelled(stop)
        if not store.artifact_ready(task.id, target):
            artifact = work / ("export" + target.suffix)
            artifact.write_text(rendered[target.suffix], encoding="utf-8")
            publish(artifact, task, store, stop, target=target)
    store.speech_succeed(task)
