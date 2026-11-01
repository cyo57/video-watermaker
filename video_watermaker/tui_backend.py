"""Small, bounded adapters for the terminal UI; never import ML libraries."""

import asyncio
import importlib.util
import json
import os
import signal
import sqlite3
import sys
from pathlib import Path

from .process import MediaError
from .speech import read_manifest
from .tui_worker import EVENT_PREFIX

PAGE_SIZE = 100
STATUSES = {"pending": "等待", "running": "运行", "succeeded": "成功", "failed": "失败"}
KINDS = {"compress": "压缩", "thumbnail": "缩略图", "transcribe": "转写"}
DEFAULTS = {
    "source": "",
    "output": "output",
    "mode": "compress",
    "thumbnail": False,
    "transcribe": False,
    "workers": "2",
    "encoder": "auto",
    "rows": "4",
    "cols": "4",
    "width": "3840",
    "quality": "80",
    "rotate": "0",
    "watermark": "",
    "font": "",
    "model": "large-v3-turbo",
    "language": "zh",
    "device": "auto",
    "speakers": "",
}
CHOICES = {
    "mode": ("compress", "thumbnail", "transcribe"),
    "encoder": ("auto", "cpu", "nvenc"),
    "rotate": ("0", "90", "180", "270"),
    "model": ("large-v3-turbo", "large-v3"),
    "device": ("auto", "cuda", "cpu"),
}


def settings_path():
    root = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return root / "video-watermaker" / "tui.json"


def load_settings(path):
    result = dict(DEFAULTS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key, default in DEFAULTS.items():
                value = data.get(key)
                if type(value) is type(default) and (key not in CHOICES or value in CHOICES[key]):
                    result[key] = value
    except (OSError, ValueError):
        pass
    return result


def save_settings(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def command_args(values, *, retry=False):
    """Validate before spawning. Arguments remain separate, never shell-interpolated."""
    if not values["source"].strip():
        raise ValueError("请填写输入视频或目录")
    source = Path(values["source"]).expanduser().resolve()
    if not source.is_file() and not source.is_dir():
        raise ValueError(f"输入不存在：{source}")
    if not values["output"].strip():
        raise ValueError("请填写输出目录")
    output = Path(values["output"]).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("输出必须是目录")
    if source == output:
        raise ValueError("输入目录和输出目录不能相同")
    for key, choices in CHOICES.items():
        if values[key] not in choices:
            raise ValueError(f"无效选项：{key}")
    mode = values["mode"]
    args = [mode, str(source), "-o", str(output)]

    def integer(key, low, high, label):
        try:
            number = int(values[key])
        except ValueError:
            raise ValueError(f"{label}必须是整数") from None
        if not low <= number <= high:
            raise ValueError(f"{label}范围：{low}–{high}")
        args.extend([f"--{key}", str(number)])

    if retry:
        args.append("--retry-failed")
    if mode != "transcribe":
        integer("workers", 1, 32, "媒体并发数")
    if mode == "compress":
        args += ["--encoder", values["encoder"]]
        args += [f"--{key}" for key in ("thumbnail", "transcribe") if values[key]]
    if mode == "thumbnail" or (mode == "compress" and values["thumbnail"]):
        for key, low, high, label in (
            ("rows", 1, 20, "网格行数"),
            ("cols", 1, 20, "网格列数"),
            ("width", 64, 16384, "图片宽度"),
            ("quality", 1, 100, "JPEG 质量"),
        ):
            integer(key, low, high, label)
        args += ["--rotate", values["rotate"]]
        if values["watermark"]:
            args += ["--watermark-text", values["watermark"]]
        if values["font"]:
            font = Path(values["font"]).expanduser().resolve()
            if not font.is_file():
                raise ValueError(f"字体文件不存在：{font}")
            args += ["--font", str(font)]
    if mode == "transcribe" or (mode == "compress" and values["transcribe"]):
        language = values["language"].strip()
        if not language:
            raise ValueError("请填写语言代码，例如 zh、en 或 auto")
        args += ["--model", values["model"], "--language", language, "--device", values["device"]]
        if values["speakers"].strip():
            integer("speakers", 1, 100, "说话人数")
    return args


def model_status(model):
    if any(importlib.util.find_spec(name) is None for name in ("faster_whisper", "torch", "pyannote")):
        return "缺少语音依赖：退出后运行 uv sync --extra speech", False
    try:
        manifest = read_manifest(model)
        if not all(Path(item["path"]).is_dir() for item in manifest.values()):
            return "模型缓存缺失，请准备模型", True
    except MediaError:
        return "模型尚未准备，请点击准备模型（首次需要联网）", True
    return "本地模型已记录；开始时检查能否加载，转写离线运行", True


def snapshot(output, status="all", search="", page=0):
    """Read-only, one consistent snapshot, at most 100 rows, short lock timeout."""
    database = output / ".video-watermaker" / "queue.sqlite3"
    empty = {"counts": dict.fromkeys(STATUSES, 0), "rows": [], "total": 0, "page": 0}
    if not database.is_file():
        return empty
    db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1)
    db.row_factory = sqlite3.Row
    try:
        db.execute("BEGIN")
        counts = dict(empty["counts"])
        counts.update(
            (row[0], row[1]) for row in db.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status")
        )
        where, params = [], []
        if status in STATUSES:
            where.append("status=?")
            params.append(status)
        if search:
            where.append("instr(lower(source), lower(?)) > 0")
            params.append(search)
        condition = " WHERE " + " AND ".join(where) if where else ""
        total = db.execute("SELECT COUNT(*) FROM tasks" + condition, params).fetchone()[0]
        page = min(max(0, page), max(0, (total - 1) // PAGE_SIZE))
        rows = db.execute(
            "SELECT id, source, target, kind, status, attempts, error FROM tasks"
            + condition
            + " ORDER BY id LIMIT ? OFFSET ?",
            [*params, PAGE_SIZE, page * PAGE_SIZE],
        ).fetchall()
        return {"counts": counts, "rows": [dict(row) for row in rows], "total": total, "page": page}
    finally:
        db.close()


class JobProcess:
    """Own a CLI child, drain both pipes and reap it even if a UI worker is cancelled."""

    def __init__(self):
        self.process = None
        self.stopping = asyncio.Event()

    def stop(self):
        self.stopping.set()
        if self.process and self.process.returncode is None:
            try:
                self.process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

    async def _watch_stop(self):
        await self.stopping.wait()
        self.stop()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=15)
        except TimeoutError:
            if self.process.returncode is None:
                self.process.kill()

    async def run(self, args, log, progress):
        if self.stopping.is_set():
            return 130
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONUNBUFFERED="1", TMPDIR="/tmp")
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).with_name("_child.py")),
            str(os.getpid()),
            sys.executable,
            "-m",
            "video_watermaker.tui_worker",
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            start_new_session=True,
        )

        def line_received(line, events):
            line = line.decode("utf-8", errors="replace").rstrip()
            if events and line.startswith(EVENT_PREFIX):
                try:
                    progress(json.loads(line[len(EVENT_PREFIX) :]))
                    return
                except ValueError:
                    pass
            if line:
                log(line[-8000:])

        async def drain(stream, events=False):
            pending = b""
            while chunk := await stream.read(4096):
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    line_received(line, events)
                if len(pending) > 65536:
                    line_received(pending[:65536], False)
                    pending = pending[65536:]
                await asyncio.sleep(0)
            if pending:
                line_received(pending, events)

        watcher = asyncio.create_task(self._watch_stop())
        readers = [
            asyncio.create_task(drain(self.process.stdout, True)),
            asyncio.create_task(drain(self.process.stderr)),
        ]
        try:
            await asyncio.gather(*readers)
            return await self.process.wait()
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            for reader in readers:
                reader.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            if self.process.returncode is None:
                self.stop()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=15)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
