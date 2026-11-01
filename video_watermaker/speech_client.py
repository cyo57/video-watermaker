"""A single cancellable inference process, with bounded IPC and parent-death cleanup."""

import json
import logging
import os
import re
import selectors
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

from .process import MediaError, _terminate, check_cancelled

logger = logging.getLogger(__name__)


def redact(message):
    return re.sub(r"hf_[A-Za-z0-9_.-]+", "[HF_TOKEN]", str(message))


def worker_environment(online=False):
    env = dict(os.environ)
    env.update(
        {
            "TMPDIR": "/tmp",
            "PYANNOTE_METRICS_ENABLED": "0",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4",
        }
    )
    if online:
        env.pop("HF_HUB_OFFLINE", None)
    else:
        env["HF_HUB_OFFLINE"] = "1"
    package_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
    # CTranslate2 needs CUDA 12/cuDNN 9 even when the system CUDA toolkit is 13.
    libraries = sorted(Path(sysconfig.get_paths()["purelib"]).glob("nvidia/*/lib"))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(path) for path in libraries] + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    )
    return env


class SpeechClient:
    def __init__(self, stop, *, online=False):
        self.stop = stop
        self.process = None
        self.online = online

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.process is not None:
            _terminate(self.process)
            for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                pipe.close()
            self.process = None

    def request(self, operation, payload, progress=lambda _: None):
        check_cancelled(self.stop)
        if self.process is None:
            launcher = Path(__file__).with_name("_child.py")
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    str(launcher),
                    str(os.getpid()),
                    sys.executable,
                    "-m",
                    "video_watermaker.speech_worker",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=worker_environment(self.online),
            )
        process = self.process
        errors, pending = bytearray(), bytearray()
        try:
            process.stdin.write((json.dumps({"op": operation, **payload}) + "\n").encode())
            process.stdin.flush()
            last_message = time.monotonic()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, "out")
                selector.register(process.stderr, selectors.EVENT_READ, "err")
                while selector.get_map():
                    check_cancelled(self.stop)
                    if time.monotonic() - last_message > 120:
                        raise MediaError("语音进程超过 120 秒未响应")
                    for key, _ in selector.select(0.2):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        if key.data == "err":
                            errors.extend(chunk)
                            del errors[:-65536]
                            continue
                        pending.extend(chunk)
                        if len(pending) > 1024 * 1024:
                            raise MediaError("语音进程返回了过大的消息")
                        while b"\n" in pending:
                            line, _, rest = pending.partition(b"\n")
                            pending[:] = rest
                            event = json.loads(line)
                            last_message = time.monotonic()
                            if event["type"] == "progress":
                                progress(event["value"])
                            elif event["type"] == "notice":
                                logger.warning("%s", redact(event["message"]))
                            elif event["type"] == "result":
                                if not event["ok"]:
                                    raise InferenceError(redact(event["error"]))
                                return event.get("value")
                raise MediaError(f"语音进程意外退出：{redact(errors.decode(errors='replace')[-4000:])}")
        except InferenceError:
            raise
        except BaseException:
            self.close()
            # Infrastructure failure must not fail every remaining video in a large queue.
            self.stop.set()
            raise


class InferenceError(MediaError):
    """A reported inference failure; the process is still healthy."""
