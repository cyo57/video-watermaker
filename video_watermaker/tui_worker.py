"""Private TUI subprocess entry: ordinary CLI logs plus bounded progress events."""

import json

from .cli import main

EVENT_PREFIX = "@video-watermaker:"


def emit(event):
    print(EVENT_PREFIX + json.dumps(event, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main(progress=emit))
