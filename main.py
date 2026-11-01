"""Compatibility entry point. Run with ``uv run main.py --help``."""

from video_watermaker.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
