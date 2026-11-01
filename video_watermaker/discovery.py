"""Incremental discovery; never follow symlinks or scan our output tree."""

import os
from collections.abc import Callable, Iterator
from pathlib import Path

VIDEO_EXTENSIONS = frozenset(
    [
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".3gp",
        ".ts",
        ".mts",
        ".m2ts",
        ".mpg",
        ".mpeg",
        ".vob",
        ".ogv",
        ".asf",
        ".mxf",
    ]
)


def discover(source: Path, output: Path, on_error: Callable[[Path, OSError], None]) -> Iterator[Path]:
    if source.is_file():
        yield source
        return
    stack = []
    try:
        stack.append(os.scandir(source))
        while stack:
            try:
                entry = next(stack[-1])
            except StopIteration:
                stack.pop().close()
                continue
            except OSError as exc:
                on_error(source, exc)
                stack.pop().close()
                continue
            path = Path(entry.path)
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if path == output or path.name == ".video-watermaker":
                        continue
                    stack.append(os.scandir(path))
                elif entry.is_file(follow_symlinks=False) and path.suffix.lower() in VIDEO_EXTENSIONS:
                    yield path
            except OSError as exc:
                on_error(path, exc)
    finally:
        for iterator in stack:
            iterator.close()


def output_path(source: Path, root: Path, output: Path, kind: str) -> Path:
    relative = source.relative_to(root)
    suffix = {"compress": ".compressed.mp4", "thumbnail": ".thumbnail.jpg", "transcribe": ".transcript.json"}[
        kind
    ]
    return output / relative.parent / (relative.name + suffix)
