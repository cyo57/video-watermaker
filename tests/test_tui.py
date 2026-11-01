import asyncio
import json
import sqlite3
import time

import pytest
from textual.widgets import DataTable, Input, Select, Static, TabbedContent

from video_watermaker.store import Store
from video_watermaker.tui import VideoApp
from video_watermaker.tui_backend import DEFAULTS, JobProcess, command_args, load_settings, snapshot


async def until(condition, timeout=30):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "TUI operation timed out"
        await asyncio.sleep(0.05)


def test_arguments_spaces_combinations_and_validation(tmp_path):
    source = tmp_path / "目录 A [test]; $name.mp4"
    source.touch()
    values = dict(
        DEFAULTS,
        source=str(source),
        output=str(tmp_path / "目录 B"),
        thumbnail=True,
        transcribe=True,
        speakers="2",
        watermark="hello [red] 中文",
    )
    args = command_args(values, retry=True)
    assert args[1] == str(source)
    assert args[args.index("--watermark-text") + 1] == values["watermark"]
    assert all(flag in args for flag in ("--retry-failed", "--thumbnail", "--transcribe", "--speakers"))
    with pytest.raises(ValueError, match="媒体并发数"):
        command_args(dict(values, workers="0"))
    with pytest.raises(ValueError, match="不能相同"):
        command_args(dict(values, source=str(tmp_path), output=str(tmp_path)))
    with pytest.raises(ValueError, match="说话人数"):
        command_args(dict(values, speakers="abc"))
    # Disabled settings must not prevent unrelated jobs from starting.
    args = command_args(dict(values, mode="transcribe", workers="bad", rows="bad"))
    assert "--workers" not in args and "--thumbnail" not in args


def test_settings_corruption_and_invalid_choices(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"mode": "bad", "workers": 3, "thumbnail": "yes", "source": "中文"}')
    values = load_settings(config)
    assert values == dict(DEFAULTS, source="中文")
    config.write_text("not JSON")
    assert load_settings(config) == DEFAULTS


@pytest.fixture
def large_queue(tmp_path):
    output = tmp_path / "output"
    state = output / ".video-watermaker"
    state.mkdir(parents=True)
    store = Store(state / "queue.sqlite3")
    store.db.executemany(
        """INSERT INTO tasks(source,target,kind,params,source_size,source_mtime,batch,status,updated,error)
           VALUES(?,?,'compress','{}',0,0,'batch',?,0,?)""",
        (
            (
                f"/library/视频 {i:05}.mp4",
                f"/output/{i}.mp4",
                "failed" if i % 100 == 0 else "pending",
                "broken [red] video" if i % 100 == 0 else None,
            )
            for i in range(10_000)
        ),
    )
    store.commit()
    store.close()
    return output


def test_snapshot_large_queue_readonly_and_bounded(large_queue, tmp_path):
    result = snapshot(large_queue)
    assert result["total"] == 10_000 and len(result["rows"]) == 100
    assert result["counts"]["failed"] == 100
    assert snapshot(large_queue, page=99)["rows"][-1]["id"] == 10_000
    assert snapshot(large_queue, page=200)["page"] == 99
    assert snapshot(large_queue, "failed")["total"] == 100
    assert snapshot(large_queue, search="视频 09999")["total"] == 1
    assert snapshot(large_queue, search="%")["total"] == 0  # Search is literal, not LIKE syntax.
    assert snapshot(tmp_path / "missing")["total"] == 0
    assert not (tmp_path / "missing").exists()


def test_headless_filter_paging_resize_and_keyboard(large_queue, tmp_path):
    async def check():
        app = VideoApp(output=large_queue, config=tmp_path / "settings.json")
        async with app.run_test(size=(100, 35)) as pilot:
            await until(lambda: app.total == 10_000)
            app.query_one("#tabs", TabbedContent).active = "queue"
            # Tab activation is asynchronous; a hidden button has a zero-sized region.
            await until(lambda: app.query_one("#next").region.width > 0)
            await pilot.pause()
            assert app.query_one("#tasks", DataTable).row_count == 100
            assert await pilot.click("#next")
            await until(lambda: "101" in app.rows)
            app.query_one("#filter", Select).value = "failed"
            await until(lambda: app.total == 100)
            assert app.page == 0
            app.query_one("#search", Input).value = "视频 09900"
            await until(lambda: app.total == 1)
            assert "broken [red] video" in str(app.query_one("#details", Static).render())
            await pilot.resize_terminal(80, 24)
            await pilot.pause()
            assert app.query_one("#details-scroll").region.bottom <= app.query_one("#actions").region.y
            assert app.query_one("#tasks").region.height >= 3
            await pilot.press("ctrl+q")

    asyncio.run(check())


def test_tui_real_compress_resume_and_failure_retry(make_clip, tmp_path):
    source = make_clip(name="目录 A/视频 [red].mp4")
    broken = source.parent / "bad.mp4"
    broken.write_bytes(b"not a video")
    output = tmp_path / "目录 B"
    config = tmp_path / "settings.json"

    async def check():
        app = VideoApp(source=source.parent, output=output, config=config)
        async with app.run_test(size=(100, 35)) as pilot:
            app.query_one("#encoder", Select).value = "cpu"
            await pilot.click("#start")
            await until(lambda: not app.busy and app.last_exit is not None)
            assert app.last_exit == 1
            state = snapshot(output)
            assert state["counts"]["succeeded"] == 1 and state["counts"]["failed"] == 1
            # Same command skips the good file and does not retry known failures implicitly.
            await pilot.press("ctrl+s")
            await until(lambda: not app.busy)
            assert all(row["attempts"] == 1 for row in snapshot(output)["rows"])
            await pilot.click("#retry")
            await until(lambda: not app.busy)
            rows = snapshot(output)["rows"]
            assert next(row for row in rows if row["status"] == "failed")["attempts"] == 2
            assert next(row for row in rows if row["status"] == "succeeded")["attempts"] == 1
            assert json.loads(config.read_text())["source"] == str(source.parent)
            await pilot.press("ctrl+q")

    asyncio.run(check())


def test_tui_safe_quit_and_resume(make_clip, tmp_path):
    source = make_clip(duration="120", size="1280x720")
    output = tmp_path / "output"

    async def check():
        app = VideoApp(source=source, output=output, config=tmp_path / "settings.json")
        async with app.run_test(size=(100, 35)) as pilot:
            app.query_one("#encoder", Select).value = "cpu"
            await pilot.click("#start")
            await until(lambda: bool(app.active))
            assert app.query_one("#form").disabled
            await pilot.press("ctrl+q")
            await until(lambda: not app.busy)
            assert app.controller.process.returncode is not None
        state = snapshot(output)
        assert state["counts"]["running"] == 0
        assert state["counts"]["pending"] == 1
        # Reuse the exact saved configuration to resume the interrupted queue.
        app = VideoApp(config=tmp_path / "settings.json")
        async with app.run_test(size=(100, 35)) as pilot:
            await pilot.press("ctrl+s")
            await until(lambda: not app.busy, timeout=120)
            assert app.last_exit == 0
            assert snapshot(output)["rows"][0]["attempts"] == 2
            await pilot.press("ctrl+q")

    asyncio.run(check())


def test_job_process_stop_before_launch():
    async def check():
        child = JobProcess()
        child.stop()
        assert await child.run([], lambda _: None, lambda _: None) == 130
        assert child.process is None

    asyncio.run(check())


def test_job_process_cancel_reaps_child(make_clip, tmp_path):
    source = make_clip(duration="120", size="1280x720")

    async def check():
        child = JobProcess()
        events = []
        task = asyncio.create_task(
            child.run(
                ["compress", str(source), "-o", str(tmp_path / "output"), "--encoder", "cpu"],
                lambda _: None,
                events.append,
            )
        )
        await until(lambda: bool(events))
        assert "counts" in events[0]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert child.process.returncode is not None
        assert snapshot(tmp_path / "output")["counts"]["running"] == 0

    asyncio.run(check())


def test_tui_requires_terminal(cli):
    result = cli("tui")
    assert result.returncode == 1
    assert "需要交互终端" in result.stderr


def test_snapshot_bad_database_surfaces_error(tmp_path):
    state = tmp_path / ".video-watermaker"
    state.mkdir()
    (state / "queue.sqlite3").write_text("bad db")
    with pytest.raises(sqlite3.DatabaseError):
        snapshot(tmp_path)
