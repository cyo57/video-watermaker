"""Chinese terminal UI over the durable CLI queue."""

import asyncio
import signal
import sqlite3
from functools import partial
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from .tui_backend import (
    DEFAULTS,
    KINDS,
    PAGE_SIZE,
    STATUSES,
    JobProcess,
    command_args,
    load_settings,
    model_status,
    save_settings,
    settings_path,
    snapshot,
)


class VideoApp(App):
    TITLE = "Video Watermaker"
    SUB_TITLE = "视频目录队列 · 压缩 / 缩略图 / 对话"
    CSS_PATH = "tui.tcss"
    BINDINGS: ClassVar = [
        Binding("ctrl+s", "start", "开始 / 续跑", priority=True),
        Binding("ctrl+x", "stop", "停止", priority=True),
        Binding("ctrl+r", "refresh", "刷新队列", priority=True),
        Binding("ctrl+q", "quit", "退出", priority=True),
        Binding("ctrl+c", "interrupt", "停止 / 退出", priority=True),
    ]

    def __init__(self, source=None, output=None, *, config=None):
        super().__init__()
        self.config_path = settings_path() if config is None else config
        self.values = load_settings(self.config_path)
        if source is not None:
            self.values["source"] = str(source)
        if output is not None:
            self.values["output"] = str(output)
        self.busy = False
        self.quitting = False
        self.controller = None
        self.active = {}
        self.rows = {}
        self.page = 0
        self.total = 0
        self.refreshing = False
        self.view_key = None
        self.table_signature = None
        self.last_error = None
        self.last_exit = None
        self.old_signals = {}

    def field(self, label, key, *, placeholder="", options=None):
        with Horizontal(classes="field"):
            yield Label(label)
            if options:
                yield Select(options, value=self.values[key], allow_blank=False, id=key)
            else:
                yield Input(self.values[key], placeholder=placeholder, id=key)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("填写路径后开始；再次使用相同设置即可续跑。", id="status", markup=False)
        yield Static("输出目录队列：尚无任务", id="counts", markup=False)
        yield ProgressBar(total=1, show_eta=False, id="overall")
        with TabbedContent(initial="settings", id="tabs"):
            with TabPane("设置", id="settings"), VerticalScroll(id="form"):
                yield Static("支持单个视频或递归目录；粘贴路径即可，空格和中文无需加引号。", classes="hint")
                yield from self.field("输入路径", "source", placeholder="/path/to/目录 A 或 movie.mp4")
                yield from self.field("输出目录", "output", placeholder="/path/to/目录 B")
                yield from self.field(
                    "处理方式",
                    "mode",
                    options=[
                        ("压缩视频", "compress"),
                        ("生成缩略图", "thumbnail"),
                        ("提取对话", "transcribe"),
                    ],
                )
                with Horizontal(id="extras"):
                    yield Checkbox("同时生成缩略图", self.values["thumbnail"], id="thumbnail")
                    yield Checkbox("同时提取对话", self.values["transcribe"], id="transcribe")
                yield Static(
                    "压缩上限：720p · 30fps · 视频 1.5 Mbps；保持比例，不放大小视频。", classes="hint"
                )
                with Collapsible(title="媒体参数", collapsed=False, id="media-options"):
                    yield from self.field("媒体并发", "workers")
                    yield from self.field(
                        "视频编码器",
                        "encoder",
                        options=[
                            ("自动（NVIDIA 优先）", "auto"),
                            ("CPU / x264", "cpu"),
                            ("NVIDIA / NVENC", "nvenc"),
                        ],
                    )
                with Collapsible(title="缩略图与水印", id="thumbnail-options"):
                    yield from self.field("网格行数", "rows")
                    yield from self.field("网格列数", "cols")
                    yield from self.field("图片总宽度", "width")
                    yield from self.field("JPEG 质量", "quality")
                    yield from self.field(
                        "逆时针旋转", "rotate", options=[(f"{n}°", str(n)) for n in (0, 90, 180, 270)]
                    )
                    yield from self.field("文字水印", "watermark", placeholder="可留空，仅添加到缩略图")
                    yield from self.field("字体路径", "font", placeholder="留空使用内置字体")
                with Collapsible(title="对话转写与说话人", id="speech-options"):
                    yield from self.field(
                        "识别模型",
                        "model",
                        options=[
                            ("large-v3-turbo（速度优先）", "large-v3-turbo"),
                            ("large-v3", "large-v3"),
                        ],
                    )
                    yield from self.field("识别语言", "language", placeholder="zh / en / auto")
                    yield from self.field(
                        "推理设备",
                        "device",
                        options=[
                            ("自动（CUDA 优先）", "auto"),
                            ("CUDA", "cuda"),
                            ("CPU", "cpu"),
                        ],
                    )
                    yield from self.field("说话人数", "speakers", placeholder="留空自动判断；可填 2、3 等")
                    yield Static(
                        "每个视频标记 A/B/C，输出 TXT、SRT、JSON。转写单并发，标签需要复核。",
                        classes="hint",
                    )
                    yield Static("", id="model-status", markup=False)
                    yield Button("准备所选模型", id="prepare")
                    yield Static(
                        "首次下载需在 Hugging Face 接受 community-1 条款并在本机登录。", classes="hint"
                    )
            with TabPane("队列", id="queue"):
                with Horizontal(id="filters"):
                    yield Select(
                        [("全部状态", "all"), *((label, key) for key, label in STATUSES.items())],
                        value="all",
                        allow_blank=False,
                        id="filter",
                    )
                    yield Input(placeholder="按输入路径搜索（含子目录）", id="search")
                yield Static("当前任务：—", id="active", markup=False)
                yield DataTable(id="tasks", cursor_type="row", zebra_stripes=True)
                with Horizontal(id="pagination"):
                    yield Button("上一页", id="previous")
                    yield Static("第 1 / 1 页 · 每页 100 条", id="page", markup=False)
                    yield Button("下一页", id="next")
                with VerticalScroll(id="details-scroll"):
                    yield Static("选中任务查看完整路径和失败原因。", id="details", markup=False)
            with TabPane("日志", id="logs"):
                yield Static(
                    "显示本次会话最近 1000 行；完整处理日志在输出目录的 .video-watermaker/run.log。",
                    classes="hint",
                )
                yield RichLog(
                    id="log", max_lines=1000, wrap=True, min_width=20, markup=False, highlight=False
                )
        with Horizontal(id="actions"):
            yield Button("开始 / 续跑", id="start", variant="primary")
            yield Button("停止", id="stop", variant="warning", disabled=True)
            yield Button("重试失败", id="retry")
            yield Button("退出", id="quit")
        yield Footer()

    def on_mount(self):
        self.theme = "textual-dark"
        table = self.query_one("#tasks", DataTable)
        for label, width in (("ID", 7), ("类型", 8), ("状态", 8), ("进度", 7), ("视频 / 输入路径", None)):
            table.add_column(label, width=width)
        self.update_options()
        self.update_model_status()
        self.query_one("#overall").tooltip = "已结束任务的比例（成功 + 失败）；是否成功请看计数和日志。"
        self.set_interval(1, self.action_refresh)
        self.action_refresh()
        self.query_one("#source", Input).focus()
        # Terminal close / external SIGTERM uses the same graceful stop as the buttons.
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGHUP):
            try:
                previous = signal.getsignal(sig)
                loop.add_signal_handler(sig, self.action_quit)
                self.old_signals[sig] = previous
            except (NotImplementedError, ValueError, RuntimeError):
                pass

    def on_unmount(self):
        if self.controller:
            self.controller.stop()
        loop = asyncio.get_running_loop()
        for sig, handler in self.old_signals.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, handler)

    def on_resize(self, event):
        self.screen.set_class(event.size.height < 30, "compact")

    def collect(self):
        result = {}
        for key, default in DEFAULTS.items():
            widget = self.query_one(f"#{key}")
            result[key] = widget.value if isinstance(default, bool) else str(widget.value)
        return result

    def update_options(self):
        mode = self.query_one("#mode", Select).value
        compression = mode == "compress"
        thumbnail = mode == "thumbnail" or (compression and self.query_one("#thumbnail", Checkbox).value)
        speech = mode == "transcribe" or (compression and self.query_one("#transcribe", Checkbox).value)
        self.query_one("#extras").display = compression
        self.query_one("#media-options").disabled = mode == "transcribe"
        self.query_one("#encoder").disabled = not compression
        self.query_one("#thumbnail-options").disabled = not thumbnail
        # Model preparation is available even before selecting transcription.
        self.query_one("#speech-options", Collapsible).collapsed = not speech

    def update_model_status(self):
        model = self.query_one("#model", Select).value
        message, _ = model_status(model)
        self.query_one("#model-status", Static).update(message)

    @on(Select.Changed, "#mode")
    @on(Checkbox.Changed, "#thumbnail")
    @on(Checkbox.Changed, "#transcribe")
    def options_changed(self):
        self.update_options()

    @on(Select.Changed, "#model")
    def model_changed(self):
        self.update_model_status()

    @on(Select.Changed, "#filter")
    @on(Input.Changed, "#search")
    @on(Input.Changed, "#output")
    def query_changed(self):
        self.page = 0
        self.action_refresh()

    def write_log(self, message):
        self.query_one("#log", RichLog).write(Text(message))

    def status(self, message):
        self.query_one("#status", Static).update(message)

    def set_busy(self, value):
        self.busy = value
        self.query_one("#form").disabled = value
        for button in ("start", "retry"):
            self.query_one(f"#{button}").disabled = value
        self.query_one("#stop").disabled = not value

    def action_start(self):
        self.start_job()

    def start_job(self, retry=False):
        if self.busy:
            return
        values = self.collect()
        try:
            args = command_args(values, retry=retry)
        except (ValueError, OSError) as exc:
            self.status(str(exc))
            self.notify(str(exc), title="请检查设置", severity="error")
            self.query_one("#tabs", TabbedContent).active = "settings"
            return
        self.values = values
        # Remember absolute paths so launching from another directory resumes the same queue.
        values["source"], values["output"] = args[1], args[3]
        self.query_one("#source", Input).value = values["source"]
        self.query_one("#output", Input).value = values["output"]
        try:
            save_settings(self.config_path, values)
        except OSError as exc:
            self.write_log(f"无法保存界面设置：{exc}；当前任务仍可继续")
        self.launch(args, "准备扫描目录…", "queue")

    def launch(self, args, message, tab):
        self.active = {}
        self.last_exit = None
        self.controller = JobProcess()
        self.set_busy(True)
        self.status(message)
        self.write_log("开始：" + " · ".join(args))
        self.query_one("#tabs", TabbedContent).active = tab
        self.run_worker(partial(self.run_job, args), group="job", name="处理队列", exit_on_error=False)

    async def run_job(self, args):
        def logged(message):
            self.write_log(message)
            if not self.controller.stopping.is_set():
                self.status(message[-250:])

        try:
            code = await self.controller.run(args, logged, self.progress_received)
            self.last_exit = code
            if self.controller.stopping.is_set() or code == 130:
                message = "已停止，队列已保留。使用相同设置开始即可续跑。"
            elif code == 0:
                message = "模型准备完成，可离线转写。" if args[0] == "models" else "本次处理完成。"
            else:
                message = f"本次处理未全部完成（退出码 {code}）；请查看日志及失败任务。"
            self.status(message)
            self.write_log(message)
        except Exception as exc:  # noqa: BLE001 -- keep the UI available to inspect failed child runs.
            self.last_exit = 1
            self.status(f"无法运行：{exc}")
            self.write_log(f"无法运行：{exc}")
        finally:
            self.active = {}
            self.query_one("#active", Static).update("当前任务：—")
            self.set_busy(False)
            self.update_model_status()
            if self.quitting:
                self.exit()
            else:
                self.action_refresh()

    def progress_received(self, event):
        self.active = {item["id"]: item for item in event.get("active", [])}
        items = list(self.active.values())
        detail = " | ".join(
            f"{KINDS[item['kind']]} {item['name']} {item['fraction']:.0%}" for item in items[:3]
        )
        if len(items) > 3:
            detail += f" …共 {len(items)} 个"
        self.query_one("#active", Static).update("当前任务：" + (detail or "切换任务中…"))
        if not self.controller.stopping.is_set():
            counts = event["counts"]
            self.status(
                f"本次队列：成功 {counts['succeeded']} · 失败 {counts['failed']} · "
                f"运行 {counts['running']} · 等待 {counts['pending']}"
            )

    def action_stop(self):
        if self.busy and self.controller:
            self.controller.stop()
            self.status("正在停止并保存队列，请稍候…")
            self.query_one("#stop").disabled = True

    def action_interrupt(self):
        if self.busy:
            self.action_stop()
        else:
            self.action_quit()

    def action_quit(self):
        if self.busy:
            self.quitting = True
            self.action_stop()
        else:
            self.exit()

    @on(Button.Pressed)
    def button_pressed(self, event):
        match event.button.id:
            case "start":
                self.action_start()
            case "stop":
                self.action_stop()
            case "retry":
                self.start_job(retry=True)
            case "quit":
                self.action_quit()
            case "prepare":
                if self.busy:
                    return
                model = self.query_one("#model", Select).value
                message, available = model_status(model)
                if not available:
                    self.notify(message, title="需要语音依赖", severity="error", timeout=10)
                    return
                self.launch(
                    ["models", "prepare", "--model", model], "准备模型，首次下载可能需要较长时间…", "logs"
                )
            case "previous":
                self.page = max(0, self.page - 1)
                self.action_refresh()
            case "next":
                self.page = min(max(0, (self.total - 1) // PAGE_SIZE), self.page + 1)
                self.action_refresh()

    def action_refresh(self):
        if not self.is_mounted or self.refreshing or self.quitting:
            return
        self.refreshing = True
        self.run_worker(self.refresh_queue, group="view", exit_on_error=False)

    def query_key(self):
        return (
            self.query_one("#output", Input).value,
            self.query_one("#filter", Select).value,
            self.query_one("#search", Input).value,
            self.page,
        )

    async def refresh_queue(self):
        key = self.query_key()
        try:
            output = Path(key[0]).expanduser().resolve()
            result = await asyncio.to_thread(snapshot, output, key[1], key[2], key[3])
            if key != self.query_key():
                return
            self.last_error = None
            self.total, self.page = result["total"], result["page"]
            counts = result["counts"]
            total = sum(counts.values())
            self.query_one("#counts", Static).update(
                f"输出目录队列 · 共 {total} 项：成功 {counts['succeeded']} · 失败 {counts['failed']} · "
                f"运行 {counts['running']} · 等待 {counts['pending']}"
            )
            done = counts["succeeded"] + counts["failed"]
            self.query_one("#overall", ProgressBar).update(total=max(1, total), progress=done)
            pages = max(1, (self.total + PAGE_SIZE - 1) // PAGE_SIZE)
            self.query_one("#page", Static).update(f"第 {self.page + 1} / {pages} 页 · 筛选 {self.total} 项")
            self.query_one("#previous").disabled = self.page == 0
            self.query_one("#next").disabled = self.page + 1 >= pages
            self.render_rows(result["rows"], key[:3])
        except (OSError, ValueError, sqlite3.Error) as exc:
            if key != self.query_key():
                return
            message = f"暂时无法读取队列：{exc}"
            self.query_one("#counts", Static).update(message)
            self.render_rows([], key[:3])
            self.query_one("#overall", ProgressBar).update(total=1, progress=0)
            if message != self.last_error:
                self.write_log(message)
                self.last_error = message
        finally:
            self.refreshing = False

    def render_rows(self, rows, key):
        self.rows = {str(row["id"]): row for row in rows}
        cells = []
        for row in rows:
            active = self.active.get(row["id"])
            percent = (
                f"{active['fraction']:.0%}" if active else ("100%" if row["status"] == "succeeded" else "—")
            )
            cells.append(
                (str(row["id"]), KINDS[row["kind"]], STATUSES[row["status"]], percent, row["source"])
            )
        signature = (key, cells, [row["error"] for row in rows])
        if signature == self.table_signature:
            return
        table = self.query_one("#tasks", DataTable)
        cursor = table.cursor_row
        selected = None
        if table.row_count and self.view_key == key:
            selected = str(table.get_row_at(cursor)[0])
        self.view_key, self.table_signature = key, signature
        table.clear()
        for cell in cells:
            table.add_row(*(Text(value) for value in cell), key=cell[0])
        if selected in self.rows:
            cursor = list(self.rows).index(selected)
        else:
            cursor = 0
        if cells:
            table.move_cursor(row=cursor)
            self.show_detail(cells[cursor][0])
        else:
            self.query_one("#details", Static).update("当前没有符合条件的任务。")

    @on(DataTable.RowHighlighted, "#tasks")
    def row_highlighted(self, event):
        self.show_detail(str(event.row_key.value))

    def show_detail(self, key):
        row = self.rows.get(key)
        if row:
            message = (
                f"输入：{row['source']}\n输出：{row['target']}\n"
                f"尝试次数：{row['attempts']} · {STATUSES[row['status']]}"
            )
            if row["kind"] == "transcribe":
                message += " · 同时导出同名 .txt / .srt"
            if row["error"]:
                message += "\n原因：" + row["error"]
            self.query_one("#details", Static).update(message)
