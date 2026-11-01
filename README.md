# Video Watermaker

面向本地视频目录的终端工具，提供中文交互界面和命令行：批量压缩、生成网格缩略图、添加图片文字水印、提取带说话人标注的对话。任务保存在 SQLite 中，支持递归导入一万个视频、限制并发、中断续跑和失败重试。

## 安装

当前运行环境为 **Linux、Python 3.11–3.13、uv**。系统需安装包含 `libx264`、AAC、PNG 编码器及 `scale` 滤镜的 FFmpeg，并确保 `ffmpeg`、`ffprobe` 在 PATH 中。HDR 转 SDR 另需 FFmpeg 的 `zscale` 和 `tonemap` 滤镜。

```bash
uv sync
uv run main.py --help
```

基础 Python 运行依赖为 Pillow 和 Textual；对话转写另行安装 `speech` 可选依赖。压缩的显卡加速使用系统 FFmpeg 的 `h264_nvenc` 和 NVIDIA 驱动；没有 NVIDIA 显卡也可以使用 CPU。

## 交互终端界面（TUI）

```bash
# 包含对话转写功能（已准备模型时可离线使用）
uv run --extra speech main.py tui

# 只做压缩 / 缩略图，无需语音依赖
uv run main.py tui

# 预填输入与输出；也可只传 -o 查看已有队列
uv run --extra speech main.py tui /path/to/A -o /path/to/B
```

在交互终端中不带参数执行 `main.py` 也会打开 TUI。建议终端至少 **80 列 × 24 行**；支持鼠标、Tab 切换焦点、方向键和 Enter 操作。设置页可滚动，更多参数在折叠栏中。非交互脚本继续使用下文的 CLI 子命令。

- **设置**：填写单个视频或递归目录和输出目录，选择压缩、缩略图或对话转写。压缩时可同时勾选缩略图和转写；支持媒体并发、编码器、网格、水印、语言、模型、设备和说话人数。
- **队列**：显示输出目录的总计数、当前任务百分比，以及每页最多 100 条任务。可以按状态筛选、按输入路径搜索；选中任务查看完整输入／输出路径、尝试次数和失败原因。总进度条表示已结束任务（成功 + 失败）的比例。
- **日志**：保留本次界面会话最近 1000 行；完整处理日志仍写入输出目录的 `.video-watermaker/run.log`。
- **开始 / 续跑**：扫描当前输入并处理未完成任务，自动跳过已成功产物。**重试失败**使用当前设置，额外重新派发该输入中失败的任务；已成功任务仍会跳过。
- **停止 / 退出**：停止派发并终止工作子进程，保留队列和语音检查点。退出会等待子进程收尾，再关闭界面；异常卡住时会在 15 秒后终止工作进程，下次开始时恢复队列。
- **准备所选模型**：在“对话转写与说话人”栏中查看本地模型状态、启动下载和加载检查。首次准备需要联网以及已接受模型条款的本机 Hugging Face 登录凭据。界面不会接收或保存令牌。

| 快捷键 | 操作 |
| --- | --- |
| `Ctrl+S` | 开始 / 续跑 |
| `Ctrl+X` | 停止当前处理 |
| `Ctrl+R` | 刷新队列 |
| `Ctrl+Q` | 安全停止并退出 |
| `Ctrl+C` | 运行时停止，空闲时退出 |

开始任务时会记住当前设置，保存在 `~/.config/video-watermaker/tui.json`（可通过 `XDG_CONFIG_HOME` 改变位置）。输入、输出按绝对路径保存，重开界面后使用原设置即可续跑。界面只分页读取 SQLite；工作任务仍由原有的限流队列执行，不会为一万个视频创建一万个界面组件或进程。运行期间设置锁定，筛选、翻页和查看日志仍可使用。

## 压缩一个目录

```bash
uv run main.py compress /path/to/A -o /path/to/B
```

默认递归扫描 A，分批入库，然后最多同时运行 **2 个任务**。目录再大也不会一次启动所有编码进程。输入也可以是单个视频。

固定输出策略：

| 项目 | 行为 |
| --- | --- |
| 容器与编码 | MP4、H.264、8-bit YUV420、faststart |
| 横屏 | 最大 1280×720，保持比例 |
| 竖屏 | 最大 720×1280，保持比例 |
| 帧率 | 最高 30fps；低帧率保留 |
| 视频码率 | 受限 VBR，`maxrate=1500k`、`bufsize=3000k` |
| 音频 | 第一条音轨，AAC 128 kbps，最多双声道；无音轨则保持无音轨 |
| 视频流 | 第一条非封面视频流；其他视频、音频、字幕和附件不输出 |
| 方向和像素 | 应用旋转元数据，转换为方形像素、偶数尺寸 |
| HDR | 对标记为 PQ／HLG 的素材执行 SDR 色调映射 |

720p 是尺寸上限，小视频不会为了凑尺寸而放大。1.5 Mbps 指**视频码率的 VBV 上限**，不是整文件总码率，也不是每一帧或任意一秒的严格上限；短视频平均码率可能受初始缓冲影响。音频与封装开销另计。所有输入都会重新编码，不承诺每个原本已高度压缩的文件都会进一步变小。

```bash
# 自动选择：NVENC 优先，不可用时回退 x264
uv run main.py compress A -o B --encoder auto

# 强制 CPU 或 NVIDIA；强制 NVIDIA 不会悄悄回退
uv run main.py compress A -o B --encoder cpu
uv run main.py compress A -o B --encoder nvenc

# 调整同时处理的任务数（1–32）；FFmpeg 自身线程数也受限制
uv run main.py compress A -o B --workers 4
```

NVENC 使用 `p5`、VBR、CQ 23，CPU 使用 x264 `medium`、CRF 23。自动模式启动时实际试编码；运行中发生硬件故障时切换到 CPU。个别素材小于显卡支持的最小尺寸时，仅该任务使用 CPU。

## 缩略图与图片水印

```bash
uv run main.py thumbnail A -o B
uv run main.py thumbnail A -o B --rows 3 --cols 5 --width 1920
uv run main.py thumbnail video.mp4 -o B --watermark-text "示例水印"

# 一次做完压缩和缩略图；缩略图从原视频提取
uv run main.py compress A -o B --thumbnail --watermark-text "示例水印"
```

均匀采样完整时间轴，支持不足五秒的短视频。拼图保留单帧比例，显示文件名、大小、时长和采样时间。短视频末尾无法解码时向前寻找可用帧，必要时重复首帧，不输出空白格。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--rows` / `--cols` | 4 / 4 | 网格行列，各为 1–20 |
| `--width` | 3840 | 图片总宽度，64–16384 像素 |
| `--rotate` | 0 | 缩略图额外逆时针旋转 0 / 90 / 180 / 270 度 |
| `--quality` | 80 | JPEG 质量，1–100 |
| `--font` | 项目内置中文字体 | 自定义字体路径 |
| `--watermark-text` | 无 | 居中文字水印，仅作用于缩略图 |

单张拼图最多 5000 万像素，超过时该任务失败并提示减小宽度。缩略图参数不会改变压缩视频的尺寸、旋转或画面；`compress` 配合 `--thumbnail` 时这些参数才作用于缩略图。

兼容旧入口 `uv run main.py A -o B`（等同于 `thumbnail`），以及旧参数名 `--watermark_text`。原文档中并不存在的 `keyframes` 子命令不再宣称支持。

## 提取对话与区分说话人

首次安装、准备模型：

```bash
uv sync --extra speech

# 先在下方模型页面接受使用条款，再登录本机 Hugging Face 账号。
# 也可以在本机通过 HF_TOKEN 环境变量提供令牌，无需写入项目文件。
uv run --extra speech hf auth login
uv run --extra speech main.py models prepare
```

说话人模型使用 [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)，首次下载需要接受其使用条款。语音依赖和模型较大，安装会下载 PyTorch、CUDA 库等数 GB 数据。`models prepare` 检查模型能否加载，并在 `~/.cache/video-watermaker/models.json` 记录模型路径和版本；设置 `XDG_CACHE_HOME` 时使用对应缓存目录。模型权重位于 Hugging Face 缓存，可用 `HF_HOME` 配置。

模型准备完成后，转写进程离线运行，音视频不上传，模型遥测关闭。系统 FFmpeg 解码为波形后传给识别和说话人模型，因此不依赖 TorchCodec 内置解码器对系统 FFmpeg 版本的支持。

```bash
# 一个文件或整个目录，默认递归
uv run --extra speech main.py transcribe A -o B

# 已知是两人普通话对话
uv run --extra speech main.py transcribe A -o B --language zh --speakers 2

# 自动判断语言，强制 CPU
uv run --extra speech main.py transcribe A -o B --language auto --device cpu

# 压缩、缩略图、转写一起排队；三项都读取原视频
uv run --extra speech main.py compress A -o B --thumbnail --transcribe

# 可选较大的识别模型，需另行准备
uv run --extra speech main.py models prepare --model large-v3
uv run --extra speech main.py transcribe A -o B --model large-v3
```

| 参数 | 默认值 | 行为 |
| --- | --- | --- |
| `--model` | `large-v3-turbo` | 速度优先；也支持 `large-v3` |
| `--language` | `zh` | 普通话为主；`auto` 按音频块检测语言，或指定 `en` 等语言代码 |
| `--device` | `auto` | CUDA 优先，GPU 不可用或显存不足时提示并回退 CPU；指定 `cuda` 时不回退 |
| `--speakers` | 自动判断 | 指定人数 1–100，不指定则模型自行判断 |
| `--retry-failed` | 关闭 | 重试失败任务，复用已经完成的识别阶段 |

GPU 识别采用 `int8_float16`，CPU 采用 `int8`。组合任务先执行压缩／缩略图，再串行转写。语音固定一个常驻进程，复用模型并分阶段占用显存；`compress --workers` 只控制媒体任务并发数。进度包含音频提取、分块识别和说话人分离，模型阶段切换时百分比可能重新调整。

每个视频输出三个文件，保留原目录结构：

```text
A/课程/movie.mp4 → B/课程/movie.mp4.transcript.json
                  B/课程/movie.mp4.transcript.txt
                  B/课程/movie.mp4.transcript.srt
```

TXT 示例：

```text
[00:00:01.200 → 00:00:03.400] 说话人 A：今天讨论视频处理。
[00:00:03.600 → 00:00:05.100] 说话人 B：好的，先从转写开始。
```

SRT 带时间戳和说话人名称，可直接作为字幕使用。JSON 的 `schema_version` 为 `1`，记录模型版本、语言、实际使用设备以及 `segments`；每段包括以秒为单位的 `start`、`end`、`text`、`speaker` 和 `overlap`。无法归属的说话人为 `null`，文本导出显示“说话人未知”。无音轨或无识别人声时成功输出空 TXT／SRT，并在 JSON 的 `reason` 中记录 `no_audio`／`no_speech`。

说话人按每个视频中首次出现的顺序标记 A/B/C；长视频对整段音频统一聚类，不会在五分钟识别分块处重新编号。标签不代表真实身份，也不跨视频关联。重叠发言会标记，但模型未必能完整恢复同时说出的两句话；噪声、音乐、相似音色和方言可能导致漏字或误分，需要人工复核。

识别每五分钟保存一次检查点，分块附带两秒上下文，利用重叠区的共同文字和时间戳合并，避免词时间戳的小幅偏差导致边界漏字或重复。块内遇到长停顿时，两侧语音独立识别后还原到原视频时间轴，避免移除静音后将后一句的开头错放到前一句末尾。整段说话人分离完成后再保存该阶段，中途打断会重跑说话人阶段，但不会重复已完成的识别块。三个导出齐全才算成功；导出丢失时使用数据库中的结果补生成，无需再次推理。输入、模型版本或语言／人数等参数改变后，不会复用旧检查点。

完整临时音频存于 `/tmp`，16kHz 单声道 float32 约 **230 MB／小时**；开始前会检查剩余空间。说话人分离的内存需求也随单个视频长度增长，万级队列支持不代表单个无限长视频可在固定内存内处理。临时音频在任务结束后清理，持久检查点保留在输出目录的 SQLite 数据库中。

## 队列、续跑与输出

```bash
# 查看持久队列及最近失败原因；执行期间也能查看
uv run main.py status -o B

# 续跑：重复相同命令
uv run main.py compress A -o B --thumbnail

# 重试失败任务，同时继续未完成任务
uv run main.py compress A -o B --thumbnail --retry-failed
```

- 扫描时每 100 个视频提交一次入库结果；同一输出目录只允许一个处理进程。
- 输入路径、大小、修改时间和处理参数均一致，且成功产物的大小与修改时间未变时，跳过该步骤。输出丢失会重新排队；已知失败默认不会自动重试。
- 压缩、缩略图与转写是独立任务，一步失败不妨碍另一步成功，续跑只补做缺项。语音依赖／模型的预检在任务派发前统一执行，配置错误不会将整个目录逐个标为失败。
- Ctrl+C 或 SIGTERM 停止派发并终止媒体／语音子进程，保留待处理状态。Linux 上父进程被强制终止时，这些子进程也会被终止；再次运行会恢复运行中任务。
- 坏视频单独失败，其余任务继续。空间不足时停止派发新任务，释放空间后带 `--retry-failed` 续跑。
- 递归时排除输出目录和队列目录，不跟随目录或文件符号链接；输入、输出目录不能相同。

原视频不会删除或覆盖。输出保留子目录与完整源文件名，避免不同扩展名撞名：

```text
A/课程/movie.mp4  → B/课程/movie.mp4.compressed.mp4
                  B/课程/movie.mp4.thumbnail.jpg
A/课程/movie.mkv  → B/课程/movie.mkv.compressed.mp4
                  B/课程/movie.mkv.thumbnail.jpg
```

输出名与旧版有所不同。**已有但不能确认与当前任务匹配的输出会报冲突，不覆盖。** 输入内容、参数或字体改变后，要重新生成已有产物，请选择新的输出目录。不要手动删除队列数据库后期待工具继续信任旧输出。

队列与日志位于 `B/.video-watermaker/`：

- `queue.sqlite3`：任务和成品状态；运行时可能有 WAL 伴随文件。
- `run.log`：运行日志，单文件 5 MiB，保留 3 个轮转备份。
- `runner.lock`：运行锁，进程结束后自动释放。

中间产物统一放在 `/tmp/video-watermaker-…/`，每个活动任务需要容纳一份完整成品。跨文件系统发布时，程序独占创建输出并记录文件身份，复制和校验完成后才标记成功；复制中的文件可能已可见，完成状态以队列为准。中断后按记录清理未完成副本；无法确认归属的文件仍报冲突。请在正常退出后再迁移整个输出目录；不要只迁移数据库。

目录识别 MP4、AVI、MKV、MOV、WMV、FLV、WebM、M4V、3GP、TS、MTS、M2TS、MPG、MPEG、VOB、OGV、ASF、MXF（大小写不限），能否解码取决于系统 FFmpeg。显式指定单文件时不限制扩展名。

退出码：`0` 成功或全部已完成，`1` 存在失败／导入冲突／运行错误，`2` 参数错误，`130` 用户中断。`status` 成功读取队列时返回 `0`，具体失败数见输出。

## 开发与验证

```bash
uv sync --group dev
uv run pytest -q --basetemp=/tmp/video-watermaker-pytest
uv run ruff check .
uv run ruff format --check .
```

测试使用 `/tmp` 中的合成素材，覆盖一万个文件入口与模拟任务执行、受限调度、续跑与重试、真实 CPU／NVENC 转码、旋转与非方形像素、变帧率、短视频拼图、中文水印、输出冲突、磁盘空间不足、Ctrl+C 和强制终止恢复。没有可用 NVENC 的机器会跳过对应硬件集成测试；万级调度测试不等于实际压缩一万部影片。

TUI 测试覆盖万级队列的分页与筛选、80×24 布局、键盘操作、真实视频压缩、失败重试、退出后续跑和工作进程回收。界面测试使用 Textual 的无头测试环境，不需要图形桌面，也不下载语音模型。

语音测试无需下载模型，覆盖万级单并发调度、整段说话人编号、边界时间戳、分块检查点、三个输出的恢复、旧数据库升级、真实 FFmpeg 音频提取、GPU 回退与语音进程取消／父进程强制终止。它们验证工程行为，不代表真实识别准确率。真实效果验收需先获得模型权限，使用带人工稿的中文双人／三人、重叠发言和长视频样本核对文字、说话人与时间，记录耗时和显存；模型尚未下载时不能宣称此验收通过。

本机已用 [AISHELL-1](https://huggingface.co/datasets/AISHELL/AISHELL-1) 三位录音者的真实中文录音构造双人、三人、重叠发言、长静音及跨五分钟边界样本，在 RTX 4060 和 CPU 上实际运行模型。测试发现并修复了长停顿后的词时间戳偏移，以及相邻识别块时间戳偏差造成的边界漏字。也确认了模型的效果限制：相似声音、短句切换和长静音后仍可能误分，部分双人短样本被自动合并为一人；指定人数也不能保证逐句归属正确，官方原始流程同样复现。转写和恢复流程可用，A/B/C 标签仍应作为待复核结果。

代码按目录发现、SQLite 存储、调度与发布、媒体处理、缩略图、语音处理、CLI 和 TUI 分离。`main.py` 仅保留入口，也可通过 `uv run python -m video_watermaker` 运行。

许可证见 [LICENSE](LICENSE)。
