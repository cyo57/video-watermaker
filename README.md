# Video Watermaker

一个强大的视频缩略图生成工具，支持生成网格状缩略图和提取关键帧。

## 功能特点

- 🎬 **网格缩略图**: 根据指定的行列数自动计算截屏数量，生成网格状视频缩略图
- 🖼️ **关键帧提取**: 从视频中提取指定数量的关键帧
- 📁 **批量处理**: 支持单个文件或整个目录的批量处理
- ⚙️ **灵活配置**: 通过命令行参数自定义行列数、分辨率、旋转角度等
- 🏃 **高性能**: 基于OpenCV和PIL优化，支持多种视频格式
- 📝 **详细日志**: 完整的处理日志和进度显示

## 支持的视频格式

支持所有OpenCV兼容的视频格式：
- MP4, AVI, MKV, MOV, WMV, FLV, WEBM, M4V, 3GP等

## 安装依赖

```bash
# 使用 uv (推荐)
uv add opencv-python Pillow numpy

# 或使用 pip
pip install opencv-python Pillow numpy
```

## 使用方法

### 基本语法

```bash
python main.py <命令> <输入路径> [选项]
```

### 命令说明

#### 1. 生成网格缩略图 (`thumbnail`)

```bash
# 生成默认4x4网格缩略图
python main.py thumbnail input_video.mp4 -o output/

# 自定义3x5网格
python main.py thumbnail input_video.mp4 -o output/ --rows 3 --cols 5

# 设置输出宽度和旋转角度
python main.py thumbnail video.mp4 -o output/ --width 1920 --rotate 90

# 批量处理整个目录
python main.py thumbnail /path/to/videos/ -o /path/to/output/ --rows 2 --cols 6
```

#### 2. 提取关键帧 (`keyframes`)

```bash
# 提取10个关键帧
python main.py keyframes input_video.mp4 -o output/

# 提取20个关键帧，设置输出宽度
python main.py keyframes video.mp4 -o output/ --count 20 --width 800

# 批量提取关键帧
python main.py keyframes /path/to/videos/ -o /path/to/output/ --count 15
```

### 参数说明

#### 缩略图命令参数 (`thumbnail`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--rows` | 4 | 网格行数 |
| `--cols` | 4 | 网格列数 |
| `--width` | 3840 | 输出图片宽度(像素) |
| `--rotate` | 0 | 旋转角度(0/90/180/270度) |
| `--quality` | 80 | JPEG质量(1-100) |
| `--font` | - | 字体文件路径(可选) |

#### 关键帧命令参数 (`keyframes`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--count` | 10 | 提取关键帧数量 |
| `--width` | 0 | 输出宽度(0为保持原尺寸) |
| `--rotate` | 0 | 旋转角度(0/90/180/270度) |

## 使用示例

### 示例1: 生成视频预览图

```bash
# 为单个视频生成3x3缩略图
python main.py thumbnail movie.mp4 -o ./thumbnails/ --rows 3 --cols 3 --width 1920

# 输出: thumbnails/movie_thumbnail.jpg
```

### 示例2: 批量处理视频库

```bash
# 处理整个视频目录，生成2x4网格缩略图
python main.py thumbnail ./videos/ -o ./output/ --rows 2 --cols 4 --quality 90

# 输出目录结构:
# output/
#   ├── video1_thumbnail.jpg
#   ├── video2_thumbnail.jpg
#   └── subdir/
#       └── video3_thumbnail.jpg
```

### 示例3: 提取关键帧制作相册

```bash
# 从视频中提取15个关键帧，每帧宽度800像素
python main.py keyframes presentation.mp4 -o ./frames/ --count 15 --width 800

# 输出: frames/keyframes_presentation/frame_0001_00-00-05.jpg
```

### 示例4: 旋转视频处理

```bash
# 处理竖屏视频，旋转90度后生成缩略图
python main.py thumbnail vertical_video.mp4 -o output/ --rotate 90 --rows 5 --cols 3

# 处理横屏视频，旋转180度
python main.py thumbnail video.mp4 -o output/ --rotate 180 --rows 2 --cols 5
```

## 核心算法

### 智能截屏计算

新版本采用智能算法，根据用户指定的行列数自动计算截屏间隔：

```
总截屏数 = 行数 × 列数
截屏间隔 = 视频时长 ÷ 总截屏数
```

这种方法确保了：
- 🎯 **精准控制**: 完全按用户指定的行列布局
- ⚡ **均匀分布**: 截屏在时间轴上均匀分布
- 🚀 **高效处理**: 无需手动调整时间间隔

### 性能优化

- **内存优化**: 流式处理大视频，避免内存溢出
- **并行处理**: 支持多文件批量处理
- **错误恢复**: 自动跳过损坏的视频文件
- **进度显示**: 实时显示处理进度

## 项目结构

```
video-watermaker/
├── main.py                    # 主程序入口
├── get_video_thumb.py         # 原始缩略图生成脚本(已废弃)
├── get_video_thumb_pic.py     # 原始关键帧脚本(已废弃)
├── pyproject.toml            # 项目配置
├── fonts/                    # 字体文件目录
│   ├── 杨任东竹石体-Heavy.ttf
│   └── Alibaba-PuHuiTi-Medium.ttf
├── sample/                   # 示例文件
├── video_watermaker.log      # 程序日志
└── README.md                # 本文档
```

## 故障排除

### 常见问题

1. **"视频文件无法打开"**
   - 检查视频文件是否损坏
   - 确认视频格式是否被OpenCV支持

2. **"字体文件找不到"**
   - 检查 `fonts/` 目录是否存在
   - 使用 `--font` 参数指定字体路径

3. **"内存不足"**
   - 减小 `--width` 参数值
   - 分批处理大文件

4. **"输出目录权限错误"**
   - 检查输出目录写入权限
   - 确保磁盘空间充足

### 日志查看

程序运行时会生成详细日志：

```bash
# 实时查看日志
tail -f video_watermaker.log
```

## 更新日志

### v2.0.0 (2025-11-13)
- ✨ **全新架构**: 重写代码，采用面向对象设计
- 🎯 **智能计算**: 支持按行列数自动计算截屏数量
- ⚡ **命令行界面**: 完整的argparse命令行参数支持
- 🗂️ **批量处理**: 支持目录递归批量处理
- 🏃 **性能优化**: 显著提升处理速度和内存使用
- 🐛 **错误处理**: 增强错误处理和日志记录
- 📖 **文档完善**: 详细的使用手册和示例

### v1.0.0
- 基础缩略图生成功能
- 关键帧提取功能

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 支持

如果这个项目对你有帮助，请给个⭐支持一下！

---

**享受视频处理的乐趣！** 🎬✨
