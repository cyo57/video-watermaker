#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Video Watermaker - 视频缩略图生成工具

功能：根据指定的行列数自动计算截屏数量，生成网格状视频缩略图
"""

import argparse
import os
import sys
import cv2
import datetime
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import logging
from pathlib import Path
import watermark

# 设置图像像素限制
Image.MAX_IMAGE_PIXELS = None

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_watermaker.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class VideoWatermaker:
    def __init__(self, font_path=None):
        self.font_path = font_path or self._find_default_font()
        
    def _find_default_font(self):
        """寻找默认字体"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        font_dirs = [
            os.path.join(script_dir, 'fonts'),
            'fonts', 
            './fonts'
        ]
        font_names = [
            'Alibaba-PuHuiTi-Medium.ttf',
            '杨任东竹石体-Heavy.ttf', 
            'DejaVuSans.ttf',
            'Arial.ttf'
        ]
        
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for font_name in font_names:
                    font_path = os.path.join(font_dir, font_name)
                    if os.path.exists(font_path):
                        return font_path
        return None
    
    def create_thumbnail(self, video_path, output_path, rows=4, cols=4, 
                        width=3840, rotate=0, quality=80, watermark_text=None):
        """
        生成视频网格状缩略图
        
        Args:
            video_path: 视频文件路径
            output_path: 输出图片路径
            rows: 行数
            cols: 列数  
            width: 输出图片宽度
            rotate: 旋转角度(0,90,180,270)
            quality: JPEG质量(1-100)
            watermark_text: 水印文字，如果为None则不添加水印
        """
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            return False
            
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {video_path}")
                return False
                
            # 获取视频信息
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = round(cap.get(cv2.CAP_PROP_FPS), 4)
            duration = int(frames / fps) if fps > 0 else 0
            width_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if frames == 0 or duration < 5:
                logger.warning(f"视频过短或无有效帧: {video_path}")
                return False
                
            cap.release()
            
            # 根据行列数计算截屏数量和间隔
            total_screenshots = rows * cols
            interval = max(1, duration // total_screenshots)
            
            logger.info(f"视频: {os.path.basename(video_path)}")
            logger.info(f"分辨率: {width_orig}x{height_orig}, 时长: {duration}s, FPS: {fps}fps")
            logger.info(f"生成 {total_screenshots} 个截图 ({rows}x{cols}), 间隔: {interval}s")
            
            # 计算缩略图尺寸
            thumbnail_width = width
            scale = width / width_orig
            thumbnail_height = int(height_orig * scale)
            each_width = width // cols
            each_height = thumbnail_height // rows if rows > 1 else thumbnail_height
            
            # 创建总图 (简化信息栏高度)
            info_height = 50
            full_img = Image.new('RGB', (thumbnail_width, thumbnail_height + info_height), 'white')
            
            # 设置字体
            font_size = max(14, width // 100)
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, font_size)
                small_font = ImageFont.truetype(self.font_path, font_size - 2)
            else:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
                
            # 添加视频信息 - 单行显示（无背景色）
            draw = ImageDraw.Draw(full_img)
            file_size = os.path.getsize(video_path)
            size_str = self._format_file_size(file_size)
            duration_str = self._format_duration(duration)
            
            # 格式化单行信息：文件名 | 大小 | 时长
            filename = os.path.basename(video_path)
            info_text = f"{filename} | {size_str} | {duration_str}"
            
            # 居中显示信息（无背景色）
            text_bbox = draw.textbbox((0, 0), info_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (thumbnail_width - text_width) // 2
            text_y = (info_height - font_size) // 2
            
            # 直接绘制文字（无背景）
            draw.text((text_x, text_y), info_text, fill='black', font=font)
            
            # 截取视频帧
            cap = cv2.VideoCapture(video_path)
            for i in range(total_screenshots):
                timestamp = i * interval + interval // 2
                frame_number = int(timestamp * fps)
                
                if frame_number >= frames:
                    break
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cv2.VideoCapture.read(cap)
                
                if ret:
                    # 转换颜色空间并调整大小
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    
                    # 先调整大小，再旋转（提高性能）
                    pil_img = pil_img.resize((each_width, each_height), Image.Resampling.LANCZOS)
                    
                    # 旋转处理
                    if rotate == 90:
                        pil_img = pil_img.rotate(90, expand=True)
                    elif rotate == 180:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif rotate == 270:
                        pil_img = pil_img.rotate(270, expand=True)
                    
                    # 计算粘贴位置
                    row = i // cols
                    col = i % cols
                    x = col * each_width
                    y = info_height + row * each_height
                    full_img.paste(pil_img, (x, y))
                    
                    # 添加时间戳（无背景色）
                    time_text = self._format_duration(timestamp)
                    
                    # 计算时间戳尺寸和位置
                    time_bbox = small_font.getbbox(time_text)
                    time_width = time_bbox[2] - time_bbox[0]
                    time_height = time_bbox[3] - time_bbox[1]
                    
                    # 确保时间戳不超出画面边界
                    padding = 5
                    max_x = x + each_width - time_width - padding
                    max_y = y + each_height - time_height - padding
                    
                    text_x = min(max(x + padding, max_x), x + each_width - time_width - padding)
                    text_y = min(y + padding, max_y)
                    
                    # 直接绘制白色文字（无背景）
                    draw.text((text_x, text_y), time_text, fill='white', font=small_font)
            
            cap.release()
            
            # 添加水印
            if watermark_text:
                try:
                    # 创建水印对象
                    wm = watermark.Watermark(
                        text=watermark_text,
                        font_path=self.font_path or 'fonts/Alibaba-PuHuiTi-Medium.ttf',
                        color=(255, 255, 255),  # 白色文字
                        opacity=0.8,  # 80% 不透明度
                        size_ratio=0.04  # 文字大小为图片宽度的 4%
                    )
                    # 添加水印
                    full_img = wm.add(full_img)
                    logger.info(f"水印已添加: {watermark_text}")
                except Exception as e:
                    logger.warning(f"添加水印失败: {str(e)}")
                    # 水印失败不影响主程序继续执行
            
            # 保存图片
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            full_img.save(output_path, 'JPEG', quality=quality)
            logger.info(f"缩略图已保存: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"处理视频时出错 {video_path}: {str(e)}")
            return False
    
    def _format_duration(self, seconds):
        """格式化时长显示"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _format_file_size(self, size_bytes):
        """格式化文件大小"""
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = float(size_bytes)
        for unit in units:
            if size < 1024 or unit == units[-1]:
                return f"{size:.1f} {unit}"
            size /= 1024
    
    def process_directory(self, input_path, output_path, **kwargs):
        """处理目录中的所有视频文件"""
        if not os.path.exists(input_path):
            logger.error(f"输入路径不存在: {input_path}")
            return
            
        # 支持的视频格式
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
        
        processed = 0
        failed = 0
        
        if os.path.isfile(input_path):
            # 单个文件
            if any(input_path.lower().endswith(ext) for ext in video_extensions):
                output_file = os.path.join(output_path, os.path.splitext(os.path.basename(input_path))[0] + '_thumbnail.jpg')
                os.makedirs(output_path, exist_ok=True)
                
                if self.create_thumbnail(input_path, output_file, **kwargs):
                    processed += 1
                else:
                    failed += 1
        else:
            # 目录处理
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        video_path = os.path.join(root, file)
                        relative_path = os.path.relpath(root, input_path)
                        output_subdir = os.path.join(output_path, relative_path) if relative_path != '.' else output_path
                        output_file = os.path.join(output_subdir, os.path.splitext(file)[0] + '_thumbnail.jpg')
                        
                        if self.create_thumbnail(video_path, output_file, **kwargs):
                            processed += 1
                        else:
                            failed += 1
        
        logger.info(f"处理完成! 成功: {processed}, 失败: {failed}")

def main():
    parser = argparse.ArgumentParser(
        description="Video Watermaker - 视频缩略图生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 生成4x4网格缩略图
  python main.py input_video.mp4 -o output/ --rows 4 --cols 4
  
  # 自定义3x5网格
  python main.py video.mp4 -o output/ --rows 3 --cols 5
  
  # 批量处理目录
  python main.py /path/to/videos/ -o /path/to/output/ --rows 2 --cols 6
  
  # 设置输出宽度和旋转
  python main.py video.mp4 -o output/ --width 1920 --rotate 90 --rows 5 --cols 3
  
  # 添加水印
  python main.py video.mp4 -o output/ --watermark_text "by VideoWatermaker"
        """
    )
    
    parser.add_argument('input', help='输入视频文件或目录')
    parser.add_argument('-o', '--output', default='output', help='输出目录 (默认: output)')
    parser.add_argument('--rows', type=int, default=4, help='网格行数 (默认: 4)')
    parser.add_argument('--cols', type=int, default=4, help='网格列数 (默认: 4)')
    parser.add_argument('--width', type=int, default=3840, help='输出图片宽度像素 (默认: 3840)')
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270], default=0, help='旋转角度 (默认: 0)')
    parser.add_argument('--quality', type=int, default=80, help='JPEG质量 1-100 (默认: 80)')
    parser.add_argument('--font', help='字体文件路径')
    parser.add_argument('--watermark_text', help='水印文字 (默认: 无水印)')
    
    args = parser.parse_args()
    
    # 创建处理器并处理视频
    try:
        watermaker = VideoWatermaker(font_path=args.font)
        watermaker.process_directory(
            args.input, args.output,
            rows=args.rows, cols=args.cols, width=args.width,
            rotate=args.rotate, quality=args.quality, watermark_text=args.watermark_text
        )
    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
