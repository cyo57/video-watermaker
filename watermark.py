from PIL import Image, ImageDraw, ImageFont


class Watermark:
    def __init__(self, text, font_path, color=(255, 255, 255), opacity=0.5, size_ratio=0.05):
        self.text = text
        self.font_path = font_path
        self.color = color
        self.opacity = opacity
        self.size_ratio = size_ratio

    def add(self, image: Image.Image) -> Image.Image:
        """
        向图片添加水印。

        Args:
            image: PIL.Image.Image 对象。

        Returns:
            添加水印后的 PIL.Image.Image 对象。
        """
        # 创建一个透明的水印层
        watermark_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark_layer)

        # 根据图片宽度和比例计算字体大小
        font_size = int(image.width * self.size_ratio)
        font = ImageFont.truetype(self.font_path, font_size)

        # 计算文本尺寸
        text_bbox = draw.textbbox((0, 0), self.text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # 计算居中位置
        x = (image.width - text_width) // 2
        y = (image.height - text_height) // 2

        # 添加阴影
        shadow_offset = 4
        shadow_color = (0, 0, 0, int(255 * self.opacity * 0.5))
        draw.text((x + shadow_offset, y + shadow_offset),
                  self.text, font=font, fill=shadow_color)

        # 绘制带有透明度的文本
        fill_color = self.color + (int(255 * self.opacity),)
        draw.text((x, y), self.text, font=font, fill=fill_color)

        # 合成水印
        image = image.convert("RGBA")
        composited_image = Image.alpha_composite(image, watermark_layer)
        return composited_image.convert("RGB")
