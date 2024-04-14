from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import string
import os

# 设置验证码图片的大小
width, height = 160, 60
# 设置字体大小
font_size = 36

# 生成随机字符串作为验证码
def generate_captcha_text(length=4):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

# 添加干扰线
def add_noise(draw):
    for _ in range(random.randint(1, 3)):  # 线条数量
        start = (random.randint(0, width), random.randint(0, height))
        end = (random.randint(0, width), random.randint(0, height))
        draw.line([start, end], fill=(0, 0, 0), width=2)

# 添加噪点
def add_noise_dots(draw):
    for _ in range(random.randint(100, 200)):  # 噪点数量
        xy = (random.randint(0, width), random.randint(0, height))
        draw.point(xy, fill=(0, 0, 0))

# 生成验证码图片
def generate_captcha_image(text):
    # 创建一个新的图片对象
    image = Image.new('RGB', (width, height), (255, 255, 255))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(image)
    # 设置字体（使用默认字体）
    font = ImageFont.truetype("arial.ttf", size=font_size)
    # 在图片上绘制文本
    draw.text((10, 5), text, font=font, fill=(0, 0, 0))
    # 添加干扰线
    add_noise(draw)
    # 添加噪点
    add_noise_dots(draw)
    # 应用滤镜
    image = image.filter(ImageFilter.GaussianBlur(1))
    
    return image

# 主函数
def main(output_dir, number_of_captchas=100):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建并打开filename.lst文件
    with open(os.path.join(output_dir, 'filename.lst'), 'w') as f:
        for _ in range(number_of_captchas):
            # 生成验证码文本
            captcha_text = generate_captcha_text()
            # 生成验证码图片
            captcha_image = generate_captcha_image(captcha_text)
            # 构建图片文件名
            file_name = f"{captcha_text}.png"
            # 保存验证码图片
            captcha_image.save(os.path.join(output_dir, file_name))
            # 将文件名写入filename.lst
            f.write(file_name + '\n')

# 调用主函数
if __name__ == '__main__':
    main('pictures')
    print("Captcha images and filenames have been generated.")