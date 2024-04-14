from captcha.image import ImageCaptcha
import random
import string
import os

replace_rule = {
    "O": "0",
    "1": "I",
    "Z": "2",
}

# 创建一个ImageCaptcha实例
image = ImageCaptcha(width=280, height=90)

# 指定保存验证码图片的目录
output_dir = './test/recaptcha/pictures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 创建并打开filename.lst文件
with open(os.path.join(output_dir, 'filename.lst'), 'w') as f:
    # 生成100张验证码图片
    for i in range(100):
        # 生成随机的四位验证码
        captcha_text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        for k, v in replace_rule.items():
            captcha_text = captcha_text.replace(k, v)
        # 生成验证码图片
        data = image.generate(captcha_text)
        # 构建图片文件名
        file_name = f"{captcha_text}.png"
        # 保存验证码图片
        image.write(captcha_text, os.path.join(output_dir, file_name))
        # 将文件名写入filename.lst
        f.write(file_name + '\n')

print(f"100张验证码图片已生成在{output_dir}目录中，并且文件名已记录在filename.lst文件中。")