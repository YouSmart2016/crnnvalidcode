# 该类实现了4种生成验证的方法
# 第一种方法,该方法是生成最简单的验证码,没有什么干扰线或干扰点的,如下
import numpy as np
from captcha.image import ImageCaptcha

import  tensorflow as tf
import random

items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
random_widths=list(range(64,100,1))
print(random_widths)
print(items)
random.shuffle(random_widths)
for i in range(10):
    img = ImageCaptcha(width=100, height=32, font_sizes=(26, 28, 30))
    random.shuffle(items)
    code=''.join([str(j) for j in  items[:4]])
    im = img.create_captcha_image(chars=code, color='white', background='black')
    im.save('./testValid/{}.jpg'.format(code))