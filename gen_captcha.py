# -*- coding:utf-8 -*-
import argparse
import json
import string
import os
import shutil
import uuid
from PIL import  Image
from captcha.image import ImageCaptcha

import itertools
import  random



def get_choices():
    digits='0123456789'
    return digits


def gen_captcha(img_dir, num_per_image, number,choices):
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir,ignore_errors=True)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    count=0
    random_widths = list(range(64, 100, 1))
    for _ in range(number):
        for i in itertools.permutations(choices, num_per_image):
            if count>=number:
                break
            else:
                captcha = ''.join(i)
                fn = os.path.join(img_dir, '%s_%s.jpg' % (captcha,count))
                ima = ImageCaptcha(width=random_widths[int(random.random()*36)], height=32, font_sizes=(26, 28, 30))
                ima = ima.create_captcha_image(chars=captcha, color='white', background='black')
                ima.save(fn)
                count += 1
gen_captcha(img_dir='while_black/',num_per_image=4,number=16000,choices=get_choices())