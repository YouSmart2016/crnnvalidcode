
import os
import tensorflow as tf

from model import build_model

from dataset import Decoder
import  random

import matplotlib.pyplot  as plt



test_path='./while_black/testimages/threecode'
def read_img_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (32, 100))
    return img
def init_show(numbers):
    n_cols = 10
    n_rows = numbers//n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 25))
    plt.subplots_adjust(hspace = 0.8)
    return axes
def  show_tests_result(index,axes,img,pred,real):
    ax = axes.flat[index]
    ax.imshow(img)
    color='blue' if pred==real else 'red'
    ax.set_title('pred: {}'.format(pred),color=color)
    ax.set_xlabel('real: {}'.format(real),color='blue')
    ax.set_xticks([])
    ax.set_yticks([])


if os.path.isdir(test_path):
    img_paths = os.listdir(test_path)
    img_paths = [os.path.join(test_path, path) for path in img_paths]
    random.shuffle(img_paths)
    img_paths=img_paths[0:100]
    imgs = list(map(read_img_and_preprocess, img_paths))
    imgs = tf.stack(imgs)
else:
    img_paths = [test_path]
    img = read_img_and_preprocess(test_path)
    imgs = tf.expand_dims(img, 0)
with open('table_path.txt', 'r') as f:
    inv_table = [char.strip() for char in f]

model = build_model(11, channels=1)
model.load_weights('savecode/2021102/cp-0018.ckpt')
decoder = Decoder(inv_table)
y_pred = model.predict(imgs)

test_result=[]
for path, g_pred, b_pred in zip(img_paths,decoder.decode(y_pred, method='greedy'),decoder.decode(y_pred, method='beam_search')):
    test_result.append((path, g_pred, b_pred))
    # print('Path: {}, greedy: {}, beam search: {}'.format(path, g_pred, b_pred))


axes=init_show(numbers=100)
for i,(path,g_pred,b_pred) in enumerate(test_result):
    filepath, filename = os.path.split(path)
    filename, ext = os.path.splitext(filename)
    label, _ = filename.split("_")
    img=plt.imread(path)
    show_tests_result(i,axes,img,g_pred,label)
plt.show()



