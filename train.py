from tensorflow.keras import datasets, layers, models

import numpy as np # linear algebra
import os
import glob
import tensorflow as tf
from tensorflow import keras
import  time

from crnnvalidcode.utils import params, char_dict, decode_to_text, data_generator, sparse_tuple_from
from crnnvalidcode.metrics import WordAccuracy
from crnnvalidcode.model import build_model
from  crnnvalidcode.losses import  CTCLoss
from crnnvalidcode.dataset import   DatasetBuilder
DATA_DIR = './while_black'
batch_size=32
def parse_filepath(filepath):
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        label, _ = filename.split("_")
        return label
    except Exception as e:
        print('error to parse %s. %s' % (filepath, e))
        return None, None


# create a pandas data frame of images, age, gender and race
files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
labels = list(map(parse_filepath, files))

sample_len=len(files)
train_files=files[0:int(sample_len*0.9)]
train_labels=labels[0:int(sample_len*0.9)]

val_files=files[int(sample_len*0.9):]
val_labels=labels[int(sample_len*0.9):]



def preprocess(x,y):
    # print(file)
    img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (32, 100))
    return img, y

dataset_builder = DatasetBuilder('./table_path.txt', 100,1, ignore_case=False)
train_ds, train_size = dataset_builder.build(train_files,train_labels, True,batch_size)
val_ds, val_size = dataset_builder.build(val_files,val_labels, True,batch_size)

'''
print('train count: %s, valid count: %s'% (len(train_files), len(val_files)))
train_gen=tf.data.Dataset.from_tensor_slices((train_files,train_label))
train_gen=train_gen.shuffle(10000).map(preprocess).batch(batch_size)
print(train_gen)

val_gen=tf.data.Dataset.from_tensor_slices((val_files,val_label))
val_gen=val_gen.shuffle(10000).map(preprocess).batch(batch_size)
'''

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=5)

Epochs=2000
model = build_model(11, channels=1)
model.summary()
model.compile(optimizer=keras.optimizers.Adam(0.0001),
              loss=CTCLoss(), metrics=[WordAccuracy()])

localtime = time.strftime("%Y%m%d%H%M%S", time.localtime())
checkpoint_path="savecode/{}".format(localtime)
os.makedirs(checkpoint_path)
checkpoint_path=checkpoint_path+"/cp-{epoch:04d}.ckpt"
# 使用 `checkpoint_path` 格式保存权重
model.save_weights(checkpoint_path.format(epoch=0))

callbacks = [keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True),
             keras.callbacks.TensorBoard(log_dir='logs/{}'.format(localtime))]
model.fit(train_ds, epochs=Epochs, callbacks=callbacks,validation_data=val_ds)