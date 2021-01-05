import os
import re

import tensorflow as tf


class DatasetBuilder():
    def __init__(self, table_path, img_width, img_channels, ignore_case=False):
        self.table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
            table_path, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE, 
            tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER), -1)
        self.img_width = img_width
        self.img_channels = img_channels
        self.ignore_case = ignore_case
        self.num_classes = self.table.size()

    def decode_and_resize(self, filename, label):
        img = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img, channels=self.img_channels)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (32, self.img_width))
        return img, label

    def tokenize(self, imgs, labels):
        chars = tf.strings.unicode_split(labels, 'UTF-8')
        tokens = tf.ragged.map_flat_values(self.table.lookup, chars)
        tokens = tokens.to_sparse()
        return imgs, tokens
        
    def build(self, img_paths,labels, shuffle, batch_size):
        """
        build dataset, it will auto detect each annotation file's format.
        """
        if self.ignore_case:
            labels = [label.lower() for label in labels]
        size = len(img_paths)
        ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.map(self.decode_and_resize, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Ignore the errors e.g. decode error or invalid data.
        ds = ds.apply(tf.data.experimental.ignore_errors())
        ds = ds.batch(batch_size)
        ds = ds.map(self.tokenize, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds, size


class Decoder:
    def __init__(self, table, blank_index=-1, merge_repeated=True):
        """
        
        Args:
            table: list, char map
            blank_index: int(default: num_classes - 1), the index of blank 
        label.
            merge_repeated: bool
        """
        self.table = table
        if blank_index == -1:
            blank_index = len(table) - 1
        self.blank_index = blank_index
        self.merge_repeated = merge_repeated

    def map2string(self, inputs):
        strings = []
        for i in inputs:
            text = [self.table[char_index] for char_index in i 
                    if char_index != self.blank_index]
            strings.append(''.join(text))
        return strings

    def decode(self, inputs, from_pred=True, method='greedy'):
        #inputs=(3,24,11)
        if from_pred:
            #[24,24,24]=[3]*24=[24,24,24]
            logit_length = tf.fill([tf.shape(inputs)[0]], tf.shape(inputs)[1])
            if method == 'greedy':
                decoded, _ = tf.nn.ctc_greedy_decoder(
                    inputs=tf.transpose(inputs, perm=[1, 0, 2]),
                    sequence_length=logit_length,
                    merge_repeated=self.merge_repeated)
            elif method == 'beam_search':
                decoded, _ = tf.nn.ctc_beam_search_decoder(
                    inputs=tf.transpose(inputs, perm=[1, 0, 2]),
                    sequence_length=logit_length)
            inputs = decoded[0]
        #decoded
        #[
        # [25,10,18,23,29,11,27,30,28,17,14,28],
        # [27,14,18,22,11,30,27,28,18,23,16,36],
        # [12,27,14,10,29,18,24,23,18,28,22,28]
        #]
        decoded = tf.sparse.to_dense(inputs, 
                                     default_value=self.blank_index).numpy()
        decoded = self.map2string(decoded)
        return decoded