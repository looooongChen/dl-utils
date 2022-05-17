import tensorflow as tf
import os
from .tfrecord_type import *
import numpy as np
import cv2


def image_feature(img):

    _, img_encoded = cv2.imencode('.png', img)
    img_encoded = img_encoded.tobytes()

    channels = 1 if len(img.shape) == 2 else img.shape[2]

    feature_dict = {
        'image/height': int64_feature(img.shape[0]),
        'image/width': int64_feature(img.shape[1]),
        'image/channels': int64_feature(channels),
        'image/image': bytes_feature(img_encoded),
    }

    return feature_dict

def image_example(img):
    return tf.train.Example(features=tf.train.Features(feature=image_feature(img)))

def image_seg_example(img, seg):
    pass

def image_label_example(img, label):
    pass

class ImageTFrecordWriter(object):

    def __init__(self, base_dir, dataset_name, num_shards=5):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.num_shards = num_shards
        self.processed_count = 0
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
    
    def __enter__(self):
        tfrecord_files = [os.path.join(self.base_dir, '{}-{:03d}-of-{:03d}'.format(self.dataset_name, idx, self.num_shards)) for idx in range(self.num_shards)]
        self.tfrecords = [tf.io.TFRecordWriter(f) for f in tfrecord_files]
        return self

    def __exit__(self, *args):
        for writer in self.tfrecords:
            writer.close()

    def add(self, tf_example):
        self.processed_count += 1
        shard_idx = self.processed_count % self.num_shards
        self.tfrecords[shard_idx].write(tf_example.SerializeToString())

        if self.processed_count % 10 == 0:
            print('Processed images: {}'.format(self.processed_count))
