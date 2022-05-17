
import glob
import cv2
from utils_tfrecord import *
import tensorflow as tf
import matplotlib.pyplot as plt

files = sorted(glob.glob('./ds_neurofinder/neurofinder.00.00/images/*.tiff'))[0:50]

with ImageTFrecordWriter('./tfrecords', 'neuron', 5) as tfw:
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img = img[0:256, 0:256]
        tfw.add(image_example(img))

ds = tf.data.TFRecordDataset(glob.glob('./tfrecords/*'))
ds = ds.map(lambda x: extract_fn_base(x, image_depth='uint16'))

for example in ds.take(1):

    plt.imshow(np.squeeze(example['image/image']))
    plt.show()