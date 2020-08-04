import tensorflow as tf


def extract_fn_base(data_record, image_depth='uint8'):
    feature_dict = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/channels': tf.io.FixedLenFeature([], tf.int64),
        'image/image': tf.io.FixedLenFeature([], tf.string),
    }
    
    sample = tf.io.parse_single_example(data_record, feature_dict)
    if image_depth == 'uint8':
        sample['image/image'] = tf.image.decode_png(sample['image/image'], dtype=tf.uint8)    
    else:
        sample['image/image'] = tf.image.decode_png(sample['image/image'], dtype=tf.uint16)
    
    return sample



