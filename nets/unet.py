import tensorflow as tf
from .unet_blocks import *

normalization_types = ['batch', None]
up_scaling_types = ['upConv', 'deConv']

def unet(inputs,
         training=None,
         filters=64,
         stages=4,
         convs=2,
         dropout=0,
         normalization='batch',
         up_scaling='deConv', # 'upConv', 'deConv'
         weight_decay=1e-5):

    assert normalization in normalization_types
    assert up_scaling in up_scaling_types
    # encoding path
    x_enc = []
    for idx in range(stages):
        if idx == 0:
            x, p = block_encoder(inputs, training, filters*(2**idx), convs=convs, normalization=normalization, dropout=0, weight_decay=weight_decay)
        else:
            x, p = block_encoder(p, training, filters*(2**idx), convs=convs, normalization=normalization, dropout=dropout, weight_decay=weight_decay)
        x_enc.append(x)
    # bottleneck
    x = block_bottleneck(p, training, filters*(2**stages), convs=convs, normalization=normalization, dropout=dropout, weight_decay=weight_decay)
    # decoding
    for idx in reversed(range(stages)):
        x = block_decoder(x, training, x_enc[idx], filters*(2**idx), convs=convs, normalization=normalization, dropout=dropout, up_scaling=up_scaling, weight_decay=weight_decay)
    
    return x
    


if __name__ == "__main__":
    input_img = tf.keras.layers.Input((256,256,3), name='input_img')
    fts = unet(input_img, filters=64)
    
    model = tf.keras.Model(inputs=input_img,
                         outputs=fts)

    model.summary()