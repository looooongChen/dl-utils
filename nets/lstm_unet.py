import tensorflow as tf
from tensorflow.keras.layers import ConvLSTM2D
from .unet_blocks import *

def unet_lstmSkip(inputs,
                filters=64,
                stages=4,
                convs=2,
                skip_convs=2,
                stateful=False,
                dropout=0,
                normalization='batch',
                up_scaling='deConv', # 'upConv', 'deConv'
                weight_decay=1e-5):
    # encoding path
    x_enc = []
    for idx in range(stages):
        if idx == 0:
            x, p = block_encoder(inputs, filters*(2**idx), convs=convs, normalization=normalization, dropout=0, weight_decay=weight_decay, ndim=5)
        else:
            x, p = block_encoder(p, filters*(2**idx), convs=convs, normalization=normalization, dropout=dropout, weight_decay=weight_decay, ndim=5)
        x = block_lstmSkip(x, filters*(2**idx), convs=skip_convs, stateful=stateful, normalization=normalization, dropout=dropout, weight_decay=weight_decay)
        x_enc.append(x)
    # bottleneck
    x = block_bottleneck(p, filters*(2**stages), convs=convs, normalization=normalization, dropout=dropout, weight_decay=weight_decay)
    x = block_lstmSkip(x, filters*(2**idx), convs=skip_convs, stateful=stateful, normalization=normalization, dropout=dropout, weight_decay=weight_decay)
    # decoding
    for idx in reversed(range(stages)):
        x = block_decoder(x, x_enc[idx], filters*(2**idx), convs=convs, normalization=normalization, dropout=dropout, up_scaling=up_scaling, weight_decay=weight_decay)
    
    return x

def unet_lstmEnc(inputs,
         filters=64,
         stages=4,
         convs=2,
         stateful=False,
         dropout=0,
         normalization='batch',
         up_scaling='deConv', # 'upConv', 'deConv'
         weight_decay=1e-5):
    # encoding path
    x_enc = []
    for idx in range(stages):
        if idx == 0:
            x, p = block_lstmEncoder(inputs, filters*(2**idx), convs=convs, stateful=stateful, normalization=normalization, dropout=0, weight_decay=weight_decay)
        else:
            x, p = block_lstmEncoder(p, filters*(2**idx), convs=convs, stateful=stateful,normalization=normalization, dropout=dropout, weight_decay=weight_decay)
        x_enc.append(x[:,-1,:,:,:])
    # bottleneck
    x = block_lstmBottleneck(p, filters*(2**stages), convs=convs, stateful=stateful, normalization=normalization, dropout=dropout, weight_decay=weight_decay)[:,-1,:,:,:]
    # decoding
    for idx in reversed(range(stages)):
        x = block_decoder(x, x_enc[idx], filters*(2**idx), convs=convs, normalization=normalization, dropout=dropout, up_scaling=up_scaling, weight_decay=weight_decay)
    
    return x
    
