from tensorflow.keras.layers import ConvLSTM2D, Conv2D, MaxPooling2D, MaxPooling3D, BatchNormalization, Dropout, Concatenate, UpSampling2D, Conv2DTranspose
from tensorflow.keras.regularizers import l2
'''
-- Long Chen, LfB, RWTH Aachen University --
tensroflow 2.x model: UNet
U-Net: Convolutional Networks for Biomedical Image Segmentation
'''

def block_conv2D(x, training, filters, convs=2, kernel_size=3, normalization='batch', weight_decay=1e-5):
    for _ in range(convs):
        # conv layers
        x = Conv2D(filters, 3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
        # batch normalization
        if normalization == 'batch':
            x = BatchNormalization()(x, training)
    return x


def block_encoder2D(x, training, filters, convs=2, normalization='batch', dropout=0, weight_decay=1e-5, ndim=4):
    # dropout layer before conv
    if dropout > 0:
        x = Dropout(dropout)(x, training)
    # conv layers
    x = block_conv2D(x=x, training=training, filters=filters, convs=convs, kernel_size=3, normalization=normalization, weight_decay=weight_decay)
    # pooling layer
    if ndim == 5:
        p = MaxPooling3D(pool_size=(1,2,2))(x)
    else:
        p = MaxPooling2D(pool_size=(2,2))(x)
    return x, p

block_encoder = block_encoder2D


def block_bottleneck2D(x, training, filters, convs=2, normalization='batch', dropout=0, weight_decay=1e-5):
    # dropout layer before conv
    if dropout > 0:
        x = Dropout(dropout)(x, training)
    # conv layers
    x = block_conv2D(x=x, training=training, filters=filters, convs=convs, kernel_size=3, normalization=normalization, weight_decay=weight_decay)
    return x

block_bottleneck = block_bottleneck2D


def block_decoder2D(x, training, x_enc, filters, convs=2, normalization='batch', dropout=0, up_scaling='deConv', weight_decay=1e-5):
    # up layer
    if up_scaling == 'upConv':
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = Conv2D(filters, 3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
    else:
        x = Conv2DTranspose(filters, 2, 2, padding='same', activation='relu', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
    x = Concatenate(axis=-1)([x, x_enc])
    # dropout layer before conv
    if dropout > 0:
        x = Dropout(dropout)(x, training)
    x = block_conv2D(x=x, training=training, filters=filters, convs=convs, normalization=normalization, weight_decay=weight_decay)

    return x

block_decoder = block_decoder2D


# def block_lstmConv2D(x, filters, convs=2, kernel_size=3, return_sequences=True, stateful=False, normalization='batch', weight_decay=1e-5):
#     for idx in range(convs):
#         # conv layers
#         if idx == convs-1:
#             x = ConvLSTM2D(filters, kernel_size, padding='same', stateful=stateful, return_sequences=return_sequences, activation="relu", kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
#         else:
#             x = ConvLSTM2D(filters, kernel_size, padding='same', stateful=stateful, return_sequences=True, activation="relu", kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
#         # batch normalization
#         if normalization == 'batch':
#             x = BatchNormalization()(x)
#     return x

# def block_lstmEncoder(x, filters, convs=2, stateful=False, normalization='batch', dropout=0, weight_decay=1e-5):
#     # dropout layer before conv
#     if dropout > 0:
#         x = Dropout(dropout)(x)
#     # conv layers
#     x = block_lstmConv2D(x=x, filters=filters, convs=convs, kernel_size=3, return_sequences=True, stateful=stateful, normalization=normalization, weight_decay=weight_decay)
#     # pooling layer
#     p = MaxPooling3D(pool_size=(1,2,2))(x)
#     return x, p


# def block_lstmBottleneck(x, filters, convs=2, stateful=False, normalization='batch', dropout=0, weight_decay=1e-5):
#     # dropout layer before conv
#     if dropout > 0:
#         x = Dropout(dropout)(x)
#     # conv layers
#     x = block_lstmConv2D(x=x, filters=filters, convs=convs, kernel_size=3, return_sequences=True, stateful=stateful, normalization=normalization, weight_decay=weight_decay)
#     return x

# def block_lstmSkip(x, filters, convs=2, stateful=False, normalization='batch', dropout=0, weight_decay=1e-5):
#     # dropout layer before conv
#     if dropout > 0:
#         x = Dropout(dropout)(x)
#     # conv layers
#     x = block_lstmConv2D(x, filters=filters, convs=convs, kernel_size=1, return_sequences=False, stateful=stateful, normalization=normalization, weight_decay=weight_decay)
#     return x


# def block_lstmDecoder(x, x_enc, filters, convs=2, stateful=False, normalization='batch', dropout=0, up_scaling='deConv', weight_decay=1e-5):
#     # up layer
#     if up_scaling == 'upConv':
#         x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
#         x = Conv2D(filters, 3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
#     else:
#         x = Conv2DTranspose(filters, 2, 2, padding='same', activation='relu', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
#     x = Concatenate(axis=-1)([x, x_enc])
#     # dropout layer before conv
#     if dropout > 0:
#         x = Dropout(dropout)(x)
#     x = block_conv2D(x=x, filters=filters, convs=convs, normalization=normalization, weight_decay=weight_decay)

#     return x
