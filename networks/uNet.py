import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ReLU, MaxPooling2D, BatchNormalization, Dropout, Concatenate, UpSampling2D, Conv2DTranspose, Add
'''
-- Long Chen, LfB, RWTH Aachen University --
tensroflow 2.x model: UNet
U-Net: Convolutional Networks for Biomedical Image Segmentation
'''

INIT = 'he_normal' # 'glorot_uniform'
PAD = 'same' # 'valid'

class UNnet(tf.keras.Model):

    def __init__(self,
               D=4,
               filters=32,
               dropout_rate=0.2,
               batch_norm=False,
               upsample='interp', # 'interp', 'conv'
               merge='cat', # 'add', 'cat'
               name='UNet',
               **kwargs):

        super().__init__(name=name, **kwargs)
        assert upsample in ['interp', 'conv']
        assert merge in ['add', 'cat']
        self.D = D
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.upsample = upsample
        self.merge = merge

        self.filters = [filters*2**i for i in range(D)] + [filters*2**(D-i) for i in range(D+1)]
        self.L = {}

        for i in range(1, 2*self.D+2):
            # dropout layer
            if self.dropout_rate < 1 and i != 1:
                self.L['dropout{:d}'.format(i)] = Dropout(self.dropout_rate)
            # conv layers
            self.L['conv{:d}_1'.format(i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
            self.L['conv{:d}_2'.format(i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
            # relu activation
            self.L['relu{:d}_1'.format(i)] = ReLU()
            self.L['relu{:d}_2'.format(i)] = ReLU()
            # batch normalization
            if self.batch_norm:
                self.L['batchnorm{:d}_1'.format(i)] = BatchNormalization()
                self.L['batchnorm{:d}_2'.format(i)] = BatchNormalization()
        
        for i in range(1, self.D+1):
            # pooling
            self.L['pool{:d}'.format(i)] = MaxPooling2D(pool_size=(2, 2))

        for i in range(self.D+2, 2*self.D+2):
            # up sampling
            if self.upsample == 'interp':
                self.L['up{:d}'.format(i)] = UpSampling2D(size=(2, 2), interpolation='bilinear')
                self.L['conv{:d}_up'.format(i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
            else:
                self.L['up{:d}'.format(i)] = Conv2DTranspose(self.filters[i-1], 2, 2, kernel_initializer=INIT)
            self.L['relu{:d}_up'.format(i)] = ReLU()
            self.L['batchnorm{:d}_up'.format(i)] = BatchNormalization()
            # merge
            if self.merge == 'cat':
                self.L['merge{:d}'.format(i)] = Concatenate(axis=-1)
            else:
                self.L['merge{:d}'.format(i)] = Add()

    def call(self, inputs):

        self.T = {}

        outputs = inputs
        for i in range(1, self.D+2):
            # dropout
            if self.dropout_rate < 1 and i != 1:
                outputs = self.L['dropout{:d}'.format(i)](outputs)
            # conv1
            outputs = self.L['conv{:d}_1'.format(i)](outputs)
            if self.batch_norm:
                outputs = self.L['batchnorm{:d}_1'.format(i)](outputs)
            outputs = self.L['relu{:d}_1'.format(i)](outputs)
            # conv2
            outputs = self.L['conv{:d}_2'.format(i)](outputs)
            if self.batch_norm:
                outputs = self.L['batchnorm{:d}_2'.format(i)](outputs)
            outputs = self.L['relu{:d}_2'.format(i)](outputs)
            # pooling
            if i != self.D+1:
                self.T['conv{:d}'.format(i)] = outputs
                outputs = self.L['pool{:d}'.format(i)](outputs)

        for i in range(self.D+2, 2*self.D+2):
            # upsampling
            outputs = self.L['up{:d}'.format(i)](outputs)
            if self.upsample == 'interp':
                outputs = self.L['conv{:d}_up'.format(i)](outputs)
            if self.batch_norm:
                outputs = self.L['batchnorm{:d}_up'.format(i)](outputs)
            outputs = self.L['relu{:d}_up'.format(i)](outputs)
            # merge
            outputs = self.L['merge{:d}'.format(i)]([outputs, self.T['conv{:d}'.format(2*self.D+2-i)]])
            # dropout
            if self.dropout_rate < 1:
                outputs = self.L['dropout{:d}'.format(i)](outputs)
            # conv1
            outputs = self.L['conv{:d}_1'.format(i)](outputs)
            if self.batch_norm:
                outputs = self.L['batchnorm{:d}_1'.format(i)](outputs)
            outputs = self.L['relu{:d}_1'.format(i)](outputs)
            # conv2
            outputs = self.L['conv{:d}_2'.format(i)](outputs)
            if self.batch_norm:
                outputs = self.L['batchnorm{:d}_2'.format(i)](outputs)
            outputs = self.L['relu{:d}_2'.format(i)](outputs)
        
        return outputs


# unet with spatial attention
class UNnetSA(tf.keras.Model):

    def __init__(self,
                 D=4,
                 filters=32,
                 dropout_rate=0.2,
                 batch_norm=False,
                 upsample='interp', # 'interp', 'conv'
                 merge='cat', # 'add', 'cat'
                 name='UNet',
                 **kwargs):

        super().__init__(name=name, **kwargs)
        assert upsample in ['interp', 'conv']
        assert merge in ['add', 'cat']
        self.D = D
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.upsample = upsample
        self.merge = merge

        self.filters = [filters*2**i for i in range(D)] + [filters*2**(D-i) for i in range(D+1)]
        self.L = {}

        for i in range(1, 2*self.D+2):
            # dropout layer
            if self.dropout_rate < 1 and i != 1:
                self.L['dropout{:d}'.format(i)] = Dropout(self.dropout_rate)
            # conv layers
            self.L['conv{:d}_1'.format(i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
            self.L['conv{:d}_2'.format(i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
            # relu activation
            self.L['relu{:d}_1'.format(i)] = ReLU()
            self.L['relu{:d}_2'.format(i)] = ReLU()
            # batch normalization
            if self.batch_norm:
                self.L['batchnorm{:d}_1'.format(i)] = BatchNormalization()
                self.L['batchnorm{:d}_2'.format(i)] = BatchNormalization()
        
        for i in range(1, self.D+1):
            # pooling
            self.L['pool{:d}'.format(i)] = MaxPooling2D(pool_size=(2, 2))

        for i in range(self.D+2, 2*self.D+2):
            # up sampling
            if self.upsample == 'interp':
                self.L['up{:d}'.format(i)] = UpSampling2D(size=(2, 2), interpolation='bilinear')
                self.L['conv{:d}_up'.format(i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
            else:
                self.L['up{:d}'.format(i)] = Conv2DTranspose(self.filters[i-1], 2, 2, kernel_initializer=INIT)
            self.L['relu{:d}_up'.format(i)] = ReLU()
            self.L['batchnorm{:d}_up'.format(i)] = BatchNormalization()
            # merge
            self.L['atten{:d}_enc_conv'.format(i)] = Conv2D(self.filters[i-1]//4, 1, padding=PAD, kernel_initializer=INIT)
            self.L['atten{:d}_dec_conv'.format(i)] = Conv2D(self.filters[i-1]//4, 1, padding=PAD, kernel_initializer=INIT)
            self.L['atten{:d}_relu'.format(i)] = ReLU()
            self.L['atten{:d}_conv'.format(i)] = Conv2D(1, 1, padding=PAD, activation='sigmoid', kernel_initializer=INIT)
            if self.merge == 'cat':
                self.L['merge{:d}'.format(i)] = Concatenate(axis=-1)
            else:
                self.L['merge{:d}'.format(i)] = Add()

    def call(self, inputs):

        self.T = {}

        outputs = inputs
        for i in range(1, self.D+2):
            # dropout
            if self.dropout_rate < 1 and i != 1:
                outputs = self.L['dropout{:d}'.format(i)](outputs)
            # conv1
            outputs = self.L['conv{:d}_1'.format(i)](outputs)
            if self.batch_norm:
                outputs = self.L['batchnorm{:d}_1'.format(i)](outputs)
            outputs = self.L['relu{:d}_1'.format(i)](outputs)
            # conv2
            outputs = self.L['conv{:d}_2'.format(i)](outputs)
            if self.batch_norm:
                outputs = self.L['batchnorm{:d}_2'.format(i)](outputs)
            outputs = self.L['relu{:d}_2'.format(i)](outputs)
            # pooling
            if i != self.D+1:
                self.T['conv{:d}'.format(i)] = outputs
                outputs = self.L['pool{:d}'.format(i)](outputs)

        for i in range(self.D+2, 2*self.D+2):
            # upsampling
            outputs = self.L['up{:d}'.format(i)](outputs)
            if self.upsample == 'interp':
                outputs = self.L['conv{:d}_up'.format(i)](outputs)
            if self.batch_norm:
                outputs = self.L['batchnorm{:d}_up'.format(i)](outputs)
            outputs = self.L['relu{:d}_up'.format(i)](outputs)
            # attention and merge
            enc = self.T['conv{:d}'.format(2*self.D+2-i)]
            atten_enc = self.L['atten{:d}_enc_conv'.format(i)](enc)
            atten_dec = self.L['atten{:d}_dec_conv'.format(i)](outputs)
            atten = self.L['atten{:d}_relu'.format(i)](atten_enc+atten_dec)
            atten = self.L['atten{:d}_conv'.format(i)](atten)
            outputs = self.L['merge{:d}'.format(i)]([outputs, enc*atten])
            # dropout
            if self.dropout_rate < 1:
                outputs = self.L['dropout{:d}'.format(i)](outputs)
            # conv1
            outputs = self.L['conv{:d}_1'.format(i)](outputs)
            if self.batch_norm:
                outputs = self.L['batchnorm{:d}_1'.format(i)](outputs)
            outputs = self.L['relu{:d}_1'.format(i)](outputs)
            # conv2
            outputs = self.L['conv{:d}_2'.format(i)](outputs)
            if self.batch_norm:
                outputs = self.L['batchnorm{:d}_2'.format(i)](outputs)
            outputs = self.L['relu{:d}_2'.format(i)](outputs)
        
        return outputs

class UNnetDilate(UNnet):
    
    def __init__(self,
                 D=4,
                 filters=32,
                 dropout_rate=0.2,
                 batch_norm=False,
                 upsample='interp', # 'interp', 'conv'
                 merge='cat', # 'add', 'cat'
                 name='UNetDilate',
                 dilation_rate=4,
                 **kwargs):

        super().__init__(D=D,
                         filters=filters,
                         dropout_rate=dropout_rate,
                         batch_norm=batch_norm,
                         upsample=upsample, # 'interp', 'conv'
                         merge=merge, # 'add', 'cat'
                         name=name)
        self.dilation_rate = dilation_rate
        
        for i in range(dilation_rate):
            self.L['conv_dilation{:d}'.format(i)] = Conv2D(filters//dilation_rate, 3, 1, dilation_rate=2**(i+1), padding=PAD, kernel_initializer=INIT)
        self.concat_dilation = Concatenate(axis=-1)
        self.conv_dilate = Conv2D(filters, 3, 1, padding=PAD, kernel_initializer=INIT)

    def call(self, inputs):
        outs = super().call(inputs)
        dilated = [outs]
        for i in range(self.dilation_rate):
            dilated.append(self.L['conv_dilation{:d}'.format(i)](outs))
        outs = self.concat_dilation(dilated)
        outs = self.conv_dilate(outs)
        return outs

# class UNnetAtten(tf.keras.Model):

#     def __init__(self,
#                  D=4,
#                  filters=32,
#                  dropout_rate=0.2,
#                  batch_norm=False,
#                  upsample='interp', # 'interp', 'conv'
#                  merge='cat', # 'add', 'cat'
#                  name='UNet',
#                  **kwargs):

#         super().__init__(name=name, **kwargs)
#         assert upsample in ['interp', 'conv']
#         assert merge in ['add', 'cat']
#         self.D = D
#         self.dropout_rate = dropout_rate
#         self.batch_norm = batch_norm
#         self.upsample = upsample
#         self.merge = merge

#         self.filters = [filters*2**i for i in range(D)] + [filters*2**(D-i) for i in range(D+1)]
#         self.L = {}

#         for i in range(1, 2*self.D+2):
#             # dropout layer
#             if self.dropout_rate < 1 and i != 1:
#                 self.L['dropout{:d}'.format(i)] = Dropout(self.dropout_rate)
#             # conv layers
#             self.L['conv{:d}_1'.format(i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
#             self.L['conv{:d}_2'.format(i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
#             # relu activation
#             self.L['relu{:d}_1'.format(i)] = ReLU()
#             self.L['relu{:d}_2'.format(i)] = ReLU()
#             # batch normalization
#             if self.batch_norm:
#                 self.L['batchnorm{:d}_1'.format(i)] = BatchNormalization()
#                 self.L['batchnorm{:d}_2'.format(i)] = BatchNormalization()
        
#         for i in range(1, self.D+1):
#             # pooling
#             self.L['pool{:d}'.format(i)] = MaxPooling2D(pool_size=(2, 2))

#         for i in range(self.D+2, 2*self.D+2):
#             # up sampling
#             if self.upsample == 'interp':
#                 self.L['up{:d}'.format(i)] = UpSampling2D(size=(2, 2), interpolation='bilinear')
#                 self.L['conv{:d}_up'.format(i)] = Conv2D(self.filters[i-1], 3, padding=PAD, kernel_initializer=INIT)
#             else:
#                 self.L['up{:d}'.format(i)] = Conv2DTranspose(self.filters[i-1], 2, 2, kernel_initializer=INIT)
#             self.L['relu{:d}_up'.format(i)] = ReLU()
#             self.L['batchnorm{:d}_up'.format(i)] = BatchNormalization()
#             # merge
#             self.L['spatial_atten{:d}_enc_conv1'.format(i)] = Conv2D(1, 1, padding=PAD, kernel_initializer=INIT)
#             self.L['spatial_atten{:d}_enc_conv2'.format(i)] = Conv2D(1, 7, padding=PAD, kernel_initializer=INIT)
#             self.L['spatial_atten{:d}_enc_concat'.format(i)] = Concatenate(axis=-1)
#             self.L['spatial_atten{:d}_dec_conv1'.format(i)] = Conv2D(1, 1, padding=PAD, kernel_initializer=INIT)
#             self.L['spatial_atten{:d}_dec_conv2'.format(i)] = Conv2D(1, 7, padding=PAD, kernel_initializer=INIT)
#             self.L['spatial_atten{:d}_dec_concat'.format(i)] = Concatenate(axis=-1)
#             self.L['channel_atten{:d}_enc_conv1'.format(i)] = Conv2D(self.filters[i-1]//16, 1, padding=PAD, kernel_initializer=INIT)
#             self.L['channel_atten{:d}_enc_conv2'.format(i)] = Conv2D(self.filters[i-1]//16, 1, padding=PAD, kernel_initializer=INIT)
#             self.L['channel_atten{:d}_dec_conv1'.format(i)] = Conv2D(self.filters[i-1]//16, 1, padding=PAD, kernel_initializer=INIT)
#             self.L['channel_atten{:d}_dec_conv2'.format(i)] = Conv2D(self.filters[i-1]//16, 1, padding=PAD, kernel_initializer=INIT)
#             self.L['channel_atten{:d}_conv'.format(i)] = Conv2D(self.filters[i-1], 1, padding=PAD, kernel_initializer=INIT)
#             if self.merge == 'cat':
#                 self.L['merge{:d}'.format(i)] = Concatenate(axis=-1)
#             else:
#                 self.L['merge{:d}'.format(i)] = Add()

#     def call(self, inputs):

#         self.T = {}

#         outputs = inputs
#         for i in range(1, self.D+2):
#             # dropout
#             if self.dropout_rate < 1 and i != 1:
#                 outputs = self.L['dropout{:d}'.format(i)](outputs)
#             # conv1
#             outputs = self.L['conv{:d}_1'.format(i)](outputs)
#             if self.batch_norm:
#                 outputs = self.L['batchnorm{:d}_1'.format(i)](outputs)
#             outputs = self.L['relu{:d}_1'.format(i)](outputs)
#             # conv2
#             outputs = self.L['conv{:d}_2'.format(i)](outputs)
#             if self.batch_norm:
#                 outputs = self.L['batchnorm{:d}_2'.format(i)](outputs)
#             outputs = self.L['relu{:d}_2'.format(i)](outputs)
#             # pooling
#             if i != self.D+1:
#                 self.T['conv{:d}'.format(i)] = outputs
#                 outputs = self.L['pool{:d}'.format(i)](outputs)

#         for i in range(self.D+2, 2*self.D+2):
#             # upsampling
#             outputs = self.L['up{:d}'.format(i)](outputs)
#             if self.upsample == 'interp':
#                 outputs = self.L['conv{:d}_up'.format(i)](outputs)
#             if self.batch_norm:
#                 outputs = self.L['batchnorm{:d}_up'.format(i)](outputs)
#             outputs = self.L['relu{:d}_up'.format(i)](outputs)
#             # merge
#             # spatial attention
#             enc = self.T['conv{:d}'.format(2*self.D+2-i)]
#             spatial_atten_enc_conv = self.L['spatial_atten{:d}_enc_conv1'.format(i)](enc)
#             spatial_atten_enc_max = tf.reduce_max(enc, axis=-1, keepdims=True)
#             spatial_atten_enc_avg = tf.reduce_mean(enc, axis=-1, keepdims=True)
#             spatial_atten_enc = self.L['spatial_atten{:d}_enc_concat'.format(i)]([spatial_atten_enc_conv, spatial_atten_enc_max, spatial_atten_enc_avg])
#             spatial_atten_enc = self.L['spatial_atten{:d}_enc_conv2'.format(i)](spatial_atten_enc)

#             spatial_atten_dec_conv = self.L['spatial_atten{:d}_dec_conv1'.format(i)](outputs)
#             spatial_atten_dec_max = tf.reduce_max(outputs, axis=-1, keepdims=True)
#             spatial_atten_dec_avg = tf.reduce_mean(outputs, axis=-1, keepdims=True)
#             spatial_atten_dec = self.L['spatial_atten{:d}_dec_concat'.format(i)]([spatial_atten_dec_conv, spatial_atten_dec_max, spatial_atten_dec_avg])
#             spatial_atten_dec = self.L['spatial_atten{:d}_dec_conv2'.format(i)](spatial_atten_dec)

#             spatial_atten = spatial_atten_enc + spatial_atten_dec
#             spatial_atten = tf.math.sigmoid(spatial_atten)

#             channel_atten_enc_max = tf.reduce_max(enc, axis=[1,2], keepdims=True)
#             channel_atten_enc_max = self.L['channel_atten{:d}_enc_conv1'.format(i)](channel_atten_enc_max)
#             channel_atten_enc_avg = tf.reduce_mean(enc, axis=[1,2], keepdims=True)
#             channel_atten_enc_avg = self.L['channel_atten{:d}_enc_conv2'.format(i)](channel_atten_enc_avg)

#             channel_atten_dec_max = tf.reduce_max(outputs, axis=[1,2], keepdims=True)
#             channel_atten_dec_max = self.L['channel_atten{:d}_dec_conv1'.format(i)](channel_atten_dec_max)
#             channel_atten_dec_avg = tf.reduce_mean(outputs, axis=[1,2], keepdims=True)
#             channel_atten_dec_avg = self.L['channel_atten{:d}_dec_conv2'.format(i)](channel_atten_dec_avg)

#             channel_atten = self.L['channel_atten{:d}_conv'.format(i)](channel_atten_dec_max+channel_atten_dec_avg+channel_atten_enc_max+channel_atten_enc_avg)
#             channel_atten = tf.math.sigmoid(channel_atten)
#             # channel attenton
#             enc = enc * spatial_atten
#             enc = enc * channel_atten
            
#             outputs = self.L['merge{:d}'.format(i)]([outputs, enc])
#             # dropout
#             if self.dropout_rate < 1:
#                 outputs = self.L['dropout{:d}'.format(i)](outputs)
#             # conv1
#             outputs = self.L['conv{:d}_1'.format(i)](outputs)
#             if self.batch_norm:
#                 outputs = self.L['batchnorm{:d}_1'.format(i)](outputs)
#             outputs = self.L['relu{:d}_1'.format(i)](outputs)
#             # conv2
#             outputs = self.L['conv{:d}_2'.format(i)](outputs)
#             if self.batch_norm:
#                 outputs = self.L['batchnorm{:d}_2'.format(i)](outputs)
#             outputs = self.L['relu{:d}_2'.format(i)](outputs)
        
#         return outputs




if __name__ == "__main__":
    import numpy as np
    import os

    # model = UNnet(D=3, filters=32, dropout_rate=0.5, batch_norm=True, upsample='interp', merge='cat')
    model = UNnetSA(D=3, filters=32, dropout_rate=0.5, batch_norm=True, upsample='interp', merge='cat')
    # model.build(input_shape=(1,512,512,1))
    # model.summary()

    @tf.function
    def trace_func():
        inputs = np.zeros((4,512,512,1)).astype(np.float32)
        return model(inputs)

    # Set up logging.
    logdir = '.\\logs_check'
    writer = tf.summary.create_file_writer(logdir)

    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export().
    tf.summary.trace_on(graph=True, profiler=True)
    # Call only one tf.function when tracing.
    z = trace_func()
    with writer.as_default():
        tf.summary.trace_export(name="network_check", step=0, profiler_outdir=logdir)
