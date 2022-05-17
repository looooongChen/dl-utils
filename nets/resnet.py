from pyexpat import model
import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.regularizers import l2
import tempfile
import os

# def resnet(inputs,
#            version='ResNet50',
#            filters=64,
#            weight_decay=1e-5):

#     x = Conv2D(3, 1, activation='relu', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(inputs)

#     if version.lower() == 'resnet50':
#         net = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
#     if version.lower() == 'resnet101':
#         net = tf.keras.applications.ResNet101V2(include_top=False, weights='imagenet')
#     if version.lower() == 'resnet152':
#         net = tf.keras.applications.ResNet152V2(include_top=False, weights='imagenet')
#     if weight_decay > 0:
#         for layer in net.layers:
#             if isinstance(layer, Conv2D):
#                 layer.kernel_regularizer = l2(weight_decay)
#                 layer.bias_regularizer=l2(weight_decay)

#         tmp_weights_path = os.path.join(tempfile.gettempdir(), 'resnet_tmp_weights.h5')
#         net.save_weights(tmp_weights_path)
#         net = tf.keras.models.model_from_json(net.to_json())        
#         net.load_weights(tmp_weights_path, by_name=True)
    
#     net = tf.keras.Model(inputs=net.inputs,
#                          outputs=[net.get_layer('conv1_conv').output, # stride 1/2
#                                   net.get_layer('conv2_block2_out').output, # stride 1/4
#                                   net.get_layer('conv3_block3_out').output, # stride 1/8
#                                   net.get_layer('conv4_block5_out').output, # stride 1/16
#                                   net.get_layer('conv5_block3_out').output]) # stride 1/32
    
#     ft2, ft4, ft8, ft16, ft32 = net(x)

#     ft16u = Conv2DTranspose(filters, 2, 2, padding='same', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(ft32)
#     ft16 = tf.concat([ft16u, ft16], axis=-1)
#     ft8u = Conv2DTranspose(filters, 2, 2, padding='same', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(ft16)
#     ft8 = tf.concat([ft8u, ft8], axis=-1)
#     ft4u = Conv2DTranspose(filters, 2, 2, padding='same', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(ft8)
#     ft4 = tf.concat([ft4u, ft4], axis=-1)
#     ft2u = Conv2DTranspose(filters, 2, 2, padding='same', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(ft4)
#     ft2 = tf.concat([ft2u, ft2], axis=-1)
#     ft0 = Conv2DTranspose(filters, 2, 2, padding='same', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(ft2)
#     ft0 = tf.concat([ft0, x], axis=-1)

#     ft0 = Conv2D(filters, 3, padding='same', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(ft0)

#     return ft0

def resnet(inputs,
           version='ResNet50',
           filters=64):

    x = Conv2D(3, 1, activation='relu')(inputs)

    if version.lower() == 'resnet50':
        net = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
    if version.lower() == 'resnet101':
        net = tf.keras.applications.ResNet101V2(include_top=False, weights='imagenet')
    if version.lower() == 'resnet152':
        net = tf.keras.applications.ResNet152V2(include_top=False, weights='imagenet')

    # if weight_decay > 0:
    #     for layer in net.layers:
    #         if isinstance(layer, Conv2D):
    #             layer.kernel_regularizer = l2(weight_decay)
    #             layer.bias_regularizer=l2(weight_decay)

    #     tmp_weights_path = os.path.join(tempfile.gettempdir(), 'resnet_tmp_weights.h5')
    #     net.save_weights(tmp_weights_path)
    #     net = tf.keras.models.model_from_json(net.to_json())        
    #     net.load_weights(tmp_weights_path, by_name=True)
    
    net = tf.keras.Model(inputs=net.inputs,
                         outputs=[net.get_layer('conv1_conv').output, # stride 1/2
                                  net.get_layer('conv2_block2_out').output, # stride 1/4
                                  net.get_layer('conv3_block3_out').output, # stride 1/8
                                  net.get_layer('conv4_block5_out').output, # stride 1/16
                                  net.get_layer('conv5_block3_out').output]) # stride 1/32
    
    ft2, ft4, ft8, ft16, ft32 = net(x, training=False)

    ft16u = Conv2DTranspose(filters, 2, 2, padding='same')(ft32)
    ft16 = tf.concat([ft16u, ft16], axis=-1)
    ft8u = Conv2DTranspose(filters, 2, 2, padding='same')(ft16)
    ft8 = tf.concat([ft8u, ft8], axis=-1)
    ft4u = Conv2DTranspose(filters, 2, 2, padding='same')(ft8)
    ft4 = tf.concat([ft4u, ft4], axis=-1)
    ft2u = Conv2DTranspose(filters, 2, 2, padding='same')(ft4)
    ft2 = tf.concat([ft2u, ft2], axis=-1)
    ft0 = Conv2DTranspose(filters, 2, 2, padding='same')(ft2)
    ft0 = tf.concat([ft0, x], axis=-1)

    ft0 = Conv2D(filters, 3, padding='same')(ft0)

    return net, ft0

if __name__ == "__main__":
    # input_img = tf.keras.layers.Input((256,256,3), name='input_img')
    # net, fts = resnet(input_img, version='ResNet50', filters=64)
    
    # model = tf.keras.Model(inputs=input_img,
    #                      outputs=fts)
    # net.trainable = False

    # model.save('./test.h5')

    model = tf.keras.models.load_model('./test.h5')


    model.summary()




