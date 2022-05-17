import tensorflow as tf
import numpy as np
from nets import unet

tf.config.set_visible_devices([], 'GPU')
sz = 448

inputs = tf.keras.Input((sz, sz, 1))
outputs = unet(inputs,
                filters=16,
                stages=2,
                convs=2,
                dropout=0,
                normalization='batch',
                up_scaling='deConv', # 'upConv', 'deConv'
                weight_decay=1e-5)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

rp_r = np.zeros((sz, sz))
rp_c = np.zeros((sz, sz))
img = np.zeros((1, sz, sz, 1))

C = sz // 2
base = np.squeeze(model(img))[C, C, :]

# for L in model.layers:
#     w = L.get_weights()
#     print(w)

for i in range(sz):
    print('scanning Row: ', i)
    img_c = np.copy(img)
    img_c[0,i,:,0] = 255
    V = np.squeeze(model(img_c))[C, C, :]
    if not np.all(V == base):
        rp_r[i, :] = 1

for i in range(sz):
    print('scanning Col: ', i)
    img_c = np.copy(img)
    img_c[0,:,i,0] = 255
    V = np.squeeze(model(img_c))[C, C, :]
    if not np.all(V == base):
        rp_c[:, i] = 1

rp = rp_r * rp_c

print(np.sum(rp) ** 0.5)
