from uNet import UNet
from resNetSeg import ResNetSeg
import numpy as np
import cv2

sz = 256

model = UNet(nfilters=16, nstage=3, stage_conv=2, up_type='deConv', merge_type='cat',)
# model = ResNetSeg(input_shape=(sz,sz),version='ResNet50', filters=16, up_type='upConv', merge_type='cat')

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

# cv2.imwrite('./rp.png', rp.astype(np.uint8)*255)