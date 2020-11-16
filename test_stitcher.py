from stitcher import Splitter, Stitcher
from skimage.io import imread
import pprint
import numpy as np
from skimage.measure import label
import cv2


# a = imread('./test/cell/gt/mcf-z-stacks-03212011_i01_s1_w14fc74585-6706-47ea-b84b-ed638d101ae8.png')
a = cv2.imread('./test/land.jpg')

sp = Splitter()
sp.patches2D(a, [200, 200], [20, 20])
# for k, v in sp.patches.items():
#     sp.patches[k] = label(v)
sp.save('./test_patches')

st = Stitcher('./test_patches')
img = st.stitch2D(mode='raw')
cv2.imwrite('tt.png', img.astype(np.uint8))
