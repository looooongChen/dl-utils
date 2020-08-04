from evaluation import *
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.morphology import dilation, cube
from skimage.io import imread
import time
import numpy as np
import matplotlib.pyplot as plt
import glob

f_gts = sorted(glob.glob('./test/cell/gt/*.png'))[0:50]
f_preds = sorted(glob.glob('./test/cell/pred/*.tif'))[0:50]

# gt = imread('./test/cell/gt/mcf-z-stacks-03212011_i01_s1_w14fc74585-6706-47ea-b84b-ed638d101ae8.png')
# pred = imread('./test/cell/pred/mcf-z-stacks-03212011_i01_s1_w14fc74585-6706-47ea-b84b-ed638d101ae8.tif')
# vis = np.stack((gt, pred, pred*0), axis=-1)

# plt.subplot(1,2,1)
# plt.imshow(label2rgb(gt))
# plt.subplot(1,2,2)
# plt.imshow(label2rgb(pred))
# plt.imshow(vis)
# plt.show()

## evalation of single image
# s = Sample(pred, gt, dimension=2, mode='area')
# print('averagePrecision', s.averagePrecision())
# print('aggregatedPrecision', s.aggregatedPrecision())
# print('averageRecall', s.averageRecall())
# print('aggregatedRecall', s.aggregatedRecall())
# print('averageF1', s.averageF1())
# print('aggregatedF1', s.aggregatedF1())
# print('aggregatedJaccard: ', s.aggregatedJaccard())
# print('aggregatedDice: ', s.aggregatedDice())
# print('averageJaccard_pred: ', s.averagedJaccard('pred'))
# print('averageJaccard_gt: ', s.averagedJaccard('gt'))
# print('averageDice_pred: ', s.averagedDice('pred'))
# print('averageDice_gt: ', s.averagedDice('gt'))
# print('SBD: ', s.SBD())
# print('match number', s.match_num(0.5, 'Jaccard'))

# evalation of a whole dataset
e = Evaluator(dimension=2, mode='area')
for f_gt, f_pred in zip(f_gts, f_preds):
    pred = imread(f_pred)
    gt = imread(f_gt)
    # add one segmentation
    e.add_example(pred, gt)

e.mAP()
e.aggregatedJaccard()
e.aggregatedDice()

