from evaluation import *
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.morphology import dilation, cube
from skimage.io import imread
import time
import numpy as np
import matplotlib.pyplot as plt
import glob

#### toy test ####
# gt = imread('./test/toy_example/gt.png')
# pred = imread('./test/toy_example/pred.png')
# sample = Sample(pred, gt)
# subject = 'pred'
# sample._computeMatch(subject=subject)
# print(sample.match_pd, sample.intersection_pd, sample.match_gt, sample.intersection_gt)
# sample._computePrecision(subject=subject)
# sample._computeRecall(subject=subject)
# sample._computeF1(subject=subject)
# sample._computeJaccard(subject=subject)
# sample._computeDice(subject=subject)
# print('precision', sample.precision_pd, sample.precision_gt)
# print('recall', sample.recall_pd, sample.recall_gt)
# print('f1', sample.f1_pd, sample.f1_gt)
# print('jaccard', sample.jaccard_pd, sample.jaccard_gt)
# print('dice', sample.dice_pd, sample.dice_gt)
# print('averagePrecision', sample.averageSegPrecision(subject))
# print('averageRecall', sample.averageSegRecall(subject))
# print('averageF1', sample.averageSegF1(subject))
# print('averageJaccard', sample.averageJaccard(subject))
# print('averageDice', sample.averageDice(subject))
# print('aggregatedJaccard', sample.aggregatedJaccard())
# print('aggregatedDice', sample.aggregatedDice())
# print('detectionPrecision', sample.detectionPrecision())
# print('AP', sample.AP())



f_gts = sorted(glob.glob('./test/cell/gt/*.png'))[0:50]
f_preds = sorted(glob.glob('./test/cell/pred/*.tif'))[0:50]

# evalation of a whole dataset
e = Evaluator(dimension=2, mode='area')
for f_gt, f_pred in zip(f_gts, f_preds):
    pred = imread(f_pred)
    gt = imread(f_gt)
    # add one segmentation
    e.add_example(pred, gt)

e.mAP()
e.averagePrecision()
e.mAJ()
e.aggregatedJaccard()
e.mAD()
e.aggregatedDice()

