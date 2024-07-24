#!/usr/bin/env python3
# coding: utf-8
# pylint: disable=E1101,C0103

'''
A set of metrics for evaluatation of eye-gaze scanpaths,
adapted from https://github.com/tarunsharma1/saliency_metrics

NOTE: Some of these functions aren't correct.

Every function expects at least two arguments:
- Predicted heatmap image, as a Numpy array.
- Reference heatmap image, as a Numpy array.

External dependencies, to be installed e.g. via pip:
- numpy

Authors:
- Luis A. Leiva <name.surname@uni.lu>
'''

import math
import numpy as np


EPS = np.finfo(np.float32).eps

def rescale(any_map):
    '''
    Apply mimaxscaler algorithm.
    See https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    '''
    return (any_map - any_map.min()) / (EPS + any_map.max() - any_map.min())


def normalize(any_map):
    '''
    Normalize heatmap values in the [0,1] range.
    '''
    any_map = rescale(any_map)
    any_map /= (EPS + any_map.sum())
    return any_map


def whiten(any_map):
    '''
    Apply whitening algorithm.
    See https://en.wikipedia.org/wiki/Whitening_transformation
    '''
    return (any_map - any_map.mean()) / (EPS + any_map.std())


def discretize(any_map):
    '''
    Normalize values in [0,1].
    NB: Actually this isn't a discretization.
    '''
    return any_map / 255


def auc_judd(sal_map, ref_map):
    '''
    Compute Judd's AUC score.
    '''
    # ref_map (groundtruth) is discrete, sal_map is continous and normalized (whiten)
    sal_map = rescale(sal_map)
#    ref_map = rescale(ref_map)

    ref_map = discretize(ref_map)
    num_fixations = np.sum(ref_map)

    # thresholds are calculated from the salience map, only at places where fixations are present
    thresholds = []
    for i in range(0, ref_map.shape[0]):
        for k in range(0, ref_map.shape[1]):
            if ref_map[i][k] > 0:
                thresholds.append(sal_map[i][k])

    thresholds = sorted(set(thresholds))

    # fp_list = []
    # tp_list = []
    area = []
    area.append((0.0, 0.0))
    for thresh in thresholds:
        # in the salience map, keep only those pixels with values above
        # threshold
        temp = np.zeros(sal_map.shape)
        temp[sal_map >= thresh] = 1.0
        assert np.max(ref_map) == 1.0, 'something is wrong with ground truth..not discretized properly max value > 1.0'
        assert np.max(sal_map) == 1.0, 'something is wrong with salience map..not normalized properly max value > 1.0'
        num_overlap = np.where(np.add(temp, ref_map) == 2)[0].shape[0]
        tp = num_overlap / (num_fixations * 1.0)

        # total number of pixels > threshold - number of pixels that overlap with gt / total number of non fixated pixels
        # this becomes nan when gt is full of fixations..this won't happen
        fp = (np.sum(temp) - num_overlap) / ((np.shape(ref_map)[0] * np.shape(ref_map)[1]) - num_fixations)

        area.append((round(tp, 4), round(fp, 4)))
        # tp_list.append(tp)
        # fp_list.append(fp)

    # tp_list.reverse()
    # fp_list.reverse()
    area.append((1.0, 1.0))
    # tp_list.append(1.0)
    # fp_list.append(1.0)
    # print tp_list
    area.sort(key=lambda x: x[0])
    tp_list = [x[0] for x in area]
    fp_list = [x[1] for x in area]
    return np.trapz(np.array(tp_list), np.array(fp_list))


def auc_borji(sal_map, ref_map, splits=100, stepsize=0.1):
    '''
    Compute Borji's AUC score.
    '''
    ref_map = discretize(ref_map)
    num_fixations = np.sum(ref_map).astype(int)

    num_pixels = sal_map.shape[0] * sal_map.shape[1]
    random_numbers = []
    for i in range(0, splits):
        temp_list = []
        for k in range(0, num_fixations):
            temp_list.append(np.random.randint(num_pixels))
        random_numbers.append(temp_list)

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(sal_map[k % sal_map.shape[0] - 1, math.floor(k/sal_map.shape[0])])
        # in these values, we need to find thresholds and calculate auc
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        r_sal_map = np.array(r_sal_map)

        # once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above
            # threshold
            temp = np.zeros(sal_map.shape)
            temp[sal_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, ref_map) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # fp = (np.sum(temp) - num_overlap)/((np.shape(ref_map)[0] * np.shape(ref_map)[1]) - num_fixations)
            # number of values in r_sal_map, above the threshold, divided by num
            # of random locations = num of fixations
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list), np.array(fp_list)))

    return np.mean(aucs)


def auc_shuff(sal_map, ref_map, rand_map=None, splits=100, stepsize=0.1):
    '''
    Compute shuffled AUC score.
    '''
    if rand_map is None:
        height, width = sal_map.shape
        rand_map = np.zeros((height, width))

    ref_map = discretize(ref_map)
    rand_map = discretize(rand_map)

    num_fixations = np.sum(ref_map)

    x, y = np.where(rand_map == 1)
    rand_map_fixs = []
    for j in zip(x, y):
        rand_map_fixs.append(j[0] * rand_map.shape[0] + j[1])

    ind = len(rand_map_fixs)
    assert ind == np.sum(rand_map), 'something is wrong in auc shuffle'

    num_fixations_other = min(ind, num_fixations)

    num_pixels = sal_map.shape[0] * sal_map.shape[1]
    random_numbers = []
    for i in range(0, splits):
        temp_list = []
        t1 = np.random.permutation(ind)
        for k in t1:
            temp_list.append(rand_map_fixs[k])
        random_numbers.append(temp_list)

    aucs = []
    # for each split, calculate auc
    for i in random_numbers:
        r_sal_map = []
        for k in i:
            r_sal_map.append(sal_map[k % sal_map.shape[0] - 1, k / sal_map.shape[0]])
        # in these values, we need to find thresholds and calculate auc
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        r_sal_map = np.array(r_sal_map)

        # once threshs are got
        thresholds = sorted(set(thresholds))
        area = []
        area.append((0.0, 0.0))
        for thresh in thresholds:
            # in the salience map, keep only those pixels with values above
            # threshold
            temp = np.zeros(sal_map.shape)
            temp[sal_map >= thresh] = 1.0
            num_overlap = np.where(np.add(temp, ref_map) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)

            # fp = (np.sum(temp) - num_overlap)/((np.shape(ref_map)[0] * np.shape(ref_map)[1]) - num_fixations)
            # number of values in r_sal_map, above the threshold, divided by
            # num of random locations = num of fixations
            fp = len(np.where(r_sal_map > thresh)[0]) / (num_fixations * 1.0)

            area.append((round(tp, 4), round(fp, 4)))

        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]

        aucs.append(np.trapz(np.array(tp_list),np.array(fp_list)))

    return np.mean(aucs)


def auc(sal_map, ref_map):
    '''
    Compute Judd's AUC score.
    '''
    return auc_judd(sal_map, ref_map)


def nss(sal_map, ref_map):
    '''
    Compute Normalized Scanpath Saliency score.
    '''
    ref_map = discretize(ref_map)
    sal_map_norm = whiten(sal_map)

    x,y = np.where(ref_map == 1)
    temp = []
    for i in zip(x,y):
        temp.append(sal_map_norm[i[0],i[1]])
    return np.mean(temp)


def infogain(sal_map, ref_map, rand_map=None):
    '''
    Compute InfoGain score.
    '''
    sal_map_norm = normalize(sal_map)
    ref_map_norm = normalize(ref_map)

    if rand_map is None:
        height, width = sal_map.shape
        rand_map = np.zeros((height, width))

    ref_map = discretize(ref_map)

    # for all places where gt=1, calculate info gain
    temp = []
    x,y = np.where(ref_map == 1)
    for i in zip(x,y):
        temp.append(np.log2(EPS + sal_map[i[0],i[1]]) - np.log2(EPS + rand_map[i[0],i[1]]))

    return np.mean(temp)


def similarity(sal_map, ref_map):
    '''
    Compute Similarity score.
    '''
    # here ref_map is not discretized nor normalized
    sal_map = normalize(sal_map)
    ref_map = normalize(ref_map)
    sal_map = sal_map / np.sum(sal_map)
    ref_map = ref_map / np.sum(ref_map)
    x,y = np.where(ref_map > 0)
    sim = 0.0
    for i in zip(x,y):
        sim = sim + min(ref_map[i[0], i[1]], sal_map[i[0], i[1]])
    return sim


def cc(sal_map, ref_map):
    '''
    Compute Coefficient of Correlation score.
    '''
    sal_map_norm = whiten(sal_map)
    ref_map_norm = whiten(ref_map)
    a = sal_map_norm
    b = ref_map_norm
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r


def kldiv(sal_map, ref_map):
    '''
    Compute Kullback-Leibler divergence.
    '''
    sal_map = sal_map / np.sum(sal_map)
    ref_map = ref_map / np.sum(ref_map)
    return np.sum(ref_map * np.log(EPS + ref_map/(sal_map + EPS)))
