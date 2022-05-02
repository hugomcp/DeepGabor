import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc
import cv2
from random import shuffle, sample
import math
import datetime
import csv
import argparse
import warnings
import matplotlib.gridspec as gridspec
from keras.layers import *
from keras.models import *
from keras.optimizers import RMSprop
from heapq import nsmallest
import time
from datetime import datetime
import random
import scipy.io
import pickle
from sklearn import metrics
from scipy.spatial.distance import cdist
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ##########################
# Configs

ap = argparse.ArgumentParser()

ap.add_argument('-d', '--dataset', required=True, help='CSV dataset file')
ap.add_argument('-do', '--dataset_original', required=True, help='CSV original dataset file')
ap.add_argument('-dc', '--dataset_corrected', required=True, help='CSV corrected dataset file')
ap.add_argument('-im', '--input_model', default=None, help='Previous model folder')
ap.add_argument('-i', '--input_folder', required=True, help='Data input folder')
ap.add_argument('-o', '--output_folder', required=True, help='Results/debug output folder')
ap.add_argument('-b', '--batch_size', type=int, default=100, help='Learning batch size')
ap.add_argument('-iw', '--image_width', type=int, default=256, help='Image width')
ap.add_argument('-ih', '--image_height', type=int, default=256, help='Image height')

args = ap.parse_args()

plt.ion()
if not os.path.isdir(args.output_folder):
    os.mkdir(args.output_folder)

date_time_folder = os.path.join(args.output_folder, 'Gabor_correct_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

if not os.path.isdir(date_time_folder):
    os.mkdir(date_time_folder)
    args_var = vars(args)
    file_out = open(os.path.join(date_time_folder, 'configs.txt'), "a+")
    file_out.write('%s\n' % ap.prog)
    for k in args_var.keys():
        file_out.write('--%s\n%s\n' % (k, args_var[k]))
    file_out.close()


########################################################################################################################
# Functions
########################################################################################################################


def read_csv_multilabel_regression(dataset, input_folder, fullpath=True):
    # ##########################
    # Load Data in '.csv' format: filename, label_1, ... label_n

    samp = []
    with open(dataset) as f:
        csv_file = csv.reader(f, delimiter=',')
        for row in csv_file:
            samp.append(row)

    for row in samp:
        for p in range(1, len(row)):
            row[p] = float(row[p])
        if fullpath:
            row[0] = os.path.join(input_folder, row[0])

    return samp

def get_input_batch_deterministic(gt, orig_set, corr_set, idx):
    tot = min(args.batch_size, len(gt) - idx)

    imgs = np.zeros((tot, args.image_height, int(args.image_width / 2), 1)).astype('float')

    labels_class = np.zeros((tot, len(gt[0])-1)).astype('float')
    orig_feat = np.zeros((tot, len(gt[0])-1)).astype('float')
    corr_feat = np.zeros((tot, len(gt[0])-1)).astype('float')

    for i in range(args.batch_size):
        if idx + i >= len(gt):
            continue
        img = cv2.imread(os.path.join(args.input_folder, gt[idx + i][0] + '_normalized_img.png'))
        # img = cv2.resize(img, (args.image_width, args.image_height*2))

        # img = np.concatenate((img, img[:, :args.largest_kernel, :]), axis=1)

        img = img[:, int(args.image_width / 2):, :]

        imgs[i, :, :, 0] = img[:, :, 0] / 255

        labels_class[i, :] = gt[idx + i][1:]
        orig_feat[i, :] = orig_set[idx + i][1:]
        corr_feat[i, :] = corr_set[idx + i][1:]

    return imgs, labels_class, orig_feat, corr_feat




def weighted_loss(weights_loss):
    def classification_inner_loss(y_true, y_pred):
        res = K.mean(weights_loss * K.square(y_true - y_pred))
        return res

    return classification_inner_loss


def set_weights_loss(gts):
    weights_loss = np.ones((args.batch_size, IDX_LABEL_END-IDX_LABEL_START)).astype('float')
    weights_loss[gts == 0] = 0.0
    return weights_loss


def correct(md, t_s, o_s, c_s):
    preds = []
    i = 0
    while i < len(t_s):

        [imgs, _, _, _] = get_input_batch_deterministic(t_s, o_s, c_s, i)

        pr = md.predict([imgs])

        preds.extend(pr)
        print('\r Done %d/%d...' % (i + args.batch_size, len(t_s)), end='')

        i += args.batch_size

    return preds


def replace_gts(dt, sai, entra):
    for r in dt:
        for p in range(1, len(r)):
            if r[p] == sai:
                r[p] = entra
    return dt


def get_stats_corrections(md, v_s):
    preds = []
    gts = []
    i = 0
    while i < len(v_s):

        [imgs, gt, _, _] = get_input_batch_deterministic(v_s, v_s, v_s, i)

        pr = md.predict([imgs])

        preds.extend(pr)
        gts.extend(gt)

        i += args.batch_size

        # print('\r Done %d/%d...' % (i, len(o_s)), end='')

    preds = np.asarray(preds)
    gts = np.asarray(gts)

    preds[gts == 0] = 0

    preds = preds.flatten()
    gts = gts.flatten()

    invalid_idx = np.argwhere(gts == 0)
    gts = np.delete(gts, invalid_idx)
    preds = np.delete(preds, invalid_idx)

    fpr, tpr, th = metrics.roc_curve(gts, preds)

    roc = np.stack((fpr, tpr, th), axis=-1)
    df = pd.DataFrame(data=roc.astype(float))
    df.to_csv(os.path.join(date_time_folder, 'optimum_ROC.txt'), sep=' ', header=['FP', 'TP', 'TH'],
              float_format='%f',
              index=False)

    # ds = cdist(np.asarray([[0, 1]]), roc[:, :2])
    # idx = np.argmin(ds)
    threshold = 0   # th[idx]

    return (np.sum(preds[gts > 0] > threshold) + np.sum(preds[gts < 0] <= threshold)) / np.sum(gts != 0), \
           preds[gts > 0], preds[gts < 0], roc

########################################################################################################################
# </Functions>
########################################################################################################################


samples = read_csv_multilabel_regression(args.dataset, args.input_folder, False)
samples_original = read_csv_multilabel_regression(args.dataset_original, args.input_folder, False)
samples_corrected = read_csv_multilabel_regression(args.dataset_corrected, args.input_folder, False)

IDX_BIT_START = 0
IDX_BIT_END = 16

samples = [[v[i] for i in [0] + list(range(1 + IDX_BIT_START, 1 + IDX_BIT_END))] for v in samples]
samples = replace_gts(samples, 9999, 0)
samples_original = [[v[i] for i in [0] + list(range(1 + IDX_BIT_START, 1 + IDX_BIT_END))] for v in samples_original]
samples_original = replace_gts(samples_original, 9999, 0)
samples_corrected = [[v[i] for i in [0] + list(range(1 + IDX_BIT_START, 1 + IDX_BIT_END))] for v in samples_corrected]
samples_corrected = replace_gts(samples_corrected, 9999, 0)


weights_loss_global = Input((IDX_BIT_END-IDX_BIT_START,))
model = load_model(os.path.join(args.input_model, 'Regressor.h5'), compile=False,
                   custom_objects={'loss': [weighted_loss(weights_loss_global)]})

corrected_codes = correct(model, samples, samples_original, samples_corrected)

with open(os.path.join(date_time_folder, 'corrected_features_Gabor_out.csv'), 'w') as f:
    csv_file = csv.writer(f, delimiter=',')
    for el in corrected_codes:
        csv_file.writerow(el)

with open(os.path.join(args.input_model, 'partition_sets.dat'), 'rb') as f:
    [learning_samples, validation_samples] = pickle.load(f)
scipy.io.savemat(os.path.join(date_time_folder, 'partition_sets.mat'),
                 mdict={'learning_samples': [el[0] for el in learning_samples], 'validation_samples': [el[0] for el in validation_samples]})

stats_corrections_v = get_stats_corrections(model, validation_samples)

print('Checking stats...Acc=%f' % stats_corrections_v[0])


