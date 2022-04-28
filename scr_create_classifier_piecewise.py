import os
import matplotlib.pyplot as plt
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from tensorflow import keras
import cv2
from random import shuffle, sample
import pandas as pd
import math
import datetime
import csv
import argparse
import warnings
import matplotlib.gridspec as gridspec
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from heapq import nsmallest, nlargest
import time
from datetime import datetime
import random
from collections import Counter
import pickle
from matplotlib.offsetbox import AnchoredText
from sklearn import metrics
from scipy.spatial.distance import cdist
from tensorflow.keras.applications import inception_v3
from tensorflow.keras import backend as K
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.optimizers import SGD



# ##########################
# Configs
ap = argparse.ArgumentParser()

ap.add_argument('-d', '--dataset', required=True, help='CSV learning dataset file')
ap.add_argument('-if', '--inf_filters', required=True, help='CSV filters information file')
ap.add_argument('-id', '--idx_filters', required=True, help='CSV index filters information file')
ap.add_argument('-s', '--scale_kernels', default=[3, 9, 17, 49], type=list, help='First layer filter sizes')
ap.add_argument('-i', '--input_folder', required=True, help='Data input folder')
ap.add_argument('-ao', '--activation_output', default='linear', help='Activation function output layer')
ap.add_argument('-lf', '--loss_function', default='l1', help='Loss function')
ap.add_argument('-o', '--output_folder', required=True, help='Results/debug output folder')
ap.add_argument('-b', '--batch_size', type=int, default=100, help='Learning batch size')
ap.add_argument('-pl', '--proportion_learning', type=float, default=0.9, help='Proportion of annotations/learning')
ap.add_argument('-iw', '--image_width', type=int, default=256, help='Image width')
ap.add_argument('-ih', '--image_height', type=int, default=64, help='Image height')
ap.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Learning rate')
ap.add_argument('-de', '--decay_rate', type=float, default=1e-2, help='Decay rate')
ap.add_argument('-dr', '--dropout_rate', type=float, default=0.25, help='Dropout rate')
ap.add_argument('-pa', '--prior_augment', type=float, default=0.3, help='Prior probability for augmentation operator')
ap.add_argument('-e', '--epochs', type=int, default=10, help='Tot. epochs')
ap.add_argument('-p', '--patience', type=int, default=0,
                help='Maximum number of consecutive iterations increasing loss')
ap.add_argument('-tn', '--total_neurons_first_layer', type=float, default=1.0,
                help='Proportion of neurons/TOT_LABELS in the first layer')
ap.add_argument('-igo', '--interval_plot_results', type=int, default=100,
                help='Number of epochs to plot results')

args = ap.parse_args()

plt.ion()
if not os.path.isdir(args.output_folder):
    os.mkdir(args.output_folder)

name_dataset = os.path.split(args.dataset)[1]
pos = [p for p, char in enumerate(name_dataset) if char == '_']
date_time_folder = os.path.join(args.output_folder,
                                'Learn_' + name_dataset[:pos[3]] + '_' + datetime.now().strftime(
                                    "%Y_%m_%d_%H_%M_%S"))

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

def replace_gts(dt, sai, entra):
    for r in dt:
        for p in range(1, len(r)):
            if r[p] == sai:
                r[p] = entra
    return dt


def get_Gabor_filters(wg, og, pg, rg):
    ret = []
    for w in wg:
        for o in og:
            for p in pg:
                for r in rg:
                    sigma_x = 0.5 * w
                    sigma_y = 0.5 * w / r

                    # Bounding box
                    nstds = 3
                    xmax = max(abs(nstds * sigma_x * np.cos(o)), abs(nstds * sigma_y * np.sin(o)))
                    xmax = np.ceil(max(1, xmax))
                    ymax = max(abs(nstds * sigma_x * np.sin(o)), abs(nstds * sigma_y * np.cos(o)))
                    ymax = np.ceil(max(1, ymax))
                    xmin = -xmax
                    ymin = -ymax
                    [x, y] = np.meshgrid(np.linspace(xmin, xmax), np.linspace(ymin, ymax))

                    # Rotation
                    x_theta = x * np.cos(o) + y * np.sin(o)
                    y_theta = -x * np.sin(o) + y * np.cos(o)

                    elt = [1 / (2 * np.pi * sigma_x * sigma_y) *
                           np.multiply(np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)),
                                       np.cos(2 * np.pi / w * x_theta + p)),
                           1 / (2 * np.pi * sigma_x * sigma_y) *
                           np.multiply(np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)),
                                       np.sin(2 * np.pi / w * x_theta + p))]

                    # remove energy
                    elt[0] = elt[0] - np.mean(elt[0])
                    elt[1] = elt[1] - np.mean(elt[1])
                    ret.append(elt)
    return ret


def read_csv(dataset, input_folder, fullpath=True):
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


def get_subjects_casia(paths):
    ret = np.zeros(len(paths)).astype('int')
    for idx in range(len(paths)):
        ret[idx] = int(paths[idx][1:5]) * 2 + (paths[idx][5] == 'L')
    return ret


def split_set_casia(s):
    subjects_un = list(set(get_subjects_casia([p[0] for p in s])))
    random.shuffle(subjects_un)

    subjects_learning = subjects_un[:math.floor(len(subjects_un) * args.proportion_learning)]

    subjects_anot_pairs = get_subjects_casia([p[0] for p in s])

    s_learning = []
    s_validation = []
    l_idx = []
    v_idx = []
    for idx, a in enumerate(s):
        if subjects_anot_pairs[idx] in subjects_learning:
            s_learning.append(a)
            l_idx.append(idx)
        else:
            s_validation.append(a)
            v_idx.append(idx)

    return s_learning, s_validation, l_idx, v_idx


def split_set(s):
    idx = random.sample(range(len(s)), round(len(s) * args.proportion_learning))

    s_learning = []
    s_validation = []
    l_idx = []
    v_idx = []
    for id, a in enumerate(s):
        if id in idx:
            s_learning.append(a)
            l_idx.append(id)
        else:
            s_validation.append(a)
            v_idx.append(id)

    return s_learning, s_validation, l_idx, v_idx


def get_input_batch(dt):
    imgs = np.zeros((args.batch_size, args.image_height, args.image_width, 1)).astype('float')
    labels = np.zeros((args.batch_size, len(dt[0]) - 1)).astype('float')

    idx = random.sample(range(len(dt)), args.batch_size)
    for i in range(args.batch_size):
        img = cv2.imread(os.path.join(args.input_folder, dt[idx[i]][0]))
        # img = cv2.resize(img, (args.image_width, args.image_height * 2))
        imgs[i, :, :, 0] = img[:, :, 0] / 255
        labels[i, :] = dt[idx[i]][1:]

    return imgs, labels


def save_best_model(mod, results):
    criterium = results  # loss
    if criterium.index(min(criterium)) != len(criterium) - 1:
        return 0
    nv_best = nsmallest(2, criterium)

    if len(nv_best) == 1:
        improvement = 0.0
    else:
        improvement = 100 - (nv_best[0] * 100) / nv_best[1]
    obj_now = datetime.now()
    print(' [%02d:%02d:%02d]. Saving best model. Curr %.3f (Improved %.3f%%). ' % (
        obj_now.hour, obj_now.minute, obj_now.second, nv_best[0], improvement))
    mod.save(os.path.join(date_time_folder, 'Regressor.h5'))

    return 1


def loss_hinge(y_true, y_pred):
    (y_true, weights_loss) = tf.split(y_true, num_or_size_splits=[-1, 256], axis=-1)
    res = K.mean(weights_loss * K.maximum(0.0, 1.0 - y_true * y_pred))
    # res = K.mean(K.maximum(0.0, 1.0 - y_true * y_pred))
    return res


def set_weights_loss(gts):
    weights_loss = np.zeros((np.shape(gts)[0], np.shape(gts)[1])).astype('float')
    for col in range(np.shape(gts)[1]):
        weights_loss[gts[:, col] == 1, col] = PRIOR_NEG_l[col] / 0.5
        weights_loss[gts[:, col] == -1, col] = PRIOR_POS_l[col] / 0.5

    return weights_loss


def get_input_batch_idx(gt, idx, augm=None):
    imgs = np.zeros((len(idx), args.image_height, int(args.image_width / 2), 1)).astype('float')
    labels = np.zeros((len(idx), len(gt[0]) - 1)).astype('float')

    for i in range(len(idx)):
        img = cv2.imread(os.path.join(args.input_folder, gt[idx[i]][0] + '_normalized_img.png'))

        if augm is not None:
            img = augm.augment_image(img)

        # img = cv2.resize(img, (args.image_width, args.image_height*2))
        img = img[:, int(args.image_width / 2):, :]

        imgs[i, :, :, 0] = img[:, :, 0] / 255
        labels[i, :] = gt[idx[i]][1:]

    return imgs, labels


def get_input_batch_deterministic(gt, idx):
    tot = min(args.batch_size, len(gt) - idx)

    imgs = np.zeros((tot, args.image_height, int(args.image_width / 2), 1)).astype('float')
    labels = np.zeros((tot, len(gt[0]) - 1)).astype('float')

    for i in range(tot):
        img = cv2.imread(os.path.join(args.input_folder, gt[idx + i][0] + '_normalized_img.png'))
        # img = cv2.resize(img, (args.image_width, args.image_height*2))
        img = img[:, int(args.image_width / 2):, :]

        imgs[i, :, :, 0] = img[:, :, 0] / 255
        labels[i, :] = gt[idx + i][1:]

    return imgs, labels


def get_stats_corrections(md, v_s):
    preds = []
    gts = []
    i = 0
    while i < len(v_s):
        imgs, gt = get_input_batch_deterministic(v_s, i)

        pr = md.predict(imgs)

        preds.extend(pr)
        gts.extend(gt)

        i += args.batch_size

        # print('\r Done %d/%d...' % (i, len(o_s)), end='')

    preds = np.asarray(preds)
    gts = np.asarray(gts)

    preds = preds.flatten()
    gts = gts.flatten()

    invalid_idx = np.argwhere(gts == 0)
    gts = np.delete(gts, invalid_idx)
    preds = np.delete(preds, invalid_idx)

    fpr, tpr, th = metrics.roc_curve(gts, preds)
    auc = metrics.roc_auc_score(gts, preds)

    roc = np.stack((fpr, tpr, th), axis=-1)
    df = pd.DataFrame(data=roc.astype(float))
    df.to_csv(os.path.join(date_time_folder, 'optimum_ROC.txt'), sep=' ', header=['FP', 'TP', 'TH'],
              float_format='%f',
              index=False)

    # ds = cdist(np.asarray([[0, 1]]), roc[:, :2])
    # idx = np.argmin(ds)
    threshold = 0  # th[idx]

    return (np.sum(preds[gts > 0] > threshold) + np.sum(preds[gts < 0] <= threshold)) / np.sum(gts != 0), \
           preds[gts > 0], preds[gts < 0], roc, auc


def get_learning_rate(ep):
    return args.learning_rate * 1.0 / (1 + args.decay_rate * ep)


def train_full(md, l_s, v_s):
    lo_l = []
    lo_v = []
    losses_learn = []
    losses_valid = []

    accuracy_v = []
    best_roc = []
    cur_pos = []
    cur_neg = []

    augmenter = iaa.Sequential([
        iaa.Sometimes(args.prior_augment, iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.Sometimes(args.prior_augment, iaa.LinearContrast((0.75, 1.5))),
        iaa.Sometimes(args.prior_augment, iaa.Multiply((0.8, 1.2), per_channel=0.2))],
        random_order=True)

    saved_model_epoch = -1
    improvements = []
    for epoch in range(1, args.epochs):
        idx = np.random.permutation(len(l_s))
        i = 0
        learn_rate = get_learning_rate(epoch - 1)
        K.set_value(md.optimizer.lr, learn_rate)
        while i < len(l_s):
            [imgs, gt] = get_input_batch_idx(l_s, idx[i:min(len(l_s), i + args.batch_size)], augmenter)
            weights_loss = set_weights_loss(gt)
            gtc = np.concatenate((gt, weights_loss), axis=-1)
            lo = md.train_on_batch(imgs, gtc)
            lo_l.append(lo)
            i += args.batch_size
            print('\r Learn [%d - %d/%d], lr=%f...' % (epoch, i, len(l_s), learn_rate), end='')

        idx = np.random.permutation(len(v_s))
        i = 0
        while i < len(v_s):
            [imgs, gt] = get_input_batch_idx(v_s, idx[i:min(len(v_s), i + args.batch_size)])
            weights_loss = set_weights_loss(gt)
            gtc = np.concatenate((gt, weights_loss), axis=-1)
            lo = md.test_on_batch(imgs, gtc)
            lo_v.append(lo)
            i += args.batch_size
            print('\r Valid [%d - %d/%d]...' % (epoch, i, len(v_s)), end='')

        print('\r Epoch %d...' % epoch, end='')
        losses_learn.append(np.mean(lo_l, axis=0))
        losses_valid.append(np.mean(lo_v, axis=0))
        lo_l = []
        lo_v = []

        if args.patience > 0:
            if len(losses_valid) - np.argmin(losses_valid) - 1 > args.patience:
                return -1

        if np.argmin(losses_valid) == len(losses_valid) - 1:
            improvements.extend([0])
        else:
            improvements[-1] += 1

        acc, cur_pos, cur_neg, cur_roc, cur_auc = get_stats_corrections(md, v_s)

        accuracy_v.append(acc)

        saved_model_flag = save_best_model(md, losses_valid)
        if saved_model_flag:
            saved_model_epoch = epoch - 1
            best_roc = cur_roc
            best_pos = cur_pos
            best_neg = cur_neg
            best_auc = cur_auc
            best_acc = acc

        ep = range(1, epoch + 1)
        fig_1 = plt.figure(1, figsize=(18, 8))
        plt.clf()
        gs = gridspec.GridSpec(3, 5, figure=fig_1)

        ax = fig_1.add_subplot(gs[0:2, 0])
        ax.plot(ep, losses_learn, '-g')
        ax.plot(ep, losses_valid, '-r')
        ax.plot(ep[saved_model_epoch], losses_valid[saved_model_epoch], 'xk')
        ax.plot([ep[saved_model_epoch], ep[saved_model_epoch]],
                [losses_valid[saved_model_epoch], losses_learn[saved_model_epoch]], '-k')
        ax.grid(True)
        ax.title.set_text('All')

        ax = fig_1.add_subplot(gs[0:2, 1])
        ax.plot(ep[-50:], losses_learn[-50:], '-g')
        ax.plot(ep[-50:], losses_valid[-50:], '-r')
        ax.set_facecolor((0.9, 0.9, 0.7))
        ax.grid(True)

        ax = fig_1.add_subplot(gs[0:2, 2])
        ax.semilogy(ep, losses_learn, '-g')
        ax.semilogy(ep, losses_valid, '-r')
        ax.semilogy(ep[saved_model_epoch], losses_valid[saved_model_epoch], 'xk')
        ax.semilogy([ep[saved_model_epoch], ep[saved_model_epoch]],
                    [losses_valid[saved_model_epoch], losses_learn[saved_model_epoch]], '-k')
        ax.grid(True)

        ax = fig_1.add_subplot(gs[0:2, 3])
        ax.semilogy(ep[-50:], losses_learn[-50:], '-g')
        ax.semilogy(ep[-50:], losses_valid[-50:], '-r')
        ax.set_facecolor((0.9, 0.9, 0.7))
        ax.grid(True)

        ax = fig_1.add_subplot(gs[2, 2])
        ax.plot(cur_roc[:, 0], cur_roc[:, 1], '-b')
        ax.plot(best_roc[:, 0], best_roc[:, 1], '-g')
        ax.plot([0, 1], [0, 1], '--k')
        ax.set_facecolor((1, 1, 1))
        ax.grid(True)
        ax.add_artist(AnchoredText('Cur %.3f, Best %.3f' % (cur_auc, best_auc), loc=4))

        ax = fig_1.add_subplot(gs[2, 3])
        ax.semilogx(cur_roc[:, 0], cur_roc[:, 1], '-b')
        ax.semilogx(best_roc[:, 0], best_roc[:, 1], '-g')
        ax.semilogx(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--k')
        ax.set_facecolor((1, 1, 1))
        ax.grid(True)

        ax = fig_1.add_subplot(gs[0, 4])
        cnts = Counter(improvements)
        ax.plot([len(losses_valid) - np.argmin(losses_valid) - 1, len(losses_valid) - np.argmin(losses_valid) - 1],
                [0, 1], '-k')
        (_, most_freq) = cnts.most_common(1)[0]
        for k, v in cnts.items():
            ax.plot([k, k], [0, v / most_freq], '-g', linewidth=5)
        ax.plot([0, 0], [0, 1.0], '--k')
        if args.patience > 0:
            ax.plot([args.patience, args.patience], [0, 1.0],
                    '--k')
        ax.set_facecolor((0.9, 0.4, 0.4))
        ax.grid(True)

        ax = fig_1.add_subplot(gs[2, 0:2])
        ax.plot(ep, accuracy_v, '-r')
        ax.set_facecolor((1, 1, 1))
        ax.grid(True)
        ax.set_xlabel('Acc')
        ax.add_artist(AnchoredText('C %.3f, B %.3f (All %.3f)' % (accuracy_v[-1], best_acc, max(accuracy_v)), loc=4))

        ax = fig_1.add_subplot(gs[1, 4])
        freqs_neg, edges_neg = np.histogram(cur_neg, bins=50)
        freqs_neg = np.divide(freqs_neg, np.sum(freqs_neg))
        bins_neg = (edges_neg[:-1] + edges_neg[1:]) / 2
        ax.bar(bins_neg, freqs_neg, width=bins_neg[1] - bins_neg[0], align='center', alpha=0.5,
               color=(1, 0, 0, 0.5))
        freqs_pos, edges_pos = np.histogram(cur_pos, bins=50)
        freqs_pos = np.divide(freqs_pos, np.sum(freqs_pos))
        bins_pos = (edges_pos[:-1] + edges_pos[1:]) / 2
        ax.bar(bins_pos, freqs_pos, width=bins_pos[1] - bins_pos[0], align='center', alpha=0.5,
               color=(0, 1, 0, 0.5))
        ax.set_xlabel('Scores')
        ax.set_ylabel('Density')

        ax = fig_1.add_subplot(gs[2, 4])
        freqs_neg, edges_neg = np.histogram(best_neg, bins=50)
        freqs_neg = np.divide(freqs_neg, np.sum(freqs_neg))
        bins_neg = (edges_neg[:-1] + edges_neg[1:]) / 2
        ax.bar(bins_neg, freqs_neg, width=bins_neg[1] - bins_neg[0], align='center', alpha=0.5,
               color=(1, 0, 0, 0.5))
        freqs_pos, edges_pos = np.histogram(best_pos, bins=50)
        freqs_pos = np.divide(freqs_pos, np.sum(freqs_pos))
        bins_pos = (edges_pos[:-1] + edges_pos[1:]) / 2
        ax.bar(bins_pos, freqs_pos, width=bins_pos[1] - bins_pos[0], align='center', alpha=0.5,
               color=(0, 1, 0, 0.5))
        ax.set_facecolor((0.9, 0.9, 0.9))
        ax.set_xlabel('Scores')
        ax.set_ylabel('Density')

        # fig_1.show()
        # plt.pause(0.01)

        plt.savefig(os.path.join(date_time_folder, 'Learning.png'))
    return 1


def read_csv_inf_filters(pat):
    # Load information of source regions for each filter in '.csv' format: [y_begin, y_end, x_begin, x_end]

    samples = []
    with open(pat) as f:
        csv_file = csv.reader(f, delimiter=',')
        for row in csv_file:
            samples.append(row)

    for row in samples:
        for p in range(0, len(row)):
            row[p] = int(row[p])

    return samples


def create_cnn(dt, inf_feat):
    imgs_input = Input((args.image_height, int(args.image_width / 2), 1))

    first_layers = []
    for sc in args.scale_kernels:
        conv11 = Conv2D(round((len(dt[0]) - 1) * args.total_neurons_first_layer),
                        kernel_size=[sc, sc],
                        strides=1,
                        input_shape=(args.image_height, int(args.image_width / 2), 1), padding="same")(imgs_input)
        conv11_a = LeakyReLU()(conv11)
        # drop11 = conv11_a
        drop11 = Dropout(args.dropout_rate)(conv11_a)
        first_layers.append(drop11)

    first_layer = Concatenate()([p for p in first_layers])

    FACTOR = 0.5

    conv12 = Conv2D(int(64 * FACTOR), kernel_size=3, strides=2, padding="same")(first_layer)
    conv12_bn = BatchNormalization(momentum=0.8)(conv12)
    conv12_a = LeakyReLU()(conv12_bn)
    # drop12 = conv12_a
    drop12 = Dropout(args.dropout_rate)(conv12_a)

    conv13 = Conv2D(int(128 * FACTOR), kernel_size=3, strides=2, padding="same")(drop12)
    conv13_bn = BatchNormalization(momentum=0.8)(conv13)
    conv13_a = LeakyReLU()(conv13_bn)
    # drop13 = conv13_a  # Dropout(args.dropout_rate)(conv13_a)
    drop13 = Dropout(args.dropout_rate)(conv13_a)

    conv14 = Conv2D(int(256 * FACTOR), kernel_size=3, strides=2, padding="same")(drop13)
    conv14_bn = BatchNormalization(momentum=0.8)(conv14)
    conv14_a = LeakyReLU()(conv14_bn)
    # drop14 = conv14_a
    drop14 = Dropout(args.dropout_rate)(conv14_a)

    conv15 = Conv2D(int(512 * FACTOR), kernel_size=3, strides=2, padding="same")(drop14)
    conv15_bn = BatchNormalization(momentum=0.8)(conv15)
    conv15_a = LeakyReLU()(conv15_bn)
    # drop15 = conv15_a
    drop15 = Dropout(args.dropout_rate)(conv15_a)

    pooled = GlobalAveragePooling2D()(drop15)
    # pooled_1 = Flatten()(pooled)
    pooled_1 = pooled

    pieces = []
    accumulated_shrink = 8
    margin = 5
    for ft in inf_feat:
        piece = Cropping2D(cropping=((min(int((ft[0] - margin) / accumulated_shrink), 0),
                                      min(0, int((args.image_height - ft[1] - margin) / accumulated_shrink))),
                                     (min(int((ft[2] - margin) / accumulated_shrink), 0),
                                      min(int((args.image_width - ft[3] - margin) / accumulated_shrink), 0))))(drop15)
        piece = Flatten()(piece)
        piece_concat = Concatenate()([piece, pooled_1])

        dense1 = Dense(128, activation='relu', kernel_constraint=None)(piece_concat)
        drop1 = Dropout(args.dropout_rate)(dense1)

        dense2 = Dense(64, activation='relu', kernel_constraint=None)(drop1)
        drop2 = Dropout(args.dropout_rate)(dense2)

        outp = Dense(1, activation=args.activation_output, kernel_constraint=None)(drop2)

        pieces.append(outp)

    if len(pieces) > 1:
        output = Concatenate()([p for p in pieces])
    else:
        output = pieces[0]

    md = Model(inputs=imgs_input, outputs=output)

    md.compile(optimizer=SGD(lr=args.learning_rate, momentum=0.8), loss=[loss_hinge])

    md.summary()
    return md


def create_cnn_2(dt, inf_feat, Gabor_k):
    imgs_input = Input((args.image_height, int(args.image_width / 2), 1))

    Gabor_layers = []
    for idx, gk in enumerate(Gabor_k):
        conv01 = Conv2D(1, name='conv2d_gabor_%d' % idx, kernel_size=[len(gk), len(gk)], strides=1,
                        input_shape=(args.image_height, int(args.image_width / 2), 1), padding="same")(imgs_input)
        Gabor_layers.append(conv01)

    Gabor_layer = Concatenate()([p for p in Gabor_layers])

    first_layers = []
    for sc in args.scale_kernels:
        conv11 = Conv2D(round((len(dt[0]) - 1) * args.total_neurons_first_layer),
                        kernel_size=[sc, sc],
                        strides=1,
                        input_shape=(args.image_height, int(args.image_width / 2), 1), padding="same")(imgs_input)
        conv11_a = LeakyReLU()(conv11)
        # drop11 = conv11_a
        drop11 = Dropout(args.dropout_rate)(conv11_a)
        first_layers.append(drop11)

    first_layer = Concatenate()([p for p in first_layers])

    all_first = Concatenate()([Gabor_layer, first_layer])

    FACTOR = 0.5

    conv12 = Conv2D(int(64 * FACTOR), kernel_size=3, strides=2, padding="same")(all_first)
    conv12_bn = BatchNormalization(momentum=0.8)(conv12)
    conv12_a = LeakyReLU()(conv12_bn)
    # drop12 = conv12_a
    drop12 = Dropout(args.dropout_rate)(conv12_a)

    conv13 = Conv2D(int(128 * FACTOR), kernel_size=3, strides=2, padding="same")(drop12)
    conv13_bn = BatchNormalization(momentum=0.8)(conv13)
    conv13_a = LeakyReLU()(conv13_bn)
    # drop13 = conv13_a  # Dropout(args.dropout_rate)(conv13_a)
    drop13 = Dropout(args.dropout_rate)(conv13_a)

    conv14 = Conv2D(int(256 * FACTOR), kernel_size=3, strides=2, padding="same")(drop13)
    conv14_bn = BatchNormalization(momentum=0.8)(conv14)
    conv14_a = LeakyReLU()(conv14_bn)
    # drop14 = conv14_a
    drop14 = Dropout(args.dropout_rate)(conv14_a)

    conv15 = Conv2D(int(512 * FACTOR), kernel_size=3, strides=2, padding="same")(drop14)
    conv15_bn = BatchNormalization(momentum=0.8)(conv15)
    conv15_a = LeakyReLU()(conv15_bn)
    # drop15 = conv15_a
    drop15 = Dropout(args.dropout_rate)(conv15_a)

    pooled = GlobalAveragePooling2D()(drop15)
    # pooled_1 = Flatten()(pooled)
    pooled_1 = pooled

    pieces = []
    accumulated_shrink = 8
    margin = 5
    for ft in inf_feat:
        piece = Cropping2D(cropping=((min(int((ft[0] - margin) / accumulated_shrink), 0),
                                      min(0, int((args.image_height - ft[1] - margin) / accumulated_shrink))),
                                     (min(int((ft[2] - margin) / accumulated_shrink), 0),
                                      min(int((args.image_width - ft[3] - margin) / accumulated_shrink), 0))))(drop15)
        piece = Flatten()(piece)
        piece_concat = Concatenate()([piece, pooled_1])

        dense1 = Dense(128, activation='relu', kernel_constraint=None)(piece_concat)
        drop1 = Dropout(args.dropout_rate)(dense1)

        dense2 = Dense(64, activation='relu', kernel_constraint=None)(drop1)
        drop2 = Dropout(args.dropout_rate)(dense2)

        outp = Dense(1, activation=args.activation_output, kernel_constraint=None)(drop2)

        pieces.append(outp)

    if len(pieces) > 1:
        output = Concatenate()([p for p in pieces])
    else:
        output = pieces[0]

    md = Model(inputs=imgs_input, outputs=output)
    md.compile(optimizer=SGD(lr=args.learning_rate, momentum=0.8), loss=[loss_hinge], sample_weight_mode='temporal')
    md.summary()

    for idx, gk in enumerate(Gabor_k):
        w = md.get_layer(name='conv2d_gabor_%d' % idx).get_weights()
        w[0][:, :, 0, 0] = gk
        md.get_layer(name='conv2d_gabor_%d' % idx).set_weights(w)
        md.get_layer(name='conv2d_gabor_%d' % idx).trainable = False

    return md


########################################################################################################################
# </Functions>
########################################################################################################################
IDX_BIT_START = 0
IDX_BIT_END = 256

wavelength_Gabor = [4, 4 * np.sqrt(2), 8, 8 * np.sqrt(2), 16]
orientation_Gabor = [0, np.pi/4, np.pi/2, 3*np.pi/4]
phase_gabor = [0, np.pi/2]
ratio_Gabor = [1]
all_Gabor = get_Gabor_filters(wavelength_Gabor, orientation_Gabor, phase_gabor, ratio_Gabor)
idx_features = read_csv_inf_filters(args.idx_filters)
Gabor_filters = [all_Gabor[ix[0]-1][ix[1]-1] for ix in idx_features]


inf_features = read_csv_inf_filters(args.inf_filters)
inf_features = [inf_features[idx] for idx in range(IDX_BIT_START, IDX_BIT_END)]

dataset = read_csv(args.dataset, args.input_folder, False)
dataset = [[v[i] for i in [0] + list(range(1 + IDX_BIT_START, 1 + IDX_BIT_END))] for v in dataset]
dataset = replace_gts(dataset, 9999, 0)

learning_set, validation_set, learning_idx, validation_idx = split_set_casia(dataset)

tot_pos_l = np.sum(np.asarray([el[1:] for el in learning_set]) > 0, 0)
tot_neg_l = np.sum(np.asarray([el[1:] for el in learning_set]) < 0, 0)
PRIOR_POS_l = np.divide(tot_pos_l, tot_pos_l + tot_neg_l)
PRIOR_NEG_l = np.divide(tot_neg_l, tot_pos_l + tot_neg_l)

# tf.data.experimental.AutoShardPolicy.OFF

with open(os.path.join(date_time_folder, 'partition_sets.dat'), 'wb') as f:
    pickle.dump([learning_set, validation_set], f)
f.close()

sometimes = lambda aug: iaa.Sometimes(args.prior_augment, aug)
augmenter = iaa.SomeOf((0, None), [sometimes(iaa.GaussianBlur(sigma=(0.0, 3.0))),
                                   sometimes(iaa.ContrastNormalization((0.75, 1.5))),
                                   sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.2))], random_order=True)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: %d ' % strategy.num_replicas_in_sync)

with strategy.scope():
    model = create_cnn_2(dataset, inf_features, Gabor_filters)


train_full(model, learning_set, validation_set)

print('Done...')
