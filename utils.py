import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from glob import glob
import datetime


def read_data(file, normalization='z-score', seed=42):
    if file.endswith('.npz'):
        data = np.load(file, allow_pickle=True)
        x, y = data['X'], data['y']
        y = np.array(y, dtype=int)
    else:
        if file.endswith('pkl'):
            func = pd.read_pickle
        elif file.endswith('csv'):
            func = pd.read_csv
        else:
            raise NotImplementedError('')

        df = func(file)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        x = df.values[:, :-1]
        y = np.array(df.values[:, -1], dtype=int)

    # train-test splitting
    rng = np.random.RandomState(seed)
    idx = rng.permutation(np.arange(len(x)))
    x, y = x[idx], y[idx]

    norm_idx = np.where(y==0)[0]
    anom_idx = np.where(y==1)[0]
    split = int(0.5 * len(norm_idx))
    train_norm_idx, test_norm_idx = norm_idx[:split], norm_idx[split:]

    x_train = x[train_norm_idx]
    y_train = y[train_norm_idx]

    x_test = x[np.hstack([test_norm_idx, anom_idx])]
    y_test = y[np.hstack([test_norm_idx, anom_idx])]

    print(f'Original size: [{x.shape}], Normal/Anomaly: [{len(norm_idx)}/{len(anom_idx)}] \n'
          f'After splitting: training/testing [{len(x_train)}/{len(x_test)}]')

    # normalization
    if normalization == 'min-max':
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(x_train)
        x_train = minmax_scaler.transform(x_train)
        x_test = minmax_scaler.transform(x_test)

    elif normalization == 'z-score':
        mus = np.mean(x_train, axis=0)
        sds = np.std(x_train, axis=0)
        sds[sds == 0] = 1
        x_train = np.array([(xx - mus) / sds for xx in x_train])
        x_test = np.array([(xx - mus) / sds for xx in x_test])

    elif normalization == 'scale':
        x_train = x_train / 255
        x_test = x_test / 255

    return x_train, y_train, x_test, y_test


def min_max_normalize(x):
    filter_lst = []
    for k in range(x.shape[1]):
        s = np.unique(x[:, k])
        if len(s) <= 1:
            filter_lst.append(k)
    if len(filter_lst) > 0:
        print('remove features', filter_lst)
        x = np.delete(x, filter_lst, 1)

    scaler = MinMaxScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    return x


def evaluate(y_true, scores):
    """calculate evaluation metrics"""
    roc_auc = metrics.roc_auc_score(y_true, scores)
    ap = metrics.average_precision_score(y_true, scores)

    # F1@k, using real percentage to calculate F1-score
    ratio =  100.0 * len(np.where(y_true==0)[0]) / len(y_true)
    thresh = np.percentile(scores, ratio)
    y_pred = (scores >= thresh).astype(int)
    y_true = y_true.astype(int)
    precision, recall, f_score, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return roc_auc, ap, f_score


def get_data_lst(dataset_dir, dataset):
    if dataset == 'FULL':
        print(os.path.join(dataset_dir, '*.*'))
        data_lst = glob(os.path.join(dataset_dir, '*.*'))
    else:
        name_lst = dataset.split(',')
        data_lst = []
        for d in name_lst:
            data_lst.extend(glob(os.path.join(dataset_dir, d + '.*')))
    data_lst = sorted(data_lst)
    if 'fmnist' in dataset_dir:
        data_lst = data_lst[::-1]
    return data_lst


def adjust_contamination(x_train, y_train, x_test, y_test,
                         contamination_r, swap_ratio=0.05, random_state=42):
    """
    used only for 50%normal-setting
    add/remove anomalies in training data to replicate anomaly contaminated data sets.
    randomly swap 5% features of two anomalies to avoid duplicate contaminated anomalies.
    """
    rng = np.random.RandomState(random_state)

    test_anomalies = x_test[np.where(y_test == 1)[0]]
    test_inliers = x_test[np.where(y_test == 0)[0]]

    a = np.arange(len(test_anomalies))
    rng.shuffle(a)
    test_anomalies = test_anomalies[a]

    anomalies = test_anomalies[:int(0.5 * len(test_anomalies))]
    rest_anomalies = test_anomalies[int(0.5 * len(test_anomalies)):]
    x_test_new = np.vstack([test_inliers, rest_anomalies])
    y_test_new = np.hstack([np.zeros(len(test_inliers)), np.ones(len(rest_anomalies))])

    # anomalies = test_anomalies
    # # anomalies = test_anomalies[:int(0.5 * len(test_anomalies))]
    # x_test_new = x_test
    # y_test_new = y_test

    # else:
    #     anomalies = x_train[np.where(y_train==1)[0]]
    #     x_test_new = x_test
    #     y_test_new = y_test

    n_add_anom = int(len(x_train) * contamination_r / (1. - contamination_r))
    n_inj_noise = n_add_anom - len(anomalies)
    print(f'Control Contamination Rate: \n'
          f'Contain  : [{n_add_anom}] Anomalies, '
          f'injecting: [{n_inj_noise}] Noisy samples, \n'
          f'testing  : {len(np.where(y_test_new==1)[0])}/{len(np.where(y_test_new==0)[0])}')

    # use all anomalies and inject new anomalies
    if n_inj_noise > 0:
        n_sample, dim = anomalies.shape
        n_swap_feat = int(swap_ratio * dim)
        inj_noise = np.empty((n_inj_noise, dim))
        for i in np.arange(n_inj_noise):
            idx = rng.choice(n_sample, 2, replace=False)
            o1 = anomalies[idx[0]]
            o2 = anomalies[idx[1]]
            swap_feats = rng.choice(dim, n_swap_feat, replace=False)
            inj_noise[i] = o1.copy()
            inj_noise[i][swap_feats] = o2[swap_feats]

        x = np.vstack([x_train, anomalies])
        y = np.hstack([y_train, np.ones(n_add_anom)])
        x = np.vstack([x, inj_noise])
        y = np.hstack([y, np.ones(n_inj_noise)])

    # use original anomalies
    else:
        n_sample, dim = anomalies.shape
        idx = rng.choice(n_sample, n_add_anom, replace=False)
        x = np.append(x_train, anomalies[idx], axis=0)
        y = np.append(y_train, np.ones(n_add_anom))
        print(x.shape)

    return x, y, x_test_new, y_test_new


