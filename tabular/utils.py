import itertools
import os
import random
import statistics
from datetime import datetime
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import ref
import tqdm
# from autofe.optuna_tuner.registry import MULTICLASS_CLASSIFICATION
# from autofe.optuna_tuner.rf_optuna import RandomForestOptuna
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def split_train_test(objects, test_percent=0.15, random_state=2021):
    length = len(objects)
    idx = np.arange(length)
    n = max(1, int(length * test_percent))
    random.seed(random_state)
    random.shuffle(idx)
    return {'train': idx[:-n], 'test': idx[-n:]}


def wavelets_transform(data, wavelet, level=1):
    for i in range(level):
        data, _ = pywt.dwt(data, wavelet)
    return data


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,
                 i,
                 format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        print('%s has been created, will use the  origin dir' % (dir))


def write_log(logfile, content):
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logfile.write('{}\t{}\n'.format(t, content))
    print(content)
