#!/usr/bin/python
# -*- coding:utf8 -*-
from scipy.io import loadmat,savemat
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score


def load_data_kfold(k,datatype,path,random_state=111):
    data = loadmat(path)
    if datatype == 'sfc':
        X = data.get('data')
    elif datatype == 'tc':
        X = data.get('tc_data_170')
    y = data.get('label').T
    folds = list(StratifiedShuffleSplit(n_splits=k,random_state=random_state).split(X, y))
    return folds, X, y

from sklearn.metrics import auc
from sklearn import metrics
def acc_pre_recall_f(y_true,y_pred,y_score):
    tp = float(sum(y_true & y_pred))
    fp = float(sum((y_true == 0) & (y_pred == 1)))
    tn = float(sum((y_true == 0) & (y_pred == 0)))
    fn = float(sum((y_true == 1) & (y_pred == 0)))
    acc = accuracy_score(y_true,y_pred)
    sensitivity = tp/(tp + fn)
    f1 = f1_score(y_true,y_pred)
    specificity = tn / (fp + tn)
    fpr, tpr, thresholds = metrics.roc_curve(y_true,y_score)
    roc_auc = auc(fpr, tpr)
    return acc,specificity,sensitivity,f1,roc_auc