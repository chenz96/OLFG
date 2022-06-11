import numpy as np
import pickle
import scipy
from scipy import linalg
from sklearn.svm import SVR, SVC

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import scipy
import os

from sklearn import metrics
def evaluate_cls(y_truth, y_pred, prob):
    auc = metrics.roc_auc_score(y_truth, prob)

    acc = metrics.accuracy_score(y_truth, y_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_truth, y_pred).ravel()
    #print(tn, fp, fn, tp)
    sen = tp / (tp+fn)
    spe = tn / (tn+fp)
    return acc, sen, spe, auc


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)







