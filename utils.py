import numpy as np
import pickle
import scipy
from scipy import linalg
import os

from sklearn import metrics
def evaluate_cls(y_truth, y_pred, prob):
    auc = metrics.roc_auc_score(y_truth, prob)

    acc = metrics.accuracy_score(y_truth, y_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_truth, y_pred).ravel()

    sen = tp / (tp+fn)
    spe = tn / (tn+fp)
    return acc, sen, spe, auc








