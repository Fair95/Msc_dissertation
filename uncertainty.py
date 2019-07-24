from __future__ import print_function

import cv2
import numpy as np
import keras
import random
import os
from keras import backend as K
from data import load_valid_data
from constants import img_rows,img_cols,sample_steps


def _tau_inv(keep_prob, N, l2=0.005, lambda_=0.00001):
    tau = keep_prob * l2 / (2. * N * lambda_)
    return 1. / tau

def compute_uncertain(model, sample, sample_steps):
    X = np.zeros([1, img_rows, img_cols])
    
    for t in range(sample_steps):
        prediction = model.predict(sample, verbose=0).reshape([1, img_rows, img_cols])
        X = np.concatenate((X, prediction))
    
    X = np.delete(X, [0], 0)

    return np.var(X, axis=0)+_tau_inv(0.5,sample_steps)



def compute_uncertainty_map(model,X_test):
    total = len(X_test)
    maps = np.ndarray((total, 1, img_rows, img_cols))
    for i in range(total):
        uncertainty = compute_uncertain(model,X_test[i:i+1],sample_steps)
        uncertainty = np.array([uncertainty])
        maps[i] = uncertainty
    return maps




if __name__ == '__main__':
    compute_uncertainty_map(model,X_test)
