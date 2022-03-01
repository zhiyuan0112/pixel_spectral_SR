import os

import numpy as np
from scipy.io import loadmat, savemat
import random


def group_data(path, key):
    fns = os.listdir(path)
    data = []
    for fn in fns:
        temp = loadmat(os.path.join(path, fn))
        if key in temp.keys():
            print(fn, temp[key].shape)
            data.append(temp[key])

    data = np.concatenate(data, axis=0)
    print(data.shape)
    savemat('data/data_T.mat', {'gt': data})


def gen_data(path, key):
    data = loadmat(path)
    data = data[key]
    data = data[:, 0:1000:10]
    print(data.shape)
    
    # ====== Generating Measurement Matrix ====== #
    random.seed(2022)
    rand16 = random.sample(range(1000), 16)
    mat_A = data[rand16, :]
    savemat('data/A.mat', {'A': mat_A, 'channel': rand16})
    
    # ====== Generating Train/Test Data ====== #
    difference = list(set(range(data.shape[0])).difference(set(rand16)))  # cal difference set
    data = data[difference, :]
    print(data.shape, data.shape[0])
    
    y = []
    for d in np.split(data, data.shape[0], axis=0):
        temp = np.einsum('ij,kj->ki', mat_A, d)
        y.append(temp)
    y = np.concatenate(y, axis=0)
    print(y.shape)
    
    test_idx = random.sample(range(data.shape[0]), 150)    
    train_idx = list(set(range(data.shape[0])).difference(set(test_idx)))

    savemat('data/test.mat', {'gt': data[test_idx, :], 'lsr': y[test_idx, :], 'channel': test_idx})
    savemat('data/train.mat', {'gt': data[train_idx, :], 'lsr': y[train_idx, :], 'channel': train_idx})
    
    

if __name__ == '__main__':
    # ========= Step 1. Groupping All Spectrum =========
    path = 'data/T'
    key = 'T_R'
    # group_data(path, key)
    
    # ========= Step 2. Generating Measurement Matrix and Train/Test Data =========
    path = 'data/data_T.mat'
    key = 'gt'
    gen_data(path, key)
    
    test_fns = ['balloons_ms.mat']
    
    
