import os

import numpy as np
from scipy.io import loadmat, savemat
import random


def get_fn(path, key, idx):
    names = []
    curve_idxs = []
    fns = os.listdir(path)
    for i in idx:
        sum = 0
        for num, fn in enumerate(fns):
            temp = loadmat(os.path.join(path, fn))
            # print(num, fn, temp[key].shape)
            sum += temp[key].shape[0]
            if (i <= sum):
                names.append(fn)
                curve_idx = i - (sum - temp[key].shape[0]) + 1
                curve_idxs.append(curve_idx)
                break
    print(names, curve_idxs)


def group_data(path, key):
    fns = os.listdir(path)
    data = []
    for fn in fns:
        temp = loadmat(os.path.join(path, fn))
        print(fn, temp[key].shape)
        data.append(temp[key])

    data = np.concatenate(data, axis=0)
    print(data.shape)
    savemat('data/data_T.mat', {'gt': data})


def gen_data(path, key):
    data = loadmat(path)
    data = data[key]
    # data = data[:, 0:1000:10]
    print(data.shape)
    
    # ====== Generating Measurement Matrix ====== #
    random.seed(2022)
    rand16 = random.sample(range(1000), 16)
    print(rand16)
    mat_A = data[rand16, :]
    savemat('data/A.mat', {'A_100': mat_A[:, 0:1000:10], 'A_500':mat_A[:, 0:1000:2], 'A_1000':mat_A, 'channel': rand16})
    
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
    group_data(path, key)
    
    # ========= Step 2. Generating Measurement Matrix and Train/Test Data =========
    path = 'data/data_T.mat'
    key = 'gt'
    gen_data(path, key)
    
    
    # ========= Testing get_fn ========= #
    '''
    idx = [544, 295, 453, 558, 317, 599,  62, 530, 802, 708, 722, 422, 701, 649, 320, 830]
    get_fn(path, key, idx)
    a = loadmat('/home/liangzhiyuan/Code/spectralSR/data/A.mat')
    a = a['A']
    print(a.shape)
    a1 = a[1, :]
    print(a1)
    
    data_t = loadmat('/home/liangzhiyuan/Code/spectralSR/data/T/T_633.mat')
    data_t = data_t['T_R']
    data1 = data_t[34, 0:1000:10]
    print(data1.shape, a1.shape)
    print(data1)
    '''
    
