import os

import numpy as np
from scipy.io import loadmat, savemat

if __name__ == '__main__':
    fns = ['T_522.mat', 'T_633.mat', 'T_253.mat', 'T_522.mat', 'T_263.mat', 'T_632.mat', 'T_333.mat', 'T_523.mat', 'T_213.mat', 'T_652.mat', 'T_652.mat', 'T_643.mat', 'T_652.mat', 'T_223.mat', 'T_263.mat', 'T_313.mat']
    idx = [4, 28, 3, 13, 13, 9, 7, 13, 5, 14, 28, 28, 7, 7, 13, 4]
    for i in range(len(idx)):
        idx[i] -= 1
    print(idx)
    
    key = 'T_R'
    data_A = np.zeros((16,1000))
    for n, (fn, i) in enumerate(zip(fns, idx)):
        print(i)
        path = os.path.join('data/T', fn)
        data = loadmat(path)[key]
        data_A[n] = data[i]
        print(fn, data.shape)
    print(data_A.shape)
    
    savemat('data/A_20220424.mat', {'A_100': data_A[:, 0:1000:10], 'A_500':data_A[:, 0:1000:2], 'A_1000':data_A, 'channel': idx})