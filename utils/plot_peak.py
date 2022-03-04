import os

import numpy as np
from matplotlib.image import imsave
from scipy.io import loadmat, savemat
from torch import threshold


test_dir = '/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/test'
fns = os.listdir(test_dir)
print(fns)

height, width, channel = loadmat(os.path.join(test_dir, fns[0]))['gt'].shape
print(channel)

for fn in fns:
    data = loadmat(os.path.join(test_dir, fn))
    data = data['gt']
    for i in range(height):
        for j in range(width):
            # print(max(data[i,j,:]), min(data[i,j,:]))
            _threshold = (max(data[i, j, :]) - min(data[i, j, :]))/2
            for c in range(1, channel-1):
                if (abs(data[i, j, c+1] - data[i, j, c]) >= _threshold and abs(data[i, j, c] - data[i, j, c-1]) >= _threshold):
                    data[i, j, 0:3] = [1, 0, 0]
                    break

    imsave('test_'+fn+'.png', data[:, :, 0:3])
    print(fn)
