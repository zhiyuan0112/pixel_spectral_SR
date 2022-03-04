from scipy.io import loadmat, savemat
from scipy.ndimage import zoom
import numpy as np
import os

if __name__ == '__main__':
    test_dir = '/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/train'
    fns = os.listdir(test_dir)
    spectrum = np.linspace(0, 30, 100)
    x = np.arange(0, 31, 1) 
    data_ssr = np.zeros((512, 512, 100))
    
    for fn in fns:
        data = loadmat(os.path.join(test_dir, fn))['gt']
        for i in range(512):
            for j in range(512):
                data_ssr[i,j,:] = np.interp(spectrum, x, np.squeeze(data[i,j,:]))
        savemat('/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/train_100/'+fn, 
                {'gt':data, 'gt_100':data_ssr})
        print(data_ssr.shape)