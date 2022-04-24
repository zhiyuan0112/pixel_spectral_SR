from scipy.io import loadmat, savemat
from scipy.ndimage import zoom
import numpy as np
import os


if __name__ == '__main__':
    test_dir = '/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/test'
    fns = os.listdir(test_dir)
    x = np.arange(0, 31, 1)
    spectrum_100 = np.linspace(0, 30, 100)
    data_ssr_100 = np.zeros((512, 512, 100))
    spectrum_500 = np.linspace(0, 30, 500)
    data_ssr_500 = np.zeros((512, 512, 500))

    for fn in fns:
        data = loadmat(os.path.join(test_dir, fn))['gt']
        for i in range(512):
            for j in range(512):
                data_ssr_100[i, j, :] = np.interp(
                    spectrum_100, x, np.squeeze(data[i, j, :]))
                data_ssr_500[i, j, :] = np.interp(
                    spectrum_500, x, np.squeeze(data[i, j, :]))
        savemat('/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/test/'+fn,
                {'gt': data, 'gt_100': data_ssr_100, 'gt_500': data_ssr_500})
        print(data_ssr_100.shape, data_ssr_500.shape)
