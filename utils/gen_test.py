from scipy.io import loadmat, savemat
from scipy.ndimage import zoom
import os

if __name__ == '__main__':
    test_dir = '/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/test'
    fns = os.listdir(test_dir)
    
    for fn in fns:
        data = loadmat(os.path.join(test_dir, fn))['gt']
        data_ssr = zoom(data, zoom=(1, 1, 100/31), mode='nearest')
        savemat('/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/test/'+fn, 
                {'gt':data, 'gt_100':data_ssr})
        print(data_ssr.shape)