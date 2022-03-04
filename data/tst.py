import os
from os.path import join

from matplotlib.image import imsave
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat


def plot_result_traditional(target_dir, output_dir, key):
    x = np.arange(0, 100, 1)
    fns = os.listdir(output_dir)
    data = loadmat(os.path.join(target_dir, fns[0]))
    data = data['gt']
    # # Remove the left pixels with peak
    # data = data[10:512, 10:512, :]
    # print(data.shape)
    
    for idx, fn in enumerate(fns):
        # get output
        data = loadmat(os.path.join(output_dir, fn))
        data = data[key]
        data = data[10:512, 10:512, :]  # Remove the left pixels with peak
        data = data[0:500:125, 0:500:125, :]
        
        # get gt
        gt = loadmat(os.path.join(target_dir, fn))
        gt = gt['gt_100']
        gt = gt[10:512, 10:512, :]  # Remove the left pixels with peak
        gt = gt[0:500:125, 0:500:125, :]
        
        cnt = 1
        for i in range(4):
            for j in range(4):
                plt.subplot(4,4,cnt)
                cnt += 1
                plt.plot(x, gt[i,j,:], color='r', label='GT', linewidth=0.7)
                plt.plot(x, data[i,j,:], color='b', label='Pred', linewidth=0.7)
        plt.legend()
        
        # if stage == 'test':
        save_dir = 'logs/figs/traditional'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(join(save_dir, (str(idx)+'.png')))
        print(str(idx), 'done')
        plt.clf()



if __name__ == '__main__':
    target_dir = '/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/test'
    output_dir = '/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/result/traditional'
    key = 'estimated'
    plot_result_traditional(target_dir, output_dir, key)