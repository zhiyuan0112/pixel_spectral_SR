import os
import socket
from datetime import datetime
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm


def get_summary_writer(prefix):
    log_dir = 'logs/runs/%s/' % (prefix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, datetime.now().strftime(
        '%b%d_%H-%M-%S')+'_'+socket.gethostname())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    return writer


def display_learning_rate(optimizer):
    lrs = []
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        print('learning rate of group %d: %.4e' % (i, lr))
        lrs.append(lr)
    return lrs


def progress_bar(total):
    return tqdm(total=total, dynamic_ncols=True)


def plot_result(n_fig, targets, outputs, prefix, stage, epoch, iter):
    x = np.arange(0, 100, 1)
    
    if n_fig == 1:
        plt.plot(x, np.squeeze(targets), color='r', label='GT')
        plt.plot(x, np.squeeze(outputs), color='b', label='Pred')
        plt.legend()
        # if stage == 'test':
        save_dir = 'logs/figs/' + prefix + '/' + stage + '/' + str(epoch)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(join(save_dir, (str(iter)+'.png')))
        # else:
        #     save_dir = 'logs/figs/' + prefix + '/' + stage
        #     os.makedirs(save_dir, exist_ok=True)
        #     plt.savefig(join(save_dir, (str(epoch)+'.png')))
    else:
        fig, axs = plt.subplots(n_fig, sharex=True)
        for i, (target, output) in enumerate(zip(np.split(targets, n_fig), np.split(outputs, n_fig))):
            axs[i].plot(x, np.squeeze(target), color='r', label='GT')
            axs[i].plot(x, np.squeeze(output), color='b', label='Pred')
        plt.legend()
        save_dir = 'logs/figs/' + prefix + '/' + stage
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(join(save_dir, (str(epoch)+'.png')))
    plt.close()
