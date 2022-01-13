import os
from datetime import datetime
from os.path import join
import socket

import numpy as np
import torch
from scipy.io import loadmat, savemat
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchnet.dataset import SplitDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import CNN


class MyDataset(Dataset):
    def __init__(self, dataset):
        super(MyDataset, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset[0])

    def __getitem__(self, idx):
        img = self.dataset[0]
        target = self.dataset[1]
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=-1)
        target = np.expand_dims(target, axis=-1)
        target = np.expand_dims(target, axis=-1)
        img = img[idx, ...]
        target = target[idx, ...]
        return img, target

def get_train_valid_dataset(dataset, valid_size=None):                           
    error_msg = "[!] valid_size should be an integer in the range [1, %d]." %(len(dataset))
    if not valid_size:
        valid_size = int(0.1 * len(dataset))
    if not isinstance(valid_size, int) or valid_size < 1 or valid_size > len(dataset):
        raise TypeError(error_msg)

    # generate train/val datasets
    partitions = {'Train': len(dataset)-valid_size, 'Valid':valid_size}

    train_dataset = SplitDataset(dataset, partitions, initial_partition='Train')
    valid_dataset = SplitDataset(dataset, partitions, initial_partition='Valid')

    return (train_dataset, valid_dataset)

def get_summary_writer(arch, prefix):
    log_dir = 'logs/runs/%s/%s/' % (arch, prefix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname())
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

basedir = 'logs/checkpoint'
def save_checkpoint(net, optimizer, loss, epoch, model_out_path=None):
    if not os.path.isdir(join(basedir)):
        os.makedirs(join(basedir))
    if not model_out_path:
        model_out_path = join(basedir, "model_epoch_%d.pth" % (epoch))
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'epoch': epoch,
    }

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


# ---------------- Train and Validate Data ---------------- #
print('==> Preparing data..')
data = loadmat('data/train.mat')
train_data = data['lsr'], data['gt']
print(train_data[0].shape, train_data[1].shape)

train_data = MyDataset(train_data)
print(len(train_data))

n_val = round(len(train_data)/10)
train_data, val_data = get_train_valid_dataset(train_data, n_val)        # (c,h,w)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True,
                          num_workers=8, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                          num_workers=8, pin_memory=True)


# ---------------- Test Data ---------------- #
data_test = loadmat('data/test.mat')
test_data = data_test['lsr'], data_test['gt']
test_data = MyDataset(test_data)
print(len(test_data))

test_loader = DataLoader(test_data, batch_size=1, shuffle=False,
                          num_workers=8, pin_memory=True)


n_epoch = 50000
lr = 1e-3
cuda = True
device = torch.device('cuda') if cuda else torch.device('cpu')
prefix = 'cnn_18_256_L1smooth'

net = CNN(in_channel=16, out_channel=100).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(
    optimizer, 'min', factor=0.1, patience=100, min_lr=5e-5, verbose=True)
writer = get_summary_writer(prefix, 'ssr')


def progress_bar(total):
    return tqdm(total=total, dynamic_ncols=True)


def train(data_loader, epoch):
    net.train()
    pbar = progress_bar(len(data_loader))
    avg_loss = 0
    for idx, (input, target) in enumerate(data_loader):
        input, target = input.to(device), target.to(device)
        input = input.float()
        target = target.float()
        
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target) + criterion(output[:,1:,:,:]-output[:,:-1,:,:], target[:,1:,:,:]-target[:,:-1,:,:])
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.detach().cpu().item()
        avg_loss /= (idx+1)
        
        pbar.set_postfix({'Loss': avg_loss})
        pbar.update()
    pbar.close()
    writer.add_scalar(join(prefix, 'train_loss_epoch'), avg_loss, epoch)    
    
    if i % 50 == 0:
        target = target[0,:,0,0].cpu().detach().numpy()
        output = output[0,:,0,0].cpu().detach().numpy()
        x = np.arange(0, 100, 1)
        plt.plot(x, target, color='r', label='GT')
        plt.plot(x, output, color='b', label='Pred')
        plt.legend()
        save_dir = 'logs/figs/' + prefix + '/train'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(join(save_dir, (str(i)+'.png')))
        plt.close()

best_loss = 1e6
def validate(data_loader, epoch):
    global best_loss
    net.eval()
    pbar = progress_bar(len(data_loader))
    avg_loss = 0
    for idx, (input, target) in enumerate(data_loader):
        input, target = input.to(device), target.to(device)
        input = input.float()
        target = target.float()
        
        with torch.no_grad():
            output = net(input)
        loss = criterion(output, target) + criterion(output[:,1:,:,:]-output[:,:-1,:,:], target[:,1:,:,:]-target[:,:-1,:,:])
        
        avg_loss += loss.detach().cpu().item()
        avg_loss /= (idx+1)
        
        pbar.set_postfix({'Loss': loss.detach().cpu().item()})
        pbar.update()
    pbar.close()
    writer.add_scalar(join(prefix, 'val_loss_epoch'), avg_loss, epoch)
    
    if avg_loss < best_loss:
            print('Best Result Saving...')
            model_best_path = join(basedir, 'model_best.pth')
            save_checkpoint(net, optimizer, avg_loss, epoch, model_out_path=model_best_path)
            best_loss = avg_loss
    
    if i % 50 == 0:
        target = target[0,:,0,0].cpu().detach().numpy()
        output = output[0,:,0,0].cpu().detach().numpy()
        x = np.arange(0, 100, 1)
        plt.plot(x, target, color='r', label='GT')
        plt.plot(x, output, color='b', label='Pred')
        plt.legend()
        save_dir = 'logs/figs/' + prefix + '/val'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(join(save_dir, (str(i)+'.png')))
        plt.close()
        
    return loss

def test(data_loader, epoch):
    net.eval()
    pbar = progress_bar(len(data_loader))
    avg_loss = 0
    for idx, (input, target) in enumerate(data_loader):
        input, target = input.to(device), target.to(device)
        input = input.float()
        target = target.float()
        
        with torch.no_grad():
            output = net(input)
        loss = criterion(output, target)
        
        avg_loss += loss.detach().cpu().item()
        avg_loss /= (idx+1)
        
        pbar.set_postfix({'Loss': loss.detach().cpu().item()})
        pbar.update()
    pbar.close()
    writer.add_scalar(join(prefix, 'test_loss_epoch'), avg_loss, epoch)
    
    target = target[0,:,0,0].cpu().detach().numpy()
    output = output[0,:,0,0].cpu().detach().numpy()
    x = np.arange(0, 100, 1)
    plt.plot(x, target, color='r', label='GT')
    plt.plot(x, output, color='b', label='Pred')
    plt.legend()
    save_dir = 'logs/figs/' + prefix + '/test'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(join(save_dir, (str(i)+'.png')))
    plt.close()



if __name__ == '__main__':
    for i in range(n_epoch):
        print('Epoch:', i)
        train(train_loader, i)
        loss = validate(val_loader, i)
        
        if i % 50 == 0:
            test(test_loader, i)
        
        scheduler.step(loss)
        display_learning_rate(optimizer)
        if i % 50 == 0:
            save_checkpoint(net, optimizer, loss, i)
