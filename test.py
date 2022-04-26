from functools import partial
from os.path import join
import os
from models.cnn import CNN
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchnet.dataset import TransformDataset
from torchvision.transforms.transforms import Compose

from trainer.engine import Engine
from utils._util import (Degrade, HSI2Tensor, ImageTransformDataset,
                         LMDBDataset, LoadMatKey, MatDataFromFolder)
from utils.helper import get_summary_writer, plot_result, progress_bar


def set_option(description=''):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--prefix', '-p', type=str, default='cnn')
    parser.add_argument('--batchSize', '-b', type=int, default=1, help='Testing batch size.')
    parser.add_argument('--resumePath', '-rp', type=str, help='Checkpoint to use.')
    
    # Hardware specifications
    parser.add_argument('--threads', type=int, default=8, help='Number of threads for data loader.')
    parser.add_argument('--no-cuda', action='store_true', help='Disable cuda?')
    
    # Log specifications
    parser.add_argument('--no-log', action='store_true', help='Disable logger?')
    parser.add_argument('--no-save', action='store_true', help='Disable Saver?')
    
    opt = parser.parse_args()
    return opt


opt = set_option()
device = torch.device('cpu') if opt.no_cuda else torch.device('cuda')
ImageTransformDataset = partial(ImageTransformDataset, target_transform=Compose([
                                lambda x: x.transpose(2, 0, 1), HSI2Tensor()]))


def get_data(opt):
    test_dir = '/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/test'
    test_data = MatDataFromFolder(test_dir, size=None)  # (340,340,103) 
    test_data = TransformDataset(test_data, LoadMatKey(key='gt_500'))

    print("Length of test set: {}.".format(len(test_data)))

    sr_degrade_test = Compose([
        Degrade(),
        lambda x: x.transpose(2, 0, 1),  # (c,h,w)
        HSI2Tensor()
    ])

    test_dataset = ImageTransformDataset(test_data, sr_degrade_test)

    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False,
                          num_workers=opt.threads, pin_memory=True)
    return test_loader

def load_model(opt):
    net = CNN(in_channel=16, out_channel=500)
    net.to(device)
    
    print('Loading from checkpoint %s..' % opt.resumePath)
    assert os.path.isdir('logs/checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(opt.resumePath)
    epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net'])
    return net, epoch

def test(opt, net, data_loader, epoch):    
    net.eval()
    pbar = progress_bar(len(data_loader))
    avg_loss = 0
    criterion = nn.L1Loss()
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
        if not opt.no_log:
            if idx % 1 == 0:
                gt = target[:,:,0:500:25,0:500:25].cpu().detach().numpy()
                pred = output[:,:,0:500:25,0:500:25].cpu().detach().numpy()
                plot_result(gt.shape[0], gt, pred, opt.prefix, 'test', epoch, idx)
    pbar.close()
    

def main():
    print("==> Loading model")
    net, epoch = load_model(opt=opt)
    test_loader = get_data(opt=opt)
    
    test(opt, net, test_loader, epoch)


if __name__ == '__main__':
    main()
