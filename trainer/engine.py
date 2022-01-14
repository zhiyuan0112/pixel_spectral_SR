import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
from models.cnn import CNN
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.helper import get_summary_writer, progress_bar, plot_result


class Engine():
    def __init__(self, opt) -> None:
        self.opt = opt
        self.device = torch.device('cpu') if self.opt.no_cuda else torch.device('cuda')
        self._setup()
        
    def _setup(self):
        self.epoch = 0
        self.iteration = 0
        self.best_loss = 1e6
        self.basedir = join('logs/checkpoint', self.opt.prefix)
        os.makedirs(self.basedir, exist_ok=True)
        
        #-------------------- Define Logger --------------------#
        self.log = not self.opt.no_log
        if self.log:
            self.writer = get_summary_writer(self.opt.prefix)
        
        #-------------------- Define CUDA --------------------#
        print('==> Cuda Acess: {}'.format(self.device))
        
        #-------------------- Define Model --------------------#
        print("==> Creating model")
        self.net = CNN(in_channel=16, out_channel=100)
        self.net.to(self.device)
        
        #-------------------- Define Loss Function --------------------#
        self.criterion = nn.L1Loss()

        #-------------------- Define Optimizer --------------------#
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.opt.lr, weight_decay=0.1)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5,
                                           patience=10, min_lr=self.opt.min_lr, verbose=True)

        #-------------------- Resume Previous Model --------------------#
        if self.opt.resume:
            self.load_checkpoint(self.opt.resumePath)
        else:
            print("==> Building model..")
            
    def load_checkpoint(self, resumePath):
        model_best_path = join(self.basedir, 'model_best.pth')
        if os.path.exists(model_best_path):
            best_model = torch.load(model_best_path)
            self.best_psnr = best_model['psnr']
            self.best_loss = best_model['loss']

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('logs/checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path)
        self.epoch = checkpoint['epoch']
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def save_checkpoint(self, net, optimizer, loss, epoch, model_out_path=None):
        if not os.path.isdir(join(self.basedir)):
            os.makedirs(join(self.basedir))
        if not model_out_path:
            model_out_path = join(self.basedir, "model_epoch_%d.pth" % (epoch))
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss,
            'epoch': epoch,
        }

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
        
    def train(self, data_loader, epoch):
        self.net.train()
        pbar = progress_bar(len(data_loader))
        avg_loss = 0
        for idx, (input, target) in enumerate(data_loader):
            input, target = input.to(self.device), target.to(self.device)
            input = input.float()
            target = target.float()
            
            self.optimizer.zero_grad()
            output = self.net(input)
            loss = self.criterion(output, target)
            # loss = self.criterion(output, target) + self.criterion(output[:,1:,:,:]-output[:,:-1,:,:], target[:,1:,:,:]-target[:,:-1,:,:])
            loss.backward()
            self.optimizer.step()
            
            avg_loss += loss.detach().cpu().item()
            avg_loss /= (idx+1)
            
            pbar.set_postfix({'Loss': avg_loss})
            pbar.update()
        pbar.close()
        
        if self.log:
            self.writer.add_scalar(join(self.opt.prefix, 'train_loss_epoch'), avg_loss, epoch)    
            if epoch % self.opt.ri == 0:
                target = target[:,:,0,0].cpu().detach().numpy()
                output = output[:,:,0,0].cpu().detach().numpy()
                plot_result(target.shape[0], target, output, self.opt.prefix, 'train', epoch, None)

    def validate(self, data_loader, epoch):
        self.net.eval()
        pbar = progress_bar(len(data_loader))
        avg_loss = 0
        for idx, (input, target) in enumerate(data_loader):
            input, target = input.to(self.device), target.to(self.device)
            input = input.float()
            target = target.float()
            
            with torch.no_grad():
                output = self.net(input)
            loss = self.criterion(output, target)
            # loss = self.criterion(output, target) + self.criterion(output[:,1:,:,:]-output[:,:-1,:,:], target[:,1:,:,:]-target[:,:-1,:,:])
            
            avg_loss += loss.detach().cpu().item()
            avg_loss /= (idx+1)
            
            pbar.set_postfix({'Loss': loss.detach().cpu().item()})
            pbar.update()
            if self.log:
                if epoch % self.opt.ri == 0 and idx % 10 == 0:
                    gt = target[:,:,0,0].cpu().detach().numpy()
                    pred = output[:,:,0,0].cpu().detach().numpy()
                    plot_result(gt.shape[0], gt, pred, self.opt.prefix, 'val', epoch, idx)
        pbar.close()
        
        
        if avg_loss < self.best_loss:
            print('Best Result Saving...')
            model_best_path = join(self.basedir, 'model_best.pth')
            self.save_checkpoint(self.net, self.optimizer, avg_loss, epoch, model_out_path=model_best_path)
            self.best_loss = avg_loss
        
        if self.log:
            self.writer.add_scalar(join(self.opt.prefix, 'val_loss_epoch'), avg_loss, epoch)
            
        return loss

    def test(self, data_loader, epoch):
        self.net.eval()
        pbar = progress_bar(len(data_loader))
        avg_loss = 0
        for idx, (input, target) in enumerate(data_loader):
            input, target = input.to(self.device), target.to(self.device)
            input = input.float()
            target = target.float()
            
            with torch.no_grad():
                output = self.net(input)
            loss = self.criterion(output, target)
            
            avg_loss += loss.detach().cpu().item()
            avg_loss /= (idx+1)
            
            pbar.set_postfix({'Loss': loss.detach().cpu().item()})
            pbar.update()
            if self.log:
                if idx % 10 == 0:
                    gt = target[:,:,0,0].cpu().detach().numpy()
                    pred = output[:,:,0,0].cpu().detach().numpy()
                    plot_result(gt.shape[0], gt, pred, self.opt.prefix, 'test', epoch, idx)
        pbar.close()
        
        if self.log:
            self.writer.add_scalar(join(self.opt.prefix, 'test_loss_epoch'), avg_loss, epoch)
            
