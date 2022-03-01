import os
import threading

import caffe
import cv2
import numpy as np
import torch
import torch.utils.data as data
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchnet.dataset import SplitDataset
from torchvision.transforms import Compose

# from .degrade import GaussianDownsample
# from .noise import GaussianNoiseBlindv2


class HSI2Tensor(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """
    def __init__(self, use_2dconv=True):
        self.use_2dconv = use_2dconv

    def __call__(self, hsi):
        if self.use_2dconv:
            img = torch.from_numpy(hsi)
        else:
            temp = hsi[None]/1.0
            img = torch.from_numpy(temp)
        return img.float()


class Degrade(object):
    def __init__(self):
        self.A = loadmat('/home/liangzhiyuan/Code/spectralSR/data/A.mat')['A']

    def __call__(self, img):
        img = np.einsum('ij,klj->kli', self.A, img)
        return img


class ImageTransformDataset(Dataset):
    def __init__(self, dataset, transform, target_transform=None):
        super(ImageTransformDataset, self).__init__()

        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        target = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class MatDataFromFolder(Dataset):
    """Wrap mat data from folder"""
    def __init__(self, data_dir, load=loadmat, suffix='mat', size=None,
                 share_dict=None, use_cache=False):
        super(MatDataFromFolder, self).__init__()
        self.filenames = [
            os.path.join(data_dir, fn) 
            for fn in os.listdir(data_dir)
            if fn.endswith(suffix)
        ]
        # self.filenames = [os.path.join(data_dir, 'nachal_0823-1223.mat')]
        # self.filenames = [os.path.join(data_dir, 'fake_and_real_food_ms.mat')]
        self.load = load

        if size and size <= len(self.filenames):
            self.filenames = self.filenames[:size]
        self.cache = share_dict
        self.use_cache = use_cache

    def get_mat(self, filename):
        if self.use_cache:
            if filename not in self.cache.keys():
                self.cache[filename] = self.load(filename)
            return self.cache[filename]
        else:
            return self.load(filename)

    def __getitem__(self, index):
        mat = self.get_mat(self.filenames[index])
        return mat

    def __len__(self):
        return len(self.filenames)


class LMDBDataset(data.Dataset):
    def __init__(self, db_path):
        import lmdb
        self.db_path = db_path
        self.env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            raw_datum = txn.get('{:08}'.format(index).encode('ascii'))

        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)

        flat_x = np.fromstring(datum.data, dtype=np.float32)
        x = flat_x.reshape(datum.channels, datum.height, datum.width)

        return x

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class LoadMatKey(object):
    def __init__(self, key):
        self.key = key
    
    def __call__(self, mat):
        # item = mat[self.key].transpose((2,0,1))
        item = mat[self.key]
        item = item.astype(np.float32)
        return item


class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self): return self

    def __next__(self):
        self.lock.acquire()
        try:
            return next(self.it)
        finally:
            self.lock.release()
