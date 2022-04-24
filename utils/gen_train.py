import os
import sys

import caffe
import cv2
import h5py
import lmdb
import numpy as np
from numpy.core.fromnumeric import reshape
from scipy.io import loadmat
from scipy.ndimage import zoom

sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from general import (Data2Volume, crop_center, data_augmentation,
                     minmax_normalize)


def create_lmdb_train(
        datadir, fns, name, matkey,
        crop_sizes, scales, ksizes, strides,
        load=h5py.File, augment=True,
        seed=2022):
    """
    Create Augmented Dataset
    """
    def preprocess(data):                                                           # h,w,c
        new_data = []
        # data = minmax_normalize(data)
        data = minmax_normalize(data.transpose((2, 0, 1)))                          # c,h,w
        # Visualize3D(data)
        if crop_sizes is not None:
            if data.shape[-1] > crop_sizes[0] and data.shape[-2] > crop_sizes[0]:
                data = crop_center(data, crop_sizes[0], crop_sizes[1])
        for i in range(len(scales)):
            temp = zoom(data, zoom=(1, scales[i], scales[i])) if scales[i] != 1 else data
            
            temp_ssr = np.zeros((500, temp.shape[1], temp.shape[2]))
            x = np.arange(0, 31, 1)
            spectrum = np.linspace(0, 30, 500)
            wid = temp.shape[1]
            hei = temp.shape[2]
            for m in range(wid):
                for n in range(hei):
                    temp_ssr[:,m,n] = np.interp(spectrum, x, np.squeeze(temp[:,m,n]))
            
            temp = temp_ssr
            print(temp.shape)
            temp = Data2Volume(temp, ksizes=ksizes, strides=list(strides[i]))
            
            new_data.append(temp)
        new_data = np.concatenate(new_data, axis=0)
        if augment:
            for i in range(new_data.shape[0]):
                new_data[i, ...] = data_augmentation(new_data[i, ...])

        return new_data.astype(np.float32)                                           # c,h,w

    np.random.seed(seed)
    scales = list(scales)
    ksizes = list(ksizes)
    assert len(scales) == len(strides)
    # calculate the shape of dataset
    if load == cv2.imread:
        data = load(datadir + fns[0], 0)
        data = np.expand_dims(data, 2)
    else:
        data = load(datadir + fns[0])[matkey]
    
    print('init:', data.shape)
    data = preprocess(data)
    N = data.shape[0]
    print('processed:', data.shape)

    # We need to prepare the database for the size. We'll set it 1.5 times
    # greater than what we theoretically need.
    map_size = data.nbytes * len(fns) * 2
    print('map size (GB):', map_size / 1024 / 1024 / 1024)

    if os.path.exists(name+'.db'):
        raise Exception('database already exist!')
    # env = lmdb.open(name+'.db', map_size=int(1e13), writemap=True)
    env = lmdb.open(name+'.db', map_size=map_size, writemap=True)
    with env.begin(write=True) as txn:
        # txn is a Transaction object
        k = 0
        for i, fn in enumerate(fns):
            try:
                if load == cv2.imread:
                    X = load(datadir + fn, 0)
                    X = np.expand_dims(X, 2)
                else:
                    X = load(datadir + fn)[matkey]
            except:
                print('loading', datadir+fn, 'fail')
                continue
            X = preprocess(X)
            N = X.shape[0]
            for j in range(N):
                datum = caffe.proto.caffe_pb2.Datum()
                datum.channels = X.shape[1]
                datum.height = X.shape[2]
                datum.width = X.shape[3]
                datum.data = X[j].tobytes()
                str_id = '{:08}'.format(k)
                k += 1
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
            print('load mat (%d/%d): %s' % (i, len(fns), fn))

        print('done')

# test: nachal_0823-1223.mat
def create_cave64_31():
    print('create cave64_31...')
    datadir = '/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/train/'
    fns = os.listdir(datadir)
    fns = [fn.split('.')[0]+'.mat' for fn in fns]

    create_lmdb_train(
        datadir, fns, '/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/CAVE64_31to500_22', 'gt',
        crop_sizes=None,
        scales=(1, 0.75, 0.5),
        ksizes=(500, 64, 64),
        strides=[(500, 64, 64), (500, 32, 32), (500, 32, 32)],
        load=loadmat, augment=True,
    )


if __name__ == '__main__':
    create_cave64_31()
