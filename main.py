from functools import partial

from matplotlib.image import imsave
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from torchnet.dataset import TransformDataset
from torchvision.transforms.transforms import Compose

from trainer.engine import Engine
from trainer.option import basic_opt
from utils._util import (Degrade, HSI2Tensor, ImageTransformDataset,
                         LMDBDataset, LoadMatKey, MatDataFromFolder)
from utils.helper import adjust_learning_rate, display_learning_rate
from utils.transforms import get_train_valid_dataset

# ---------------- Setup Engine ---------------- #
opt = basic_opt()
print(opt)
engine = Engine(opt)


# ---------------- Train and Validate Data ---------------- #
print('==> Preparing data..')
train_data = LMDBDataset(opt.dir)
train_data = TransformDataset(train_data, lambda x: x) 
print("Length of train and val set: {}.".format(len(train_data)))
print(train_data[0].shape)

n_val = round(len(train_data)/10)
train_data, val_data = get_train_valid_dataset(train_data, n_val)        # (c,h,w)


# HSI2Tensor = partial(HSI2Tensor, use_2dconv=True)

sr_degrade = Compose([
    lambda x: x.transpose(1, 2, 0),  # (h,w,c)
    Degrade(),
    lambda x: x.transpose(2, 0, 1),  # (c,h,w)
    HSI2Tensor()
])

ImageTransformDataset = partial(ImageTransformDataset, target_transform=HSI2Tensor())
train_dataset = ImageTransformDataset(train_data, sr_degrade)
val_dataset = ImageTransformDataset(val_data, sr_degrade)

print(train_data[0].shape, len(val_data))

train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True,
                          num_workers=opt.threads, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                          num_workers=opt.threads, pin_memory=True)


# ---------------- Test Data ---------------- #
'''
test_dir = '/media/exthdd/datasets/hsi/lzy_data/CAVE_22_10/test'
test_data = MatDataFromFolder(test_dir, size=None)  # (340,340,103) 
test_data = TransformDataset(test_data, LoadMatKey(key='gt_500'))

print("Length of test set: {}.".format(len(test_data)))

sr_degrade_test = Compose([
    Degrade(),
    lambda x: x.transpose(2, 0, 1),  # (c,h,w)
    HSI2Tensor()
])

ImageTransformDataset = partial(ImageTransformDataset, target_transform=Compose([
                                lambda x: x.transpose(2, 0, 1), HSI2Tensor()]))
test_dataset = ImageTransformDataset(test_data, sr_degrade_test)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                          num_workers=opt.threads, pin_memory=True)
'''

def main():
    adjust_learning_rate(engine.optimizer, opt.lr) 
    for i in range(opt.nEpochs):
        engine.train(train_loader)
        loss = engine.validate(val_loader)

        engine.scheduler.step(loss)
        display_learning_rate(engine.optimizer)
        if i % opt.ri == 0:
            engine.save_checkpoint(engine.net, engine.optimizer, loss)
            # engine.test(test_loader)


if __name__ == '__main__':
    main()
