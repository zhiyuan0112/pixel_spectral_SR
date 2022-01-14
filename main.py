from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader

from trainer.engine import Engine
from trainer.option import basic_opt
from utils.helper import *
from utils.transforms import *


# ---------------- Setup Engine ---------------- #
opt = basic_opt()
print(opt)
engine = Engine(opt)


# ---------------- Train and Validate Data ---------------- #
print('==> Preparing data..')
data = loadmat('data/train.mat')
train_data = data['lsr'], data['gt']
print(train_data[0].shape, train_data[1].shape)

train_data = MyDataset(train_data)
print(len(train_data))

n_val = round(len(train_data)/10)
train_data, val_data = get_train_valid_dataset(train_data, n_val)        # (c,h,w)

train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=True,
                          num_workers=opt.threads, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False,
                          num_workers=opt.threads, pin_memory=True)


# ---------------- Test Data ---------------- #
data_test = loadmat('data/test.mat')
test_data = data_test['lsr'], data_test['gt']
test_data = MyDataset(test_data)
print(len(test_data))

test_loader = DataLoader(test_data, batch_size=1, shuffle=False,
                          num_workers=opt.threads, pin_memory=True)



def main():
    for i in range(opt.nEpochs):
        print('\n Epoch:', i)
        engine.train(train_loader, i)
        loss = engine.validate(val_loader, i)

        engine.scheduler.step(loss)
        display_learning_rate(engine.optimizer)
        if i % opt.ri == 0:
            engine.save_checkpoint(engine.net, engine.optimizer, loss, i)
            engine.test(test_loader, i)


if __name__ == '__main__':
    main()
