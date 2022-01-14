import numpy as np
from torch.utils.data.dataset import Dataset
from torchnet.dataset import SplitDataset


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
    error_msg = "[!] valid_size should be an integer in the range [1, %d]." % (
        len(dataset))
    if not valid_size:
        valid_size = int(0.1 * len(dataset))
    if not isinstance(valid_size, int) or valid_size < 1 or valid_size > len(dataset):
        raise TypeError(error_msg)

    # generate train/val datasets
    partitions = {'Train': len(dataset)-valid_size, 'Valid': valid_size}

    train_dataset = SplitDataset(
        dataset, partitions, initial_partition='Train')
    valid_dataset = SplitDataset(
        dataset, partitions, initial_partition='Valid')

    return (train_dataset, valid_dataset)
