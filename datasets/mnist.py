import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torchvision
from torchvision.transforms import CenterCrop, ToPILImage, ToTensor


class MnistProblemDataset(torchvision.datasets.MNIST):

    def __init__(self, *args, perm=None, **kwargs):
        super().__init__(*args, **kwargs)
        if perm is not None:
            self.perm = perm

    def __getitem__(self, item):
        data, label = super().__getitem__(item)
        data = data.reshape(784, 1)
        if hasattr(self, 'perm'):
            data = data[self.perm]
        return data, label

    def __len__(self):
        return super().__len__()


class RandomMnistProblemDataset(torchvision.datasets.MNIST):

    def __init__(self, *args, perm=None, cr=8, **kwargs):
        super().__init__(*args, **kwargs)
        if perm is not None:
            self.perm = perm

        self.targets = self.targets[torch.randperm(len(self.targets))]
        self.cr = 8

    def __getitem__(self, item):
        data, label = super().__getitem__(item)
        data = ToTensor()(CenterCrop(self.cr)(ToPILImage()(data)))
        data = data.reshape(self.cr * self.cr, 1)
        if hasattr(self, 'perm'):
            data = data[self.perm]
        return data, label

    def __len__(self):
        return super().__len__()


if __name__ == '__main__':
    s = Subset(MnistProblemDataset('data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), np.arange(50000))

    s = Subset(RandomMnistProblemDataset('data', train=True, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ])), np.arange(50000))

    print(len(s))
