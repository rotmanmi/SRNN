import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
import torchvision
import torchvision.transforms.functional as F


class BigMnistProblemDataset(torchvision.datasets.MNIST):

    def __init__(self, *args, perm=None, **kwargs):
        super().__init__(*args, **kwargs)
        if perm is not None:
            self.perm = perm

    def __getitem__(self, item):
        data, label = super().__getitem__(item)
        data = F.to_tensor(F.pad(F.to_pil_image(data), 14))
        data = data.reshape(3136, 1)
        if hasattr(self, 'perm'):
            data = data[self.perm]
        return data, label

    def __len__(self):
        return super().__len__()




if __name__ == '__main__':
    s = Subset(BigMnistProblemDataset('data', train=True, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ])), np.arange(50000))

    print(next(enumerate(s)))
    # l = next(enumerate(s))
    # print(l)
