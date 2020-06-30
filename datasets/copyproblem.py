import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CopyingMemoryProblemDataset(Dataset):
    def __init__(self, ds_size=1000, sample_len=50):
        super().__init__()
        self.sample_len = sample_len + 20
        self.ds_size = ds_size

    def generate_sample(self, num_samples):
        assert (self.sample_len > 20)  # must be

        X = np.zeros((self.sample_len, 1))
        data = np.random.randint(low=1, high=9, size=(10, 1))
        X[:10] = data
        X[-11] = 9
        Y = np.zeros((self.sample_len, 1))
        Y[-10:] = X[:10]
        return X, Y

    def __getitem__(self, item):
        return [torch.tensor(x, dtype=torch.long) for x in self.generate_sample(1)]

    def __len__(self):
        return self.ds_size


if __name__ == '__main__':
    s = CopyingMemoryProblemDataset(ds_size=1000, sample_len=21)
    d = DataLoader(s, batch_size=1)
    t = next(enumerate(d))[1]
    print(t[0])
    print(t[1])
