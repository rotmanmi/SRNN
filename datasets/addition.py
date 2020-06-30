import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AddingProblemDataset(Dataset):
    def __init__(self, ds_size=1000, sample_len=50):
        super().__init__()
        self.sample_len = sample_len
        self.ds_size = ds_size

    def generate_sample(self, num_samples):
        X_value = np.random.uniform(low=0, high=1, size=(self.sample_len, 1))
        X_mask = np.zeros((self.sample_len, 1))
        half = int(self.sample_len / 2)
        first_i = np.random.randint(half)
        second_i = np.random.randint(half) + half
        X_mask[(first_i, second_i), 0] = 1
        Y = np.sum(X_value[(first_i, second_i), 0])
        X = np.concatenate((X_value, X_mask), 1)
        return X, Y

    def __getitem__(self, item):
        return [torch.tensor(x, dtype=torch.float) for x in self.generate_sample(1)]

    def __len__(self):
        return self.ds_size


if __name__ == '__main__':
    s = AddingProblemDataset(ds_size=1000, sample_len=6)
    d = DataLoader(s, batch_size=1)
    print(next(enumerate(d))[1])
