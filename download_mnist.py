from datasets.mnist import MnistProblemDataset
from datasets.bigmnist import BigMnistProblemDataset


dataset = MnistProblemDataset('data', train=True, download=True)
dataset = BigMnistProblemDataset('data', train=True, download=True)