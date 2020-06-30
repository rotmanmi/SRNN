from sacred import Experiment
from sacred.observers import FileStorageObserver
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torchvision
from datasets.addition import AddingProblemDataset
from datasets.copyproblem import CopyingMemoryProblemDataset
from datasets.mnist import MnistProblemDataset
from datasets.bigmnist import BigMnistProblemDataset
from datasets.timit_loader import TIMIT
from models.SRNN import RNNtanh, LSTM, GRU, SRNN, SRNNFast, nnRNN
from models.urnncell import URNN
from models.nrucell import NRUWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np
import os
from utils import better_hparams
import math
from nnrnnutils import select_optimizer
import time

ex = Experiment('Measure Runtime')
storage_path = 'storage_time'
ex.observers.append(FileStorageObserver.create(storage_path))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model_dict = {'srnn': SRNN, 'srnnfast': SRNNFast, 'urnn': URNN, 'rnntanh': RNNtanh, 'lstm': LSTM, 'nru': NRUWrapper,
              'gru': GRU}
datasets_dict = {'copy': CopyingMemoryProblemDataset, 'addition': AddingProblemDataset, 'pmnist': MnistProblemDataset,
                 'timit': TIMIT}
outputsize_dict = {'copy': 10, 'addition': 1, 'pmnist': 10, 'bmnist': 10, 'timit': 129}
singleoutput_dict = {'copy': False, 'addition': True, 'pmnist': True, 'bmnist': True, 'timit': False}
embedding_dict = {'copy': True, 'addition': False, 'pmnist': False, 'bmnist': False, 'timit': False}
inputsize_dict = {'copy': 10, 'addition': 2, 'pmnist': 1, 'bmnist': 1, 'timit': 129}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@ex.config
def cfg():
    sample_len = 300
    epochs = 1000
    lr = 1e-3  # rmsprop uses 1e-3, adam 1e-4
    seed = 1234
    hidden_size = 128
    hyper_size = 32
    model = 'rnntanh'  # Could be Mixed, Anti
    problem = 'pmnist'  # 'addition', 'copy','TIMIT','bmnist'
    batch_size = 100
    optimizer = 'rmsprop'
    n_layers = 1
    lr_orth = 1e-6  # Needed for nnRNN
    Tdecay = 1e-4  # Needed for nnRNN
    delta = 1e-4  # Needed for nnRNN


def train_mnist(epoch, model, dataset, optimizer):
    model.train()
    total_loss = 0
    total_accuracy = 0
    with tqdm(total=len(dataset)) as pbar:
        for i, (x, y) in enumerate(dataset):
            model.zero_grad()
            if isinstance(optimizer, tuple):
                optimizer[1].zero_grad()
            x = x.cuda()
            y = y.cuda()
            output = model(x)[0]
            loss = F.cross_entropy(output.transpose(2, 1), y.unsqueeze(1))
            total_loss += loss.item()
            _, predictions = torch.max(output, 2)
            accuracy = (predictions.squeeze() == y.squeeze()).float().mean().item()
            total_accuracy += accuracy

            pbar.set_description(
                'ce: {:.4f} accuracy: {:.4f}'.format(total_loss / (i + 1), total_accuracy / (i + 1)))
            if isinstance(model, nnRNN):
                lossnnrnn = model.rnncell.alpha_loss(1e-4) + loss
                lossnnrnn.backward()
            else:
                loss.backward()
            if isinstance(model, NRUWrapper):
                clip_grad_norm_(model.parameters(), 1)
            if isinstance(optimizer, tuple):
                model.rnncell.orthogonal_step(optimizer[1])
                optimizer[0].step()
            else:
                optimizer.step()

            pbar.update()

    pbar.set_description(
        'ce: {:.4f} accuracy: {:.4f}'.format(total_loss / len(dataset), total_accuracy / len(dataset)))

    return total_loss / len(dataset), total_accuracy / len(dataset)


def masked_loss(lossfunc, logits, y, lens):
    """ Computes the loss of the first `lens` items in the batches """
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for i, l in enumerate(lens):
        mask[i, :l, :] = 1
    logits_masked = torch.masked_select(logits, mask)
    y_masked = torch.masked_select(y, mask)
    return lossfunc(logits_masked, y_masked)


def train_timit(epoch, model, dataset, optimizer):
    model.train()
    total_loss = 0
    processed = 0
    with tqdm(total=len(dataset)) as pbar:
        for i, (x, y, lens) in enumerate(dataset):
            model.zero_grad()
            if isinstance(optimizer, tuple):
                optimizer[1].zero_grad()
            x = x.cuda()
            y = y.cuda()
            output = model(x)[0]
            loss = masked_loss(F.mse_loss, output, y.squeeze(), lens)
            total_loss += loss.item()
            processed += len(x)
            pbar.set_description('mse: {:.4f}'.format(total_loss / (i + 1)))
            if isinstance(model, nnRNN):
                lossnnrnn = model.rnncell.alpha_loss(1e-4) + loss
                lossnnrnn.backward()
            else:
                loss.backward()
            if isinstance(model, NRUWrapper):
                clip_grad_norm_(model.parameters(), 1)
            if isinstance(optimizer, tuple):
                model.rnncell.orthogonal_step(optimizer[1])
                optimizer[0].step()
            else:
                optimizer.step()

            pbar.update()

    pbar.set_description('MSE: {:.4f}'.format(total_loss / len(dataset)))

    return total_loss / len(dataset)


def train(epoch, model, dataset, optimizer):
    model.train()
    total_loss = 0
    total_accuracy = 0
    with tqdm(total=len(dataset)) as pbar:
        for i, (x, y) in enumerate(dataset):
            model.zero_grad()
            if isinstance(optimizer, tuple):
                optimizer[1].zero_grad()
            x = x.cuda()
            y = y.cuda()
            output = model(x)[0]
            if model.single_output:
                loss = F.mse_loss(output.squeeze(), y.squeeze())
                total_loss += loss.item()
                pbar.set_description('mse: {:.4f}'.format(total_loss / (i + 1)))
            else:

                loss = F.cross_entropy(output.transpose(2, 1), y.squeeze(-1))
                total_loss += loss.item()
                _, predictions = torch.max(output, 2)
                accuracy = (predictions.squeeze() == y.squeeze()).float().mean().item()
                total_accuracy += accuracy
                pbar.set_description(
                    'ce: {:.4f} accuracy: {:.4f}'.format(total_loss / (i + 1),
                                                         total_accuracy / (i + 1)))
            if isinstance(model, nnRNN):
                lossnnrnn = model.rnncell.alpha_loss(1e-4) + loss
                lossnnrnn.backward()
            else:
                loss.backward()
            if isinstance(model, NRUWrapper):
                clip_grad_norm_(model.parameters(), 1)
            if isinstance(optimizer, tuple):
                model.rnncell.orthogonal_step(optimizer[1])
                optimizer[0].step()
            else:
                optimizer.step()

            pbar.update()
        if model.single_output:
            pbar.set_description('mse: {:.4f}'.format(total_loss / len(dataset)))
        else:
            pbar.set_description(
                'ce: {:.4f} accuracy: {:.4f}'.format(total_loss / len(dataset), total_accuracy / len(dataset)))

    if model.single_output:
        return total_loss / len(dataset)
    else:
        return total_loss / len(dataset), total_accuracy / len(dataset)


def save_stat_file(fname, model_name, time, num_params, path):
    lines = ['{}: Number of model parameters: {} \n {}: Total Time {}'.format(model_name, num_params, model_name, time)]
    lines = '\n'.join(lines)
    with open(os.path.join(path, '{}.txt'.format(fname)), 'w') as f:
        f.write(lines)


@ex.automain
def main(_run):
    print(_run.config)
    torch.manual_seed(_run.config['seed'])
    np.random.seed(_run.config['seed'])
    for problem in ['copy', 'addition', 'pmnist', 'timit']:
        path = os.path.join(storage_path, problem + 'time')
        print(path)
        for model_name in model_dict.keys():
            try:
                model = model_dict[model_name](inputsize_dict[problem], outputsize_dict[problem],
                                               _run.config['hidden_size'], _run.config['n_layers'],
                                               single_output=singleoutput_dict[problem],
                                               embedding=embedding_dict[problem],
                                               hyper_size=_run.config['hyper_size']).cuda()

                perm = np.random.permutation(784)

                if problem == 'pmnist':
                    dataset = DataLoader(Subset(datasets_dict[problem]('data', train=True, download=True,
                                                                       transform=torchvision.transforms.Compose([
                                                                           torchvision.transforms.ToTensor(),
                                                                           torchvision.transforms.Normalize(
                                                                               (0.1307,), (0.3081,))
                                                                       ]), perm=perm),
                                                np.arange(50000)), batch_size=_run.config['batch_size'], shuffle=True)
                elif problem == 'timit':
                    dataset = DataLoader(datasets_dict[problem]('data/TIMIT', mode='train'),
                                         batch_size=_run.config['batch_size'],
                                         shuffle=True)

                elif problem == 'addition':
                    dataset = DataLoader(
                        datasets_dict[problem](ds_size=100 * _run.config['batch_size'],
                                               sample_len=_run.config['sample_len']),
                        batch_size=_run.config['batch_size'])


                elif problem == 'copy':
                    dataset = DataLoader(
                        datasets_dict[problem](ds_size=1000, sample_len=_run.config['sample_len']),
                        batch_size=_run.config['batch_size'])

                if model_name == 'nnrnn':
                    optimizer = select_optimizer(model, _run.config['lr'], _run.config['lr_orth'], 0.99, (0.9, 0.999),
                                                 _run.config['Tdecay'], _run.config['optimizer'])
                else:
                    if _run.config['optimizer'] == 'rmsprop':
                        optimizer = torch.optim.RMSprop(model.parameters(), lr=_run.config['lr'], alpha=0.9)
                    elif _run.config['optimizer'] == 'adam':
                        optimizer = torch.optim.Adam(model.parameters(), lr=_run.config['lr'])

                print('{}: {}: Number of model parameters: {}'.format(problem, model_name, count_parameters(model)))
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                start = time.time()
                if problem == 'pmnist':
                    _ = train_mnist(0, model, dataset, optimizer)
                elif problem == 'timit':
                    _ = train_timit(0, model, dataset, optimizer)
                else:
                    _ = train(0, model, dataset, optimizer)
                torch.cuda.synchronize()
                end = time.time()
                print('{}: {}: Running Time: {}'.format(problem, model_name, end - start))
                save_stat_file('{}_{}'.format(model_name, _run.config['hidden_size']), model_name, end - start,
                               count_parameters(model), path)

            except:
                pass

    return
