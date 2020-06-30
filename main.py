from sacred import Experiment
from sacred.observers import FileStorageObserver
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torchvision
from datasets.addition import AddingProblemDataset
from datasets.copyproblem import CopyingMemoryProblemDataset
from datasets.mnist import MnistProblemDataset, RandomMnistProblemDataset
from datasets.bigmnist import BigMnistProblemDataset
from datasets.timit_loader import TIMIT
from models.SRNN import RNNtanh, LSTM, GRU, SRNN, nnRNN, SRNNFast
from models.urnncell import URNN
from models.nrucell import NRUWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np
import os
# from utils import better_hparams
import math
from nnrnnutils import select_optimizer

ex = Experiment('Adding Problem')
storage_path = 'storage'
ex.observers.append(FileStorageObserver.create(storage_path))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model_dict = {'srnn': SRNN, 'urnn': URNN, 'rnntanh': RNNtanh, 'lstm': LSTM, 'nru': NRUWrapper, 'gru': GRU,
              'nnrnn': nnRNN, 'srnnfast': SRNNFast}
datasets_dict = {'copy': CopyingMemoryProblemDataset, 'addition': AddingProblemDataset, 'pmnist': MnistProblemDataset,
                 'bmnist': BigMnistProblemDataset, 'rmnist': RandomMnistProblemDataset, 'timit': TIMIT}
outputsize_dict = {'copy': 10, 'addition': 1, 'pmnist': 10, 'rmnist': 10, 'bmnist': 10, 'timit': 129}
singleoutput_dict = {'copy': False, 'addition': True, 'pmnist': True, 'bmnist': True, 'timit': False, 'rmnist': True}
embedding_dict = {'copy': True, 'addition': False, 'pmnist': False, 'rmnist': False, 'bmnist': False, 'timit': False}
inputsize_dict = {'copy': 10, 'addition': 2, 'pmnist': 1, 'rmnist': 1, 'bmnist': 1, 'timit': 129}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@ex.config
def cfg():
    sample_len = 30
    epochs = 1000
    lr = 1e-3  # rmsprop uses 1e-3, adam 1e-4
    seed = 1234
    hidden_size = 128
    hyper_size = 64
    load_model = None  # 'storage/8/model.t7'
    model = 'rnntanh'  # 'rnntanh','srnn','lstm','gru','nnrnn','nru','urnn'
    problem = 'pmnist'  # 'addition', 'copy','timit','pmnist','bmnist'
    batch_size = 20
    optimizer = 'rmsprop'  # 'rmsprop' or 'adam'
    n_layers = 1  # Number of hidden layers in f_r
    lr_orth = 1e-6  # Needed for nnRNN
    Tdecay = 1e-4  # Needed for nnRNN
    delta = 1e-4  # Needed for nnRNN
    cropped = 8  # Needed for Random Label MNIST experiment
    rmnistsize = 1000
    no_gate = False


@ex.named_config
def add():
    problem = 'addition'


@ex.named_config
def copy():
    problem = 'copy'


@ex.named_config
def timit():
    problem = 'timit'


@ex.named_config
def pmnist():
    problem = 'pmnist'


@ex.named_config
def bmnist():
    problem = 'bmnist'


@ex.named_config
def rmnist():
    problem = 'rmnist'


def train_mnist(epoch, model, dataset, optimizer):
    model.train()
    total_loss = 0
    total_accuracy = 0
    processed = 0
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
            accuracy = (predictions.squeeze() == y.squeeze()).float().sum().item()
            total_accuracy += accuracy
            processed += len(x)

            pbar.set_description(
                'ce: {:.4f} accuracy: {:.4f}'.format(total_loss / (i + 1), total_accuracy / (i + 1)))
            if isinstance(model, nnRNN):
                lossnnrnn = model.rnncell.alpha_loss(1e-4) + loss
                lossnnrnn.backward()
            else:
                loss.backward()
            if isinstance(model, NRUWrapper):
                clip_grad_norm_(model.parameters(), 1)
            if isinstance(model, LSTM) or isinstance(model, RNNtanh) or isinstance(model, GRU):
                clip_grad_norm_(model.parameters(), 1)
            if isinstance(optimizer, tuple):
                model.rnncell.orthogonal_step(optimizer[1])
                optimizer[0].step()
            else:
                optimizer.step()

            pbar.update()

    pbar.set_description('ce: {:.4f} accuracy: {:.4f}'.format(total_loss / len(dataset), total_accuracy / len(dataset)))

    return total_loss / len(dataset), total_accuracy / float(processed)


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
    total_accuracy = 0
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
            if isinstance(model, LSTM) or isinstance(model, RNNtanh) or isinstance(model, GRU):
                clip_grad_norm_(model.parameters(), 1)

            if isinstance(optimizer, tuple):
                model.rnncell.orthogonal_step(optimizer[1])
                optimizer[0].step()
            else:
                optimizer.step()

            pbar.update()

    pbar.set_description('MSE: {:.4f}'.format(total_loss / len(dataset)))

    return total_loss / len(dataset)


def eval_mnist(epoch, model, dataset, optimizer):
    total_loss = 0
    total_accuracy = 0
    with tqdm(total=len(dataset)) as pbar:
        for i, (x, y) in enumerate(dataset):
            model.zero_grad()
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

            pbar.update()
        pbar.set_description(
            'ce: {:.4f} accuracy: {:.4f}'.format(total_loss / len(dataset), total_accuracy / len(dataset)))

    return total_loss / len(dataset), total_accuracy / len(dataset)


def eval_timit(epoch, model, dataset, optimizer):
    total_loss = 0
    total_accuracy = 0
    processed = 0
    with tqdm(total=len(dataset)) as pbar:
        for i, (x, y, lens) in enumerate(dataset):
            model.zero_grad()
            x = x.cuda()
            y = y.cuda()
            output = model(x)[0]
            loss = masked_loss(F.mse_loss, output, y.squeeze(), lens)
            total_loss += loss.item()
            processed += len(x)
            pbar.set_description('mse: {:.4f}'.format(total_loss / (i + 1)))

            pbar.update()
    pbar.set_description(
        'MSE: {:.4f}'.format(total_loss / len(dataset)))

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
                'ce: {:.4f} accuracy: {:.4f}'.format(total_loss / len(dataset),
                                                     total_accuracy / len(dataset)))

    if model.single_output:
        return total_loss / len(dataset)
    else:
        return total_loss / len(dataset), total_accuracy / len(dataset)


def save_model(model, cfg, path):
    with open(os.path.join(path, 'model.t7'), 'wb') as f:
        torch.save({'state': model.state_dict(), 'args': cfg}, f)


def save_stat_file(fname, field, path):
    lines = ['Step,{}'.format(fname)] + ['{},{:.6f}'.format(i + 1, val) for i, val in enumerate(field)]
    lines = '\n'.join(lines)
    with open(os.path.join(path, '{}.txt'.format(fname)), 'w') as f:
        f.write(lines)


@ex.automain
def main(_run):
    print(_run.config)
    torch.manual_seed(_run.config['seed'])
    np.random.seed(_run.config['seed'])

    problem = _run.config['problem']
    path = os.path.join(storage_path, problem, _run._id)
    print(path)
    if problem == 'pmnist' or problem == 'bmnist' or problem == 'rmnist':
        if problem == 'pmnist':
            perm = np.random.permutation(784)
        elif problem == 'bmnist':
            perm = np.random.permutation(784 * 4)
        elif problem == 'rmnist':
            perm = None
        else:
            perm = None
        if problem == 'rmnist':
            dataset = DataLoader(Subset(datasets_dict[problem]('data', train=True, download=True,
                                                               transform=torchvision.transforms.Compose([
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.Normalize(
                                                                       (0.1307,), (0.3081,))
                                                               ]), perm=perm, cr=_run.config['cropped']),
                                        np.arange(_run.config['rmnistsize'])), batch_size=_run.config['batch_size'],
                                 shuffle=True)
        else:
            dataset = DataLoader(Subset(datasets_dict[problem]('data', train=True, download=True,
                                                               transform=torchvision.transforms.Compose([
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.Normalize(
                                                                       (0.1307,), (0.3081,))
                                                               ]), perm=perm),
                                        np.arange(50000)), batch_size=_run.config['batch_size'], shuffle=True)

        val_dataset = DataLoader(Subset(datasets_dict[problem]('data', train=True, download=True,
                                                               transform=torchvision.transforms.Compose([
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.Normalize(
                                                                       (0.1307,), (0.3081,))
                                                               ]), perm=perm),
                                        np.arange(50000, 60000)), batch_size=_run.config['batch_size'], shuffle=True)
        test_dataset = DataLoader(datasets_dict[problem]('data', train=False, download=True,
                                                         transform=torchvision.transforms.Compose([
                                                             torchvision.transforms.ToTensor(),
                                                             torchvision.transforms.Normalize(
                                                                 (0.1307,), (0.3081,))
                                                         ]), perm=perm), batch_size=1, shuffle=False)

        if _run.config['model'] == 'srnn' or _run.config['model'] == 'srnnfast':
            if problem == 'rmnist':
                summary_path = os.path.join(storage_path, problem,
                                            _run.config['model'] + ('no_gate' if _run.config['no_gate'] else ''),
                                            str(_run.config['rmnistsize']),
                                            _run.config['optimizer'],
                                            str(_run.config['hidden_size']), str(_run.config['hyper_size']),
                                            str(_run.config['n_layers']))
            else:
                summary_path = os.path.join(storage_path, problem,
                                            _run.config['model'] + ('no_gate' if _run.config['no_gate'] else ''),
                                            _run.config['optimizer'],
                                            str(_run.config['hidden_size']), str(_run.config['hyper_size']),
                                            str(_run.config['n_layers']))

        else:
            if problem == 'rmnist':
                summary_path = os.path.join(storage_path, problem, _run.config['model'], str(_run.config['rmnistsize']))
            else:
                summary_path = os.path.join(storage_path, problem, _run.config['model'])

    elif problem == 'copy':
        dataset = DataLoader(
            datasets_dict[problem](ds_size=1000, sample_len=_run.config['sample_len']),
            batch_size=_run.config['batch_size'])
        summary_path = os.path.join(storage_path, problem, str(_run.config['sample_len']),
                                    _run.config['model'] + ('no_gate' if _run.config['no_gate'] else ''),
                                    str(_run.config['hidden_size']), str(_run.config['hyper_size']),
                                    str(_run.config['n_layers']), str(_run._id))

    elif problem == 'timit':
        dataset = DataLoader(datasets_dict[problem]('data/TIMIT', mode='train'), batch_size=_run.config['batch_size'],
                             shuffle=True)
        val_dataset = DataLoader(datasets_dict[problem]('data/TIMIT', mode='val'), batch_size=192,
                                 shuffle=True)
        test_dataset = DataLoader(datasets_dict[problem]('data/TIMIT', mode='test'), batch_size=400,
                                  shuffle=True)

        if _run.config['model'] == 'srnn' or _run.config['model'] == 'srnnfast':
            summary_path = os.path.join(storage_path, problem,
                                        _run.config['model'] + ('no_gate' if _run.config['no_gate'] else ''),
                                        _run.config['optimizer'],
                                        str(_run.config['hidden_size']), str(_run.config['hyper_size']),
                                        str(_run.config['n_layers']))
        else:
            summary_path = os.path.join(storage_path, problem, _run.config['model'],
                                        _run.config['optimizer'], str(_run.config['hidden_size']),
                                        str(_run.config['hidden_size']))
    elif problem == 'addition':
        dataset = DataLoader(
            datasets_dict[problem](ds_size=100 * _run.config['batch_size'], sample_len=_run.config['sample_len']),
            batch_size=_run.config['batch_size'])
        summary_path = os.path.join(storage_path, problem, str(_run.config['sample_len']),
                                    _run.config['model'] + ('no_gate' if _run.config['no_gate'] else ''))

    else:
        dataset = DataLoader(
            datasets_dict[problem](ds_size=1000 * _run.config['batch_size'], sample_len=_run.config['sample_len']),
            batch_size=_run.config['batch_size'])
        summary_path = os.path.join(storage_path, problem, str(_run.config['sample_len']),
                                    _run.config['model'] + ('no_gate' if _run.config['no_gate'] else ''))

    model = model_dict[_run.config['model']](inputsize_dict[problem], outputsize_dict[problem],
                                             _run.config['hidden_size'], _run.config['n_layers'],
                                             single_output=singleoutput_dict[problem],
                                             embedding=embedding_dict[problem],
                                             hyper_size=_run.config['hyper_size'],
                                             multihead=not _run.config['no_gate']).cuda()

    if _run.config['load_model'] is not None:
        state = torch.load(_run.config['load_model'])
        model.load_state_dict(state['state'])
    else:
        _run.config['load_model'] = 'None'

    if _run.config['model'] == 'nnrnn':
        optimizer = select_optimizer(model, _run.config['lr'], _run.config['lr_orth'], 0.99, (0.9, 0.999),
                                     _run.config['Tdecay'], _run.config['optimizer'])
    else:
        if _run.config['optimizer'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=_run.config['lr'], alpha=0.9)
        elif _run.config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=_run.config['lr'])

    writer = SummaryWriter(log_dir=os.path.join(summary_path))
    print('Number of model parameters: {}'.format(count_parameters(model)))
    best_mse = np.inf
    best_acc = 0
    best_ce = np.inf
    metric_dict = {'MSE/Best_MSE': best_mse, 'CE/Best_CE': best_ce, 'Accuracy/Best_Accuracy': best_acc}
    # sei = better_hparams(writer, hparam_dict=_run.config, metric_dict=metric_dict)

    if problem == 'pmnist' or problem == 'bmnist' or problem == 'rmnist':
        VALsCE = []
        TESTsCE = []
        VALsACC = []
        TESTsACC = []
        with tqdm(total=_run.config['epochs']) as pbar:
            for i in range(1, _run.config['epochs']):
                pbar.set_description('Epoch: {:04d}'.format(i))
                res = train_mnist(i, model, dataset, optimizer)
                writer.add_scalar('CE/Train', res[0], i)
                writer.add_scalar('Accuracy/Train', res[1], i)
                if problem != 'rmnist':
                    res_eval = eval_mnist(i, model, val_dataset, optimizer)
                else:
                    res_eval = [7, -1]
                writer.add_scalar('CE/Validation', res_eval[0], i)
                writer.add_scalar('Accuracy/Validation', res_eval[1], i)
                if problem == 'rmnist':
                    VALsCE.append(res[0])
                    VALsACC.append(res[1])
                else:
                    VALsCE.append(res_eval[0])
                    VALsACC.append(res_eval[1])
                if res_eval[0] < best_ce and problem != 'rmnist':
                    best_ce = res_eval[0]
                    save_model(model, _run.config, summary_path)
                    writer.add_scalar('CE/Best_CE', res[0], i)

                    res_eval_test = eval_mnist(i, model, test_dataset, optimizer)
                    TESTsCE.append(res_eval_test[0])
                    TESTsACC.append(res_eval_test[1])
                    writer.add_scalar('CE/Test', res_eval_test[0], i)
                    writer.add_scalar('Accuracy/Test', res_eval_test[1], i)

                if res_eval[1] < best_acc and problem != 'rmnist':
                    best_acc = res_eval[1]
                    writer.add_scalar('Accuracy/Best_Accuracy', res[1], i)

                pbar.update()
                save_stat_file('CEvalid', VALsCE, summary_path)
                save_stat_file('CEtest', TESTsCE, summary_path)
                save_stat_file('ACCvalid', VALsACC, summary_path)
                save_stat_file('ACCtest', TESTsACC, summary_path)

    elif problem == 'timit':
        VALsMSE = []
        TESTsMSE = []
        with tqdm(total=_run.config['epochs']) as pbar:
            for i in range(1, _run.config['epochs']):
                pbar.set_description('Epoch: {:04d}'.format(i))
                res = train_timit(i, model, dataset, optimizer)
                writer.add_scalar('MSE/Train', res, i)
                res_eval = eval_timit(i, model, val_dataset, optimizer)
                writer.add_scalar('MSE/Validation', res_eval, i)
                VALsMSE.append(res_eval)
                if res_eval < best_mse:
                    best_mse = res_eval
                    save_model(model, _run.config, summary_path)
                    writer.add_scalar('MSE/Best_MSE', res, i)

                    res_eval_test = eval_timit(i, model, test_dataset, optimizer)
                    TESTsMSE.append(res_eval_test)
                    writer.add_scalar('MSE/Test', res_eval_test, i)

                pbar.update()
                save_stat_file('MSEvalid', VALsMSE, summary_path)
                save_stat_file('MSEtest', TESTsMSE, summary_path)
    else:
        sample_len = _run.config['sample_len']
        MSEs = []
        CEs = []
        accs = []
        with tqdm(total=_run.config['epochs']) as pbar:
            for i in range(1, _run.config['epochs']):

                res = train(i, model, dataset, optimizer)
                MSEs.append(res)
                if problem == 'addition':
                    pbar.set_description('Addition Epoch: {:04d}; Sample Length: {:04d}'.format(i, sample_len))
                    writer.add_scalar('MSE/Train', res, i)

                    if res < best_mse:
                        best_mse = res
                        save_model(model, _run.config, summary_path)
                        writer.add_scalar('MSE/Best_MSE', res, i)

                    save_stat_file('MSE', MSEs, summary_path)
                elif problem == 'copy' or problem == 'denoising' or problem == 'vcopy':
                    if problem == 'copy':
                        pbar.set_description('Memory Copy E: {:04d}; Time Gap: {:04d}'.format(i, sample_len))
                    elif problem == 'vcopy':
                        pbar.set_description(
                            'Variable Memory Copy E: {:04d}; Time Gap: {:04d}'.format(i, sample_len))
                    else:
                        pbar.set_description('Denoising E: {:04d}; Time Gap: {:04d}'.format(i, sample_len))
                    writer.add_scalar('CE/Train', res[0], i)
                    writer.add_scalar('Accuracy/Train', res[1], i)
                    CEs.append(res[0])
                    accs.append(res[1])
                    if res[0] < best_ce:
                        best_ce = res[0]
                        save_model(model, _run.config, summary_path)
                        writer.add_scalar('CE/Best_CE', res[0], i)

                    if res[1] < best_acc:
                        best_acc = res[1]
                        writer.add_scalar('Accuracy/Best_Accuracy', res[1], i)
                    save_stat_file('CE', CEs, summary_path)
                    save_stat_file('Accuracy', accs, summary_path)
                pbar.update()

    # writer.file_writer.add_summary(sei)

    return best_mse
