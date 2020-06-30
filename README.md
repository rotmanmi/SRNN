# Shifting Recurrent Neural Network

This repository is a supplemental to our work Shifting Recurrent Neural Network. It also contains implementations of baseline methods: the nnRNN has been adapted from [https://github.com/KyleGoyette/nnRNN](https://github.com/KyleGoyette/nnRNN), the NRU code has been adapted from [https://github.com/apsarath/NRU](https://github.com/apsarath/NRU) and the urnn code from [https://github.com/rand0musername/urnn/](https://github.com/rand0musername/urnn/).
 


We provide benchmarks for 5 tasks:
* Memory Copy - copy
* Addition - add
* Permuted MNIST - pmnist
* Big Permuted MNIST - bmnist
* TIMIT - timit
* Random MNIST Labels - rmnist (appears in the supplemental)
 
In order to run the tasks use the following commands:

```bash
    python main with task_name model=model_name hidden_size=...  n_layers=... hyper_size=... batch_size=... optimizer=...
```
Note that for both the add and copy tasks another argument is ```sample_len``` that determines the time lag ```T```. ```hyper_size``` is the size of the hidden layers contained in ```f_r``` of network ```b```, and ```n_layers``` are the number of hidden layers in ```f_r```.
The possible models are
* srnnfast ([CUDA compilation instructions](./README.md#SRNN-CUDA-IMPLEMENTATION))
* srnn
* rnntanh
* lstm
* gru
* nru
* urnn
* nnrnn

For instance, to run our SRNN model on the Memory Copy task with T=200 with a hidden size of 128, with ```f_r``` consisting of one hidden layer with size of 8 and a batch size of 20:
```bash
    python main.py with copy model='srnn' sample_len=200 hidden_size=128 hyper_size=8 batch_size=20 optimizer=rmsprop
``` 

In order to experiment with SRNN model without gating, use the additional flag ```no_gate=True```. Note that this flag only works for the srnn version and not the srnnfast.

After each execution, a new subfolder named ```storage``` would be created containing a folder corresponding to each task. Inside each folder you can find the relevant CE.txt or MSE.txt containing the relevant cross entropy or MSE for that experiment. The MNIST and TIMIT folders also contain the cross entropy and MSE over the validation and test set. ```storage/logs``` contains tensorboard logs should you want to visualize the results.

The [nnRNN](https://github.com/KyleGoyette/nnRNN) model requires additional parameters:
*    lr_orth = 1e-6 
*    Tdecay = 1e-4 
*    delta = 1e-4 
More information on these can be found in the [official github repository](https://github.com/KyleGoyette/nnRNN)

In order to view all available parameters run
```bash
python main.py print_config
```

## Permuted MNIST and BIG Permuted MNIST
Before running these experiments please run 
```bash
python download_mnist.py
```
in order to download the MNIST dataset.

## TIMIT
For the TIMIT experiment please follow the instructions on how to preprocess the data for pytorch available in [https://github.com/Lezcano/expRNN](https://github.com/Lezcano/expRNN). Our code assumes the ```*.pt``` files are then placed under ```data/TIMIT```. 

## Random MNIST Labels
In order to change the patch size use ```cropped=8``` or ```cropped=16``` as an argument.

## Runtime
Runtime measurement can be calculated using 
```bash
python measure_runtimes.py
```

## Requirements
This code was tested using PyTorch 1.3.1
```bash
numpy == 1.17.4
sacred == 0.8.1
tqdm == 4.40.2
scipy == 1.4.1

```

##SRNN CUDA IMPLEMENTATION
In order to use the SRNN implementation, first compile the cuda binaries by (tested using CUDA 10.1)
```bash
cd models/cuda
python setup.py install
```
Next, make sure that L91 in ```models/SRNN.py``` points to the srnn.so generated. (the folder depends on the OS and python version)

