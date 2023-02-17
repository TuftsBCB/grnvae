# GRN-VAE

This repository include code and documentation for GRN-VAE, a stablized SEM style Variational autoencoder for gene regulatory network inference. 

The pre-print of this paper could be found [here](https://bcb.cs.tufts.edu/GRN-VAE/GRNVAE_ISMB_submission.pdf)

# Getting Started with GRN-VAE

This document provides an end-to-end demonstration on how to infer GRN with our implementation of GRN-VAE. 


```python
import numpy as np
from data import load_beeline
from logger import LightLogger
from runner import runGRNVAE
from evaluate import extract_edges, get_metrics
import seaborn as sns
import matplotlib.pyplot as plt
```

## Model Configurations

First you need to define some configs for running the model. We suggest you start with the following set of parameters. The three key concepts proposed in the GRN-VAE paper are controlled by the following parameters. 

- `delayed_steps_on_sparse`: Number of delayed steps on introducing the sparse loss. 
- `dropout_augmentation`: The proportion of data that will be randomly masked as dropout in each traing step.
- `train_on_non_zero`: Whether to train the model on non-zero expression data

Note that here we also use lower learning rates for a longer period of time when model is trained on non-zero data. 


```python
configs = {
    # Train/Test split
    'train_split': 1.0,
    'train_split_seed': None, 
    
    # Neural Net Definition
    'hidden_dim': 128,
    'z_dim': 1,
    'A_dim': 0,
    'train_on_non_zero': True,
    'dropout_augmentation': 0.1,
    'cuda': True,
    
    # Loss
    'alpha': 0.1,
    'beta': 1,
    'h_scale': 0,
    'delayed_steps_on_sparse': 30,
    
    # Neural Net Training
    'batch_size': 256,
    'n_epochs': 1000,
    'schedule': [30, 100],
    'eval_on_n_steps': 10,
    'early_stopping': 0,
    'lr_nn': 1e-4,
    'lr_adj': 2e-5,
    'K1': 1,
    'K2': 1
}
```

## Data loading
[BEELINE benchmarks](https://github.com/Murali-group/Beeline) could be loaded by the `load_beeline` function, where you specify where to look for data and which benchmark to load. If it's the first time, this function will download the files automatically. 

The `data` object exported by `load_beeline` is an [annData](https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.html#anndata.AnnData) object read by [scanpy](https://scanpy.readthedocs.io/en/stable/). The `ground_truth` object includes ground truth edges based on the BEELINE benchmark but it's not required for network inference. 

When you use GRN-VAE on a real world data to discover noval regulatory relationship, here are a few tips on preparing your data:

- You can read in data in any formats but make sure your data has genes in the column/var and cells in the rows/obs. Transpose your data if it's necessary. 
- Find out the most variable genes. Unlike many traditional algorithm, GRN-VAE has the capacity to run on large amount of data. Therefore you can set the number of variable genes very high. As described in the paper, we used 5,000 for our Hammond experiment. The only reason why we need this gene filter is to help converge the model.
- Normalize your data. A simple log transformation is good enough. 


```python
# Load data from a BEELINE benchmark
data, ground_truth = load_beeline(
    data_dir='data', 
    benchmark_data='hESC', 
    benchmark_setting='500_STRING'
)
```

## Model Training

Model training is simple with the `runGRNVAE` function. As said above, if ground truth is not available, just set `ground_truth` to be `None`.


```python
logger = LightLogger()
# runGRNVAE initializes and trains a GRNVAE model with the configs specified. 
vae, adjs = runGRNVAE(
    data.X, configs, ground_truth=ground_truth, logger=logger)
```

    /h/hao/miniconda3/lib/python3.10/site-packages/torch/cuda/__init__.py:497: UserWarning: Can't initialize NVML
      warnings.warn("Can't initialize NVML")
      0%|          | 0/1000 [00:00<?, ?it/s]/h/hao/miniconda3/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:138: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
      warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    100%|██████████| 1000/1000 [01:04<00:00, 15.47it/s]


The learned adjacency matrix could be obtained by the `get_adj()` method. For BEELINE benchmarks, you can get the performance metrics of this run using the `get_metrics` function. 


```python
A = vae.get_adj()
get_metrics(A, ground_truth)
```




    {'AUPR': 0.0552235472765148,
     'AUPRR': 2.295960814041767,
     'EP': 476,
     'EPR': 4.648827955381867}
