# GRN-VAE

This repository include code and documentation for GRN-VAE, a stablized SEM style Variational autoencoder for gene regulatory network inference. 

The pre-print of this paper could be found [here](https://bcb.cs.tufts.edu/GRN-VAE/GRNVAE_ISMB_submission.pdf)

# Getting Started with GRN-VAE

This document provides an end-to-end demonstration on how to infer GRN with our implementation of GRN-VAE. 


```python
import numpy as np
from data import load_beeline
from logger import LightLogger
from runner import runGRNVAE, runGRNVAE_ensemble, DEFAULT_GRNVAE_CONFIGS
from runner import runDeepSEM, runDeepSEM_ensemble, DEFAULT_DEEPSEM_CONFIGS
from evaluate import extract_edges, get_metrics
import seaborn as sns
import matplotlib.pyplot as plt
```

## Model Configurations

First you need to define some configs for running the model. Here we provide a default set of parameters in `runner` called `DEFAULT_GRNVAE_CONFIGS`. It comes in the forms of a standard python dictionary so it's very easy to modify as needed. 

The three key concepts proposed in the GRN-VAE paper are controlled by the following parameters. 

- `delayed_steps_on_sparse`: Number of delayed steps on introducing the sparse loss. 
- `dropout_augmentation`: The proportion of data that will be randomly masked as dropout in each traing step.
- `train_on_non_zero`: Whether to train the model on non-zero expression data

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
    data.X, DEFAULT_GRNVAE_CONFIGS, ground_truth=ground_truth, logger=logger)
```

    100%|██████████| 120/120 [00:33<00:00,  3.63it/s]


The learned adjacency matrix could be obtained by the `get_adj()` method. For BEELINE benchmarks, you can get the performance metrics of this run using the `get_metrics` function. 


```python
A = vae.get_adj()
get_metrics(A, ground_truth)
```




    {'AUPR': 0.05958849485016752,
     'AUPRR': 2.4774368161948437,
     'EP': 504,
     'EPR': 4.922288423345506}

We also provide our own implementation of [DeepSEM](https://www.nature.com/articles/s43588-021-00099-8). You can execute DeepSEM and the ensemble version of it using `runDeepSEM` and `runDeepSEM_ensemble`.
