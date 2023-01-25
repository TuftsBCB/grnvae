# GRN-VAE

This repository include code and documentation for GRN-VAE, a stablized SEM style Variational autoencoder for gene regulatory network inference. 

The pre-print of this paper could be found [here](https://bcb.cs.tufts.edu/GRN-VAE/GRNVAE_ISMB_submission.pdf)

## Getting Started with GRN-VAE

Here we provide an end-to-end demonstration on how to infer GRN with our implementation of GRN-VAE. 


```python
from data import load_beeline
from logger import LightLogger
from runner import runGRNVAE
from evaluate import extract_edges
```

### Model Configurations

First you need to define some configs for running the model. We suggest you start with the following set of parameters. The three key concepts proposed in the GRN-VAE paper are controlled by the following parameters. 

- `delays_on_sparse`: Number of delayed steps on introducing the sparse loss. 
- `dropout_augmentation`: The proportion of data that will be randomly masked as dropout in each traing step.
- `train_on_non_zero`: Whether to train the model on non-zero expression data


```python
configs = {
    # Train/Test split
    'train_split': 0.8,
    'train_split_seed': None, 
    
    # Neural Net Definition
    'hidden_dim': 128,
    'z_dim': 1,
    'train_on_non_zero': True,
    'dropout_augmentation': 0.1,
    'cuda': True,
    
    # Loss
    'alpha': 100,
    'beta': 1,
    'delays_on_sparse': 30,
    
    # Neural Net Training
    'batch_size': 64,
    'n_epochs': 250,
    'eval_on_n_steps': 10,
    'lr_nn': 1e-4,
    'lr_adj': 2e-5,
    'K1': 1,
    'K2': 1
}
```

### Data loading
[BEELINE benchmarks](https://github.com/Murali-group/Beeline) could be loaded by the `load_beeline` function, where you specify where to look for data and which benchmark to load. If it's the first time, this function will download the files automatically. 

The `data` object exported by `load_beeline` is an [annData](https://anndata.readthedocs.io/en/stable/generated/anndata.AnnData.html#anndata.AnnData) object read by [scanpy](https://scanpy.readthedocs.io/en/stable/). The `ground_truth` object includes ground truth edges based on the BEELINE benchmark but it's not required for network inference. 

When you use GRN-VAE on a real world data to discover noval regulatory relationship, here are a few tips on preparing your data:

- Read your data using `scanpy.read`. You data should have genes in the column/var and cells in the rows/obs. Transpose your data if it's necessary. 
- Find out the most variable genes. Unlike many traditional algorithm, GRN-VAE has the capacity to run on large amount of data. Therefore you can set the number of variable genes very high. As described in the paper, we used 5,000 for our Hammond experiment. The only reason why we need this gene filter is to help converge the model.
- Normalize your data. A simple log transformation is good enough. 


```python
# Load data from a BEELINE benchmark
data, ground_truth = load_beeline(
    data_dir='data', 
    benchmark_data='mDC', 
    benchmark_setting='500_Non-ChIP'
)
```


### Model Training

Model training is simple with the `runGRNVAE` function. As said above, if ground truth is not available, just set `ground_truth` to be `None`.


```python
logger = LightLogger()
# runGRNVAE initializes and trains a GRNVAE model with the configs specified. 
vae = runGRNVAE(
    data.X, configs, ground_truth=ground_truth, logger=logger)
```

### Edge extraction
We can extract the adjacency matrix using the `.get_adj()` method. With proper labeling, we can convert the predicted adjacency matrix to an adjacency list in a pandas dataframe using the `extract_edges` function. This table could then be exported using the `.to_csv()` method. 


```python
logs = logger.to_df()
predicted_A = vae.get_adj()

gene_names = data.var_names
predicted_edges = extract_edges(A=predicted_A, gene_names=gene_names)
predicted_edges.head(10)
```
