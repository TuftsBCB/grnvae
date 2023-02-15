import numpy as np
import scanpy as sc
import pandas as pd
import urllib
from tqdm import tqdm
import zipfile
import os

# Read ground truth
def load_beeline_ground_truth(data_dir, gene_names):
    n_gene = len(gene_names)
    ground_truth = pd.read_csv(f'{data_dir}/label.csv')
    TF = set(ground_truth['Gene1'])
    All_gene = set(ground_truth['Gene1']) | set(ground_truth['Gene2'])

    evaluate_mask = np.zeros([n_gene, n_gene])
    TF_mask = np.zeros([n_gene, n_gene])
    for i, item in enumerate(gene_names):
        for j, item2 in enumerate(gene_names):
            if i == j:
                continue
            if item in TF and item2 in All_gene:
                evaluate_mask[i, j] = 1
            if item in TF:
                TF_mask[i, j] = 1
    
    truth_df = pd.DataFrame(np.zeros([n_gene, n_gene]), 
                            index=gene_names, columns=gene_names)
    for i in range(ground_truth.shape[0]):
        truth_df.loc[ground_truth.iloc[i, 0], ground_truth.iloc[i, 1]] = 1
    A_truth = truth_df.values

    idx_source, idx_target = np.where(A_truth)
    truth_edges = set(zip(idx_source, idx_target))
    
    eval_flat_mask = (evaluate_mask.flatten() != 0)
    y_true = A_truth.flatten()[eval_flat_mask]
    
    return eval_flat_mask, y_true, truth_edges

def load_beeline(data_dir, benchmark_data='hESC', 
                 benchmark_setting='500_STRING'):
    ''' Load BEELINE
    
    Load BEELINE data into memory (download if necessary).
    
    Parameters
    ----------
    data_dir: str
        Root folder where the BEELINE data is/will be located. 
    benchmark_data: str
        Benchmark datasets. Choose among `hESC`, `hHep`, `mDC`, 
        `mESC`, `mHSC`, `mHSC-GM`, and `mHSC-L`.
    benchmark_setting: str
        Benchmark settings. Choose among `500_STRING`, 
        `1000_STRING`, `500_Non-ChIP`, `1000_Non-ChIP`, 
        `500_ChIP-seq`, `1000_ChIP-seq`, `500_lofgof`,
        and `1000_lofgof`. If either of the `lofgof` settings
        is choosed, only `mESC` data is available. 
        
    Returns
    -------
    tuple
        First element is a scanpy data with cells on rows and 
        genes on columns. Second element is the corresponding 
        BEELINE ground truth data 
    '''
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(f'{data_dir}/BEELINE/'):
        download_beeline(data_dir)
    data_dir = f'{data_dir}/BEELINE/{benchmark_setting}_{benchmark_data}'
    data = sc.read(f'{data_dir}/data.csv')
    # We do need to transpose the data to have cells on rows and genes on columns
    data = data.transpose()
    ground_truth = load_beeline_ground_truth(data_dir, data.var_names)
    return data, ground_truth

def download_beeline(save_dir, remove_zip=True):
    if not os.path.exists(save_dir):
        raise Exception("save_dir does not exist")
    zip_path = os.path.join(save_dir, 'BEELINE.zip')
    download_file('https://bcb.cs.tufts.edu/GRN-VAE/BEELINE.zip', 
                  zip_path)
    with zipfile.ZipFile(zip_path,"r") as zip_ref:
        for file in tqdm(desc='Extracting', iterable=zip_ref.namelist(), 
                         total=len(zip_ref.namelist())):
            zip_ref.extract(member=file, path=save_dir)
    if remove_zip:
        os.remove(zip_path)

# Modified from https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download_file(url, file_path, chunk_size=1024):
    req = urllib.request.urlopen(url)
    with open(file_path, 'wb') as f, tqdm(
        desc=f'Downloading {file_path}', total=req.length, unit='iB', 
        unit_scale=True, unit_divisor=1024
    ) as bar:
        for _ in range(req.length // chunk_size + 1):
            chunk = req.read(chunk_size)
            if not chunk: break
            size = f.write(chunk)
            bar.update(size)
