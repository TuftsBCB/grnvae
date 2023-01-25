import numpy as np
import pandas as pd

# Modified from 
# https://github.com/HantaoShu/DeepSEM/blob/master/src/utils.py

# Evaluation
def top_k_filter(A, evaluate_mask, topk):
    A= abs(A)
    if evaluate_mask is None:
        evaluate_mask = np.ones_like(A) - np.eye(len(A))
    A = A * evaluate_mask
    A_val = list(np.sort(abs(A.reshape(-1, 1)), 0)[:, 0])
    A_val.reverse()
    cutoff_all = A_val[topk]
    A_above_cutoff = np.zeros_like(A)
    A_above_cutoff[abs(A) > cutoff_all] = 1
    return A_above_cutoff

def get_epr(A, ground_truth):
    ''' Calculate EPR
    
    Calculate EPR given predicted adjacency matrix and BEELINE 
    ground truth
    
    Parameters
    ----------
    A: numpy.array 
        Predicted adjacency matrix. Expected size is |g| x |g|.
    ground_truth: tuple
        BEELINE ground truth object exported by 
        data.load_beeline_ground_truth. It's a tuple with the 
        first element being truth_edges and second element being
        evaluate_mask.
        
    Returns
    -------
    tuple
        A tuple with calculated EP (in counts) and EPR
    '''
    truth_edges, evaluate_mask = ground_truth
    num_nodes = A.shape[0]
    num_truth_edges = len(truth_edges)
    A_above_cutoff = top_k_filter(A, evaluate_mask, num_truth_edges)
    idx_source, idx_target = np.where(A_above_cutoff)
    A_edges = set(zip(idx_source, idx_target))
    overlap_A = A_edges.intersection(truth_edges)
    EP = len(overlap_A)
    EPR = 1. * EP / ((num_truth_edges ** 2) / np.sum(evaluate_mask))
    return EP, EPR

def extract_edges(A, gene_names, TFmask=None):
    '''Extract predicted edges
    
    Extract edges from the predicted adjacency matrix
    
    Parameters
    ----------
    A: numpy.array 
        Predicted adjacency matrix. Expected size is |g| x |g|.
    gene_names: list or numpy.array
        List of Gene Names. Usually accessible in the var_names 
        field of scanpy data. 
    TFmask: numpy.array
        A masking matrix indicating the position of TFs. Expected 
        size is |g| x |g|.
        
    Returns
    -------
    pandas.DataFrame
        A DataFrame including all the predicted links with predicted
        link strength.
    '''
    gene_names = np.array(gene_names)
    num_nodes = A.shape[0]
    mat_indicator_all = np.zeros([num_nodes, num_nodes])
    if TFmask is not None:
        A_masked = A * TFmask
    else:
        A_masked = A
    mat_indicator_all[abs(A_masked) > 0] = 1
    idx_source, idx_target = np.where(mat_indicator_all)
    edges_df = pd.DataFrame(
        {'Source': gene_names[idx_source], 'Target': gene_names[idx_target], 
         'EdgeWeight': (A[idx_source, idx_target]),
         'AbsEdgeWeight': (np.abs(A[idx_source, idx_target]))
        })
    edges_df = edges_df.sort_values('AbsEdgeWeight', ascending=False)

    return edges_df.reset_index(drop=True)