import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Modified from 
# https://github.com/HantaoShu/DeepSEM/blob/master/src/utils.py

def get_metrics(A, ground_truth):
    ''' Calculate Metrics including AUPR, AUPRR, EP, and EPR
    
    Calculate EPR given predicted adjacency matrix and BEELINE 
    ground truth
    
    Parameters
    ----------
    A: numpy.array 
        Predicted adjacency matrix. Expected size is |g| x |g|.
    ground_truth: tuple
        BEELINE ground truth object exported by 
        data.load_beeline_ground_truth. The first element of this
        tuple is eval_flat_mask, the boolean mask on the flatten
        adjacency matrix to identify TFs and target genes. The
        second element is the lable values y_true after flatten. 
        
    Returns
    -------
    tuple
        A tuple with AUPR, AUPR ratio, EP (in counts), and EPR
    '''
    eval_flat_mask, y_true, _ = ground_truth
    y_pred = np.abs(A.flatten()[eval_flat_mask])
    
    AUPR = average_precision_score(y_true, y_pred)
    AUPRR = AUPR / np.mean(y_true)
    
    num_truth_edge = int(y_true.sum())
    cutoff = np.partition(y_pred, -num_truth_edge)[-num_truth_edge]
    y_above_cutoff = y_pred > cutoff
    EP = int(np.sum(y_true[y_above_cutoff]))
    EPR = 1. * EP / ((num_truth_edge ** 2) / np.sum(eval_flat_mask))
    
    return {'AUPR': AUPR, 'AUPRR': AUPRR, 
            'EP': EP, 'EPR': EPR}

# def top_k_filter(A, evaluate_mask, topk):
#     A= abs(A)
#     if evaluate_mask is None:
#         evaluate_mask = np.ones_like(A) - np.eye(len(A))
#     A = A * evaluate_mask
#     A_val = list(np.sort(abs(A.reshape(-1, 1)), 0)[:, 0])
#     A_val.reverse()
#     cutoff_all = A_val[topk]
#     A_above_cutoff = np.zeros_like(A)
#     A_above_cutoff[abs(A) > cutoff_all] = 1
#     return A_above_cutoff

# def get_epr(A, ground_truth):
#     ''' Calculate EPR
    
#     Calculate EPR given predicted adjacency matrix and BEELINE 
#     ground truth
    
#     Parameters
#     ----------
#     A: numpy.array 
#         Predicted adjacency matrix. Expected size is |g| x |g|.
#     ground_truth: tuple
#         BEELINE ground truth object exported by 
#         data.load_beeline_ground_truth. It's a tuple with the 
#         first element being truth_edges and second element being
#         evaluate_mask.
        
#     Returns
#     -------
#     tuple
#         A tuple with calculated EP (in counts) and EPR
#     '''
#     eval_flat_mask, y_true, truth_edges, evaluate_mask = ground_truth
#     num_nodes = A.shape[0]
#     num_truth_edges = len(truth_edges)
#     A_above_cutoff = top_k_filter(A, evaluate_mask, num_truth_edges)
#     idx_source, idx_target = np.where(A_above_cutoff)
#     A_edges = set(zip(idx_source, idx_target))
#     overlap_A = A_edges.intersection(truth_edges)
#     EP = len(overlap_A)
#     EPR = 1. * EP / ((num_truth_edges ** 2) / np.sum(evaluate_mask))
#     return EP, EPR

def extract_edges(A, gene_names=None, TFmask=None):
    '''Extract predicted edges
    
    Extract edges from the predicted adjacency matrix
    
    Parameters
    ----------
    A: numpy.array 
        Predicted adjacency matrix. Expected size is |g| x |g|.
    gene_names: None, list or numpy.array
        (Optional) List of Gene Names. Usually accessible in the var_names 
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
    num_nodes = A.shape[0]
    mat_indicator_all = np.zeros([num_nodes, num_nodes])
    if TFmask is not None:
        A_masked = A * TFmask
    else:
        A_masked = A
    mat_indicator_all[abs(A_masked) > 0] = 1
    idx_source, idx_target = np.where(mat_indicator_all)
    if gene_names is None:
        source_lbl = idx_source
        target_lbl = idx_target
    else:
        source_lbl = gene_names[idx_source]
        target_lbl = gene_names[idx_target]
    edges_df = pd.DataFrame(
        {'Source': source_lbl, 'Target': target_lbl, 
         'EdgeWeight': (A[idx_source, idx_target]),
         'AbsEdgeWeight': (np.abs(A[idx_source, idx_target]))
        })
    edges_df = edges_df.sort_values('AbsEdgeWeight', ascending=False)

    return edges_df.reset_index(drop=True)
