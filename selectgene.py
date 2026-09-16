import scanpy as sc
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

nmi = normalized_mutual_info_score
ami = adjusted_mutual_info_score
ari = adjusted_rand_score
    
def geneSelection_modified_by_score(data, n=1000, plot=False):
    if data.ndim != 2:
        raise ValueError("data must be a 2D array with shape (cells, genes)")

    n_cells, n_genes = data.shape
    mean_expr = np.mean(data, axis=0)       
    var_expr  = np.var(data, axis=0)        
    dispersion = var_expr / (mean_expr + 1e-12) 

    min_mean = 0.01  
    valid_mask = mean_expr > min_mean
    valid_count = int(np.sum(valid_mask))
    
    selection_mask = np.zeros(n_genes, dtype=bool)

    if valid_count > n:
        
        valid_dispersion = dispersion[valid_mask]
        denom = np.std(valid_dispersion)
        
        if denom == 0:
            dispersion_zscore = np.zeros_like(valid_dispersion)
        else:
            dispersion_zscore = (valid_dispersion - np.mean(valid_dispersion)) / denom

        top_indices_in_valid = np.argsort(dispersion_zscore)[-n:]
        top_indices = np.where(valid_mask)[0][top_indices_in_valid]

        selection_mask[top_indices] = True

        if plot:
            plt.figure(figsize=(10, 4))
            valid_mean_expr = mean_expr[valid_mask]
            plt.scatter(np.log10(valid_mean_expr + 1e-12), valid_dispersion, c='gray', alpha=0.5, s=10)
            plt.scatter(np.log10(valid_mean_expr[top_indices_in_valid] + 1e-12),
                        valid_dispersion[top_indices_in_valid],
                        c='red', label=f"Top {n} genes")
            plt.xlabel("log10(mean expression)")
            plt.ylabel("dispersion")
            plt.legend()
            plt.tight_layout()
            plt.show()
    else:
        selection_mask[:] = True
        if plot:
            print(f"The number of valid genes is {valid_count}, which is <= n ({n}). No filtering is performed, and all genes are retained.")
    return selection_mask
