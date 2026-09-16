from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
import h5py
from sklearn.model_selection import train_test_split
import scipy
from sklearn.preprocessing import LabelEncoder
from selectgene import geneSelection_modified_by_score


def read_dataset(adata, transpose=False, test_split=False, copy=False):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6:
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error
    if transpose: adata = adata.transpose()
    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'
    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))
    return adata

    
def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        adata.X = adata.X.astype(float)
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if normalize_input:
        sc.pp.scale(adata)
    return adata


def load_data(data_path, geneSelec_num = 1000):
    data_mat = h5py.File(r"data/"+ data_path + "." + "h5")    #spector   CITEseq_GSE100866_anno
    flag = False
    x1 = np.array(data_mat['X1'])
    x2 = np.array(data_mat['X2'])
 
    y = np.array(data_mat['Y'])
    data_mat.close()

    print(f"Original shape of x1: {x1.shape}")
    selection_mask_x1 = geneSelection_modified_by_score(x1, n=geneSelec_num, plot=True) 
    x1 = x1[:, selection_mask_x1]
    print(f"Selected {x1.shape[1]} genes for x1. New shape: {x1.shape}")

    if (len(x2[0]) > 0): 
        print(f"Original shape of x2: {x2.shape}")
        selection_mask_x2 = geneSelection_modified_by_score(x2, n=geneSelec_num, plot=True)
        x2 = x2[:, selection_mask_x2]
        print(f"Selected {x2.shape[1]} genes for x2. New shape: {x2.shape}")
        flag = True 
    else:
        flag = False
        print("x2 has no genes to select from.")
        
    x = np.array(np.concatenate([x1,x2],axis=1))        

    adata = sc.AnnData(x)
    adata1 = sc.AnnData(x1)
    adata2 = sc.AnnData(x2)
    adata.obs['Group'] = y
    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata1 = read_dataset(adata1,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata1 = normalize(adata1,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    
    adata2 = sc.AnnData(x2)
    adata2.obs['Group'] = y
    adata2 = read_dataset(adata2,
                     transpose=False,
                     test_split=False,
                     copy=True)
    
    adata2 = normalize(adata2,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    label=np.array(y)
    Y=pd.DataFrame(label)
    Y = Y.dropna()
    Y = Y[Y.keys()].apply(LabelEncoder().fit_transform)
    label = np.array(Y,dtype=int)
    label=label.reshape(-1,)
    Y=np.array(label).astype(int)
    data_mat.close()
    
    return adata1.X, adata2.X, y