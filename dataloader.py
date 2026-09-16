import numpy as np
from torch.utils.data import Dataset
import torch
import preprocess

class GSE100866(Dataset):
    def __init__(self, path):
        data1,data2,labels = preprocess.load_data("GSE100866",2000)
        self.x1 = data1.astype(np.float32)
        self.x2 = data2.astype(np.float32)
        self.y = labels.astype(np.int64) 
        print(data1.shape)
        print(data2.shape)
        print(labels.shape)

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], torch.from_numpy(np.array([self.y[idx]])), torch.from_numpy(np.array(idx)).long()

class GSE128639(Dataset):
    def __init__(self, path):
        data1,data2,labels = preprocess.load_data("GSE128639",2000)
        self.x1 = data1.astype(np.float32)
        self.x2 = data2.astype(np.float32)
        self.y = labels.astype(np.int64) 
        print(data1.shape)
        print(data2.shape)
        print(labels.shape)

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], torch.from_numpy(np.array([self.y[idx]])), torch.from_numpy(np.array(idx)).long()
    
class PBMC10k(Dataset):
    def __init__(self, path):
        data1,data2,labels = preprocess.load_data("PBMC10k",2000)
        self.x1 = data1.astype(np.float32)
        self.x2 = data2.astype(np.float32)
        self.y = labels.astype(np.int64) 
        print(data1.shape)
        print(data2.shape)
        print(labels.shape)

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], torch.from_numpy(np.array([self.y[idx]])), torch.from_numpy(np.array(idx)).long()
    
class PBMC3k(Dataset):
    def __init__(self, path):
        data1,data2,labels = preprocess.load_data("PBMC3k",2000)
        self.x1 = data1.astype(np.float32)
        self.x2 = data2.astype(np.float32)
        self.y = labels.astype(np.int64) 
        print(data1.shape)
        print(data2.shape)
        print(labels.shape)

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], torch.from_numpy(np.array([self.y[idx]])), torch.from_numpy(np.array(idx)).long()
    
    
class PBMC2k(Dataset):
    def __init__(self, path):
        data1,data2,labels = preprocess.load_data("PBMC2k",2000)
        self.x1 = data1.astype(np.float32)
        self.x2 = data2.astype(np.float32)
        self.y = labels.astype(np.int64) 
        print(data1.shape)
        print(data2.shape)
        print(labels.shape)

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], torch.from_numpy(np.array([self.y[idx]])), torch.from_numpy(np.array(idx)).long()
    
class Spector(Dataset):
    def __init__(self, path):
        data1,data2,labels = preprocess.load_data("PBMC_Spector",2000)
        self.x1 = data1.astype(np.float32)
        self.x2 = data2.astype(np.float32)
        self.y = labels.astype(np.int64) 
        print(data1.shape)
        print(data2.shape)
        print(labels.shape)

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], torch.from_numpy(np.array([self.y[idx]])), torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "GSE100866":
        dataset = GSE100866('./data/')
        dims = [2000,10]  
        view = 2
        data_size = 1182
        class_num = 6
    elif dataset == "GSE128639":
        dataset = GSE128639('./data/')
        dims = [2000,25]  
        view = 2
        data_size = 30672
        class_num = 27
    elif dataset == "PBMC10k":
        dataset = PBMC10k('./data/')
        dims = [2000,2000]     
        view = 2
        data_size = 11020
        class_num = 12
    elif dataset == "PBMC3k":
        dataset = PBMC3k('./data/')
        dims = [2000,2000]  
        view = 2
        data_size = 2585
        class_num = 14
    elif dataset == "PBMC2k":
        dataset = PBMC2k('./data/')
        dims = [2000, 10]  
        view = 2
        data_size = 1181
        class_num = 6
    elif dataset == "PBMC_Spector":
        dataset = Spector('./data/')
        dims = [2000,49]  
        view = 2
        data_size = 3762
        class_num = 16
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
