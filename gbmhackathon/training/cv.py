import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import ShuffleSplit

def get_mccv_loaders(dataset, 
                     collate_fn,
                     n_splits, 
                     test_size=0.8, 
                     train_batch_size=32, 
                     val_batch_size=128, 
                     device='cpu', 
                     random_state=6262):
    """
    Returns loaders designed to perform Monte Carlo Cross Validation (MCCV).
    """
    indices = list(dataset.ind2patient.keys())
    indices_arr = np.zeros((len(indices),1))
    
    mccv =  ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    for train_idx, val_idx in mccv.split(indices_arr):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=val_batch_size, collate_fn=collate_fn, shuffle=False)
        yield train_loader, val_loader