import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA as skPCA
from sklearn.manifold import TSNE as skTSNE, MDS as skMDS
import umap

from typing import Dict, Union
import warnings

class PCA(nn.Module):
    def __init__(
        self,
        explained_variance_threshold: float = 0.999,
        n_components : Union[int, None] = None
    ):
        """
        PyTorch module that runs sklearn-PCA and chooses the minimum number
        of components needed to reach `explained_variance_threshold` of total variance.

        Args:
            explained_variance_threshold: float in (0, 1], e.g. 0.999 for 99.9%.
            n_components: if set, overrides the explained_variance.
        """
        super().__init__()
        if not (0.0 < explained_variance_threshold <= 1.0):
            raise ValueError("`explained_variance_threshold` must be in (0, 1].")
        self.explained_variance_threshold = explained_variance_threshold
        self.n_components = n_components

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (n_samples, n_features)
        Returns: (n_samples, n_selected_components), 
                 where n_selected_components is the smallest k s.t.
                 cumulative_explained_variance[k-1] ≥ explained_variance_threshold.
        """
        if x.dim() != 2:
            raise ValueError("Input to PCA must be 2D: (n_samples, n_features).")
        x_np = x.detach().cpu().numpy()
        n_samples, n_features = x_np.shape


       # Method is data rank limited so a boundary exists on n_components
        actual_max = min(n_samples, n_features)

        # 2) Fit PCA with all possible components up to actual_max
        #    so we can inspect explained_variance_ratio_ for every k ≤ actual_max.
        pca_full = skPCA(n_components=actual_max)
        pca_full.fit(x_np)  
        evr = pca_full.explained_variance_ratio_  # length = actual_max

        if self.n_components is None:
            # 3) Cumulative explained variance
            cum_evr = np.cumsum(evr)
    
            # 4) Find smallest k such that cum_evr[k-1] ≥ threshold
            self.final_dim = int(np.searchsorted(cum_evr, self.explained_variance_threshold) + 1)
            print(f"{self.final_dim} dimensions hold {self.explained_variance_threshold*100:.2f}% of variance.")
            print(f"Effective compression ratio: {self.final_dim*100/x_np.shape[1]:.2f}%")
            # 5) Take the first n_components = k dimensions
        else:
            # if x_np.shape[0] - 1 < self.n_components:
            #     print(f"Limiting n_components to {x_np.shape[0] - 1}", end='\r')
            #     print(f"Effective compression ratio: {(x_np.shape[0] - 1)*100/x_np.shape[1]:.2f}%", end='\r')
            # else:
            #      print(f"Effective compression ratio: {self.n_components*100/x_np.shape[1]:.2f}%", end='\r')
            n_components = min(x_np.shape[0], self.n_components)
            self.final_dim = n_components
        x_reduced = pca_full.fit_transform(x_np)[:,:self.final_dim]

        self.reducer = pca_full
        return torch.from_numpy(x_reduced).to(x.device)
        
    def transform(self, x: torch.Tensor, batch: bool = False) -> torch.Tensor:
        if batch:
            x = x.detach().cpu().numpy()
        else:
            x = x.detach().cpu().numpy().reshape(1,-1)
        return torch.from_numpy(self.reducer.transform(x)[:,:self.final_dim]).to(x.device)
    
class UMAP(nn.Module):
    def __init__(self, n_components: int, random_state: int = 6262, **umap_kwargs):
        """
        UMAP embedding into a fixed number of dimensions.
        Args:
            n_components: int. Output size
            umap_kwargs: any other arguments you want to pass to `umap.UMAP(...)`.
        """
        super().__init__()
         
        self.n_components = n_components
        self.random_state = random_state
        self.umap_kwargs = umap_kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (n_samples, n_features)
        Returns: (n_samples, n_components)
        """
        warnings.filterwarnings("ignore", "'force_all_finite'")
        x_np = x.detach().cpu().numpy()
        # if x_np.shape[0] - 1 < self.n_components:
        #     print(f"Limiting n_components to {x_np.shape[0] - 1}", end='\r')
        #     print(f"Effective compression ratio: {(x_np.shape[0] -1)*100/x_np.shape[1]:.2f}%", end='\r')
        # else:
        #      print(f"Effective compression ratio: {self.n_components*100/x_np.shape[1]:.2f}%", end='\r')
        n_components = min(x_np.shape[0] - 1, self.n_components)
        reducer = umap.UMAP(n_components=n_components, random_state=self.random_state, **self.umap_kwargs)
        x_reduced = reducer.fit_transform(x_np)
        self.reducer = reducer
        return torch.from_numpy(x_reduced).to(x.device)

    def transform(self, x: torch.Tensor, batch: bool = False) -> torch.Tensor:
        if batch:
            x = x.detach().cpu().numpy()
        else:
            x = x.detach().cpu().numpy().reshape(1,-1)
        return torch.from_numpy(self.reducer.transform(x)).to(x.device)
# class TSNE(nn.Module):
#     def __init__(self, compression_ratio: float = 0.33, random_state: int = 6262,**tsne_kwargs):
#         """
#         TSNE embedding into a fixed number of dimensions.
#         Args:
#             compression_ratio: float [0,1]. The fraction of input size the output will be.
#             tsne_kwargs: any other arguments you want to pass to TSNE(...).
#         """
#         super().__init__()
#         if compression_ratio > 1:
#             raise ValueError("compression_ratio must be < 1")
#         self.compression_ratio = compression_ratio
#         self.random_state = random_state
#         self.tsne_kwargs = tsne_kwargs
       
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: shape (n_samples, n_features)
#         Returns: (n_samples, n_components)
#         """
#         x_np = x.detach().cpu().numpy()
#         n_components = int(x_np.shape[1] * self.compression_ratio)
#         print(f"Size after compression: {n_components}")
#         tsne = skTSNE(n_components=n_components, random_state=self.random_state, **self.tsne_kwargs)
#         x_reduced = tsne.fit_transform(x_np)
#         return torch.from_numpy(x_reduced).to(x.device)


class MDS(nn.Module):
    def __init__(self, n_components: int, **mds_kwargs):
        """
        MDS embedding into a fixed number of dimensions.
        Args:
            n_components: int. Output size
            mds_kwargs: any other arguments you want to pass to MDS(...).
        """
        super().__init__()
         
        self.n_components = n_components
        self.mds_kwargs = mds_kwargs

    def forward(self, x: torch.Tensor, ind2patient: Dict) -> torch.Tensor:
        """
        Expects x: shape (n_samples, n_features). MDS uses Euclidean distances.
        Returns: (n_samples, n_components)
        """
        x_np = x.detach().cpu().numpy()
        # if x_np.shape[0] - 1 < self.n_components:
        #     print(f"Limiting n_components to {x_np.shape[0] - 1}", end='\r')
        #     print(f"Effective compression ratio: {(x_np.shape[0] -1)*100/x_np.shape[1]:.2f}%", end='\r')
        # else:
        #     print(f"Effective compression ratio: {self.n_components*100/x_np.shape[1]:.2f}%", end='\r')
        n_components = min(x_np.shape[0] - 1, self.n_components)
        mds = skMDS(n_components=n_components, **self.mds_kwargs)
        x_reduced = mds.fit_transform(x_np)
        self.mapping = {ind2patient[idx]:torch.from_numpy(x_reduced[idx,:]).to(x.device) for idx in ind2patient.keys()}
        return torch.from_numpy(x_reduced).to(x.device), self.mapping
