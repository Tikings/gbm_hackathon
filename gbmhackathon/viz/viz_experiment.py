from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from umap import UMAP
import pandas as pd
import seaborn as sns
import numpy as np
import torch

from typing import Union, Optional, List, Dict, Tuple
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from PIL import Image


def visualize_embeddings(
    embeddings: Union[torch.Tensor, np.ndarray],
    method: str = 'pca',
    labels: Optional[List] = None,
    modality_info: Optional[Dict[int, str]] = None,
    patient_info: Optional[Dict[int, int]] = None,
    figsize: Tuple[int, int] = (12, 10),
    title: Optional[str] = None,
    show_legend: bool = True,
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    return_fig: bool = False
) -> Optional[plt.Figure]:
    """
    Visualize embeddings using dimensionality reduction techniques (PCA, t-SNE, MDS, or UMAP).

    Parameters:
    -----------
    embeddings : torch.Tensor or numpy.ndarray
        The embeddings to visualize. Shape should be [n_samples, n_features].
    method : str, default='pca'
        Dimensionality reduction method: 'pca', 'tsne', 'mds', or 'umap'.
    labels : list, optional
        Labels for each embedding point. If provided, used for coloring.
    modality_info : dict, optional
        Mapping from sample index to modality name for modality-based coloring.
    patient_info : dict, optional
        Mapping from sample index to patient ID for patient-based coloring.
    figsize : tuple, default=(12,10)
        Size of the figure.
    title : str, optional
        Plot title. Defaults to method-based title.
    show_legend : bool, default=True
        Whether to display legend.
    colors : list, optional
        Color palette to use.
    markers : list, optional
        Marker styles to use.
    save_path : str, optional
        Path to save the figure. If provided, figure is saved instead of shown.
    return_fig : bool, default=False
        If True, returns the matplotlib Figure object instead of showing/saving.

    Returns:
    --------
    plt.Figure or None
        The figure object if return_fig=True, else None.
    """
    # Convert to numpy
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape {embeddings.shape}")
    n_samples, _ = embeddings.shape

    # Select reducer
    method_l = method.lower()
    if method_l == 'pca':
        reducer = PCA(n_components=2)
        if title is None:
            title = f'PCA of {n_samples} samples'
    elif method_l == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(30, n_samples-1), random_state=42)
        if title is None:
            title = f't-SNE of {n_samples} samples'
    elif method_l == 'mds':
        reducer = MDS(n_components=2, random_state=42)
        if title is None:
            title = f'MDS of {n_samples} samples'
    elif method_l == 'umap':
        reducer = UMAP(n_components=2, min_dist=0.1 if n_samples>100 else 0.5,
                       n_neighbors=min(15, n_samples-1), random_state=42)
        if title is None:
            title = f'UMAP of {n_samples} samples'
    else:
        raise ValueError(f"Unknown method {method}. Choose 'pca','tsne','mds', or 'umap'.")

    # Remove near-zero variance features
    vars_ = np.var(embeddings, axis=0)
    if np.any(vars_ < 1e-10):
        embeddings = embeddings[:, vars_ >= 1e-10]

    # Reduce
    try:
        emb2d = reducer.fit_transform(embeddings)
    except Exception:
        emb2d = reducer.fit_transform(embeddings + np.random.normal(0,1e-5, embeddings.shape))

    # Setup figure
    fig, ax = plt.subplots(figsize=figsize)
    if colors is None:
        colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    if markers is None:
        markers = ['o','s','^','D','v','<','>','p','*','h','H','+','x','|','_']

    # Plot data
    if modality_info is not None:
        # color by modality
        unique_mods = sorted(set(modality_info.values()))
        for i, mod in enumerate(unique_mods):
            idxs = [i0 for i0,m in modality_info.items() if m==mod]
            ax.scatter(emb2d[idxs,0], emb2d[idxs,1],
                       c=colors[i%len(colors)], marker=markers[0],
                       label=mod, alpha=0.7, s=70)
    elif patient_info is not None:
        unique_pats = sorted(set(patient_info.values()))
        for i, pat in enumerate(unique_pats):
            idxs = [i0 for i0,p in patient_info.items() if p==pat]
            ax.scatter(emb2d[idxs,0], emb2d[idxs,1],
                       c=colors[i%len(colors)], marker=markers[0],
                       label=str(pat), alpha=0.7, s=70)
    elif labels is not None:
        unique_lbls = sorted(set(labels))
        for i, lbl in enumerate(unique_lbls):
            idxs = [i0 for i0,l in enumerate(labels) if l==lbl]
            ax.scatter(emb2d[idxs,0], emb2d[idxs,1],
                       c=colors[i%len(colors)], marker=markers[0],
                       label=str(lbl), alpha=0.7, s=70)
    else:
        ax.scatter(emb2d[:,0], emb2d[:,1], c=colors[0], alpha=0.7, s=70)

    # Labels, title, grid
    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    # ax.set_ylim(-10,10)
    # ax.set_xlim(-5,5)
    ax.grid(True, linestyle='--', alpha=0.5)
    if show_legend and (modality_info or patient_info or labels):
        ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()

    # Save or show
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if return_fig:
        return fig
    else:
        plt.show()
        return None


def make_matplotlib_gif(frame_dir: str,
                        out_path: str,
                        fps: int = 2,
                        dpi: int = 150):
    """
    Assemble PNG frames in `frame_dir` into an animated GIF or MP4
    using Matplotlib's animation API.

    Parameters:
    -----------
    frame_dir : str
        Directory containing epoch_###.png files
    out_path : str
        Output path, ending in .gif or .mp4
    fps : int
        Frames per second
    dpi : int
        Resolution of saved animation
    """
    # 1) List frames in epoch order
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith('.png'))
    if not files:
        raise ValueError(f"No .png files found in {frame_dir}")

    # 2) Create figure
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')  # hide axes

    # 3) Load images
    frames = []
    for fname in files:
        img = Image.open(os.path.join(frame_dir, fname))
        im = plt.imshow(img, animated=True)
        frames.append([im])

    # 4) Build animation
    ani = animation.ArtistAnimation(
        fig, frames,
        interval=1000/fps,   # milliseconds between frames
        blit=True,
        repeat_delay=1000    # pause at end
    )

    # 5) Choose writer based on extension
    ext = os.path.splitext(out_path)[1].lower()
    if ext == '.gif':
        # Requires ImageMagick or PillowWriter
        ani.save(out_path, writer='pillow', dpi=dpi)
    else:
        # MP4 via ffmpeg
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(out_path, writer=writer, dpi=dpi)

    plt.close(fig)
    print(f"Saved animation to {out_path}")
    
def see_emb(batch_all=None, gbmnet=None, reducer='tsne', patient_embs=None):
    if patient_embs is None:
        # Activate inference mode
        gbmnet.eval()
        out_dict, _ = gbmnet(batch_all[2])
        mme_embs, sep_mod_embs = [], []
        mod_info, patient_info = {}, {}
        idx = 0
        for pid in range(out_dict[list(out_dict.keys())[0]].size(0)):
            tensors = [out_dict[mod][pid] for mod in out_dict.keys()]
            mme_embs.append(torch.cat(tensors, dim=0).unsqueeze(0))
            sep_mod_embs += [t.unsqueeze(0) for t in tensors]
            for mod in out_dict.keys():
                mod_info[idx], patient_info[idx] = mod, pid
                idx += 1
    
        patient_embs = torch.cat(mme_embs, dim=0)
        sep_mod_embs = torch.cat(sep_mod_embs, dim=0)
    
    figs = []
    figs.append(visualize_embeddings(patient_embs, reducer, return_fig=True))
    if patient_embs is None:
        figs.append(visualize_embeddings(sep_mod_embs, reducer, modality_info=mod_info, return_fig=True))
        figs.append(visualize_embeddings(sep_mod_embs, reducer, modality_info=patient_info, return_fig=True))
    return figs