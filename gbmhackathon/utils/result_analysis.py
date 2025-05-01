from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import torch




def visualize_embeddings(embeddings: Union[torch.Tensor, np.ndarray], 
                         method: str = 'pca',
                         labels: Optional[List] = None,
                         modality_info: Optional[Dict[str, int]] = None,
                         patient_info: Optional[Dict[str, int]] = None,
                         figsize: Tuple[int, int] = (12, 10),
                         title: Optional[str] = None,
                         show_legend: bool = True,
                         colors: Optional[List[str]] = None,
                         markers: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Visualize embeddings using dimensionality reduction techniques (PCA, t-SNE, or UMAP).
    
    Parameters:
    -----------
    embeddings : torch.Tensor or numpy.ndarray
        The embeddings to visualize. Shape should be [n_samples, n_features].
    method : str, default='pca'
        The dimensionality reduction method to use ('pca', 'tsne', or 'umap').
    labels : list, optional
        Labels for each embedding point. If None, no color-coding.
    modality_info : dict, optional
        Dictionary mapping modality names to indices for color-coding by modality.
    patient_info : dict, optional
        Dictionary mapping patient IDs to indices for grouping by patient.
    figsize : tuple, default=(12, 10)
        Figure size as (width, height) in inches.
    title : str, optional
        Title for the plot. If None, a default title based on the method is used.
    show_legend : bool, default=True
        Whether to show the legend.
    colors : list, optional
        List of colors to use for different categories. If None, defaults are used.
    markers : list, optional
        List of markers to use. If None, defaults are used.
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed but not saved.
    
    Returns:
    --------
    None
        The function displays or saves the visualization.
    """
    # Convert torch tensor to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # Input validation
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape {embeddings.shape}")
    
    n_samples, n_features = embeddings.shape
    
    # Choose dimensionality reduction method
    if method.lower() == 'pca':
        if title is None:
            title = f'PCA Visualization of {n_samples} Embeddings'
        reducer = PCA(n_components=2)
    elif method.lower() == 'tsne':
        if title is None:
            title = f't-SNE Visualization of {n_samples} Embeddings'
        reducer = TSNE(n_components=2, perplexity=min(30, n_samples-1), random_state=42)
    elif method.lower() == 'umap':
        if title is None:
            title = f'UMAP Visualization of {n_samples} Embeddings'
        min_dist = 0.1 if n_samples > 100 else 0.5
        reducer = UMAP(n_components=2, min_dist=min_dist, n_neighbors=min(15, n_samples-1), random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'pca', 'tsne', or 'umap'.")
    
    # Check for zero variance features that could cause problems
    feature_vars = np.var(embeddings, axis=0)
    if np.any(feature_vars < 1e-10):
        print(f"Warning: {np.sum(feature_vars < 1e-10)} features have near-zero variance.")
        # Remove zero variance features
        non_zero_idx = feature_vars >= 1e-10
        embeddings = embeddings[:, non_zero_idx]
    
    # Apply dimensionality reduction
    try:
        embeddings_2d = reducer.fit_transform(embeddings)
    except Exception as e:
        print(f"Error during dimensionality reduction: {e}")
        # Add a small amount of noise if we have issues
        embeddings = embeddings + np.random.normal(0, 1e-5, embeddings.shape)
        embeddings_2d = reducer.fit_transform(embeddings)
    
    # Prepare visualization
    plt.figure(figsize=figsize)
    
    # Set default colors and markers
    if colors is None:
        colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
    
    # Plot based on the available information
    if modality_info and patient_info:
        # Group by both modality and patient
        modalities = list(modality_info.keys())
        patients = list(patient_info.keys())
        
        # Create a scatter plot for each modality-patient combination
        for p_idx, patient in enumerate(patients):
            for m_idx, modality in enumerate(modalities):
                mask = (np.array(labels) == f"{patient}_{modality}")
                if np.any(mask):
                    plt.scatter(
                        embeddings_2d[mask, 0], 
                        embeddings_2d[mask, 1],
                        c=colors[m_idx % len(colors)], 
                        marker=markers[p_idx % len(markers)],
                        label=f"{patient} - {modality}" if np.sum(mask) > 0 else None,
                        alpha=0.7,
                        s=70
                    )
    elif modality_info:
        # Group by modality only
        modalities = list(modality_info.keys())
        for m_idx, modality in enumerate(modalities):
            if labels:
                mask = np.array([label.endswith(modality) if isinstance(label, str) else False for label in labels])
            else:
                # Assume embeddings are ordered by modality if no labels
                start_idx = modality_info[modality]
                end_idx = modality_info[list(modality_info.keys())[m_idx+1]] if m_idx+1 < len(modalities) else n_samples
                mask = np.zeros(n_samples, dtype=bool)
                mask[start_idx:end_idx] = True
            
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=colors[m_idx % len(colors)], 
                label=modality if np.sum(mask) > 0 else None,
                alpha=0.7,
                s=70
            )
    elif patient_info:
        # Group by patient only
        patients = list(patient_info.keys())
        for p_idx, patient in enumerate(patients):
            if labels:
                mask = np.array([label.startswith(patient) if isinstance(label, str) else False for label in labels])
            else:
                # Assume embeddings are ordered by patient if no labels
                start_idx = patient_info[patient]
                end_idx = patient_info[list(patient_info.keys())[p_idx+1]] if p_idx+1 < len(patients) else n_samples
                mask = np.zeros(n_samples, dtype=bool)
                mask[start_idx:end_idx] = True
            
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=colors[p_idx % len(colors)], 
                label=patient if np.sum(mask) > 0 else None,
                alpha=0.7,
                s=70
            )
    elif labels is not None:
        # Use provided labels for coloring
        unique_labels = sorted(set(labels))
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=colors[i % len(colors)], 
                label=label if np.sum(mask) > 0 else None,
                alpha=0.7,
                s=70
            )
    else:
        # No grouping information, plot all points with same style
        plt.scatter(
            embeddings_2d[:, 0], 
            embeddings_2d[:, 1],
            c=colors[0], 
            alpha=0.7,
            s=70
        )
    
    # Add metadata to the plot
    plt.title(title, fontsize=14)
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Compute and display explained variance for PCA
    if method.lower() == 'pca' and isinstance(reducer, PCA):
        explained_var = reducer.explained_variance_ratio_
        plt.xlabel(f'PC1 ({explained_var[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({explained_var[1]:.2%} variance)', fontsize=12)
        
        # Add text about total explained variance
        total_var = sum(explained_var)
        plt.figtext(0.5, 0.01, f'Total explained variance: {total_var:.2%}', 
                    ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add zero ratio information
    zero_ratio = (np.abs(embeddings) < 1e-6).sum() / embeddings.size
    plt.figtext(0.01, 0.01, f'Zero ratio: {zero_ratio:.2%}', 
                fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Show legend if requested
    if show_legend and (modality_info or patient_info or labels is not None):
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()

    fig=plt.gcf()
    return fig