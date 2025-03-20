"""Script containing functions to visualize Visium data."""

# import all necessary packages
from typing import Dict, List, Optional, Union

import anndata as ad
import scanpy as sc
import tiffslide
import pandas as pd
import boto3
import io


from gbmhackathon.utils.visium_functions import convert_obsm_to_adata
from gbmhackathon.definitions import MOSAIC_BUCKET


def plot_spatial_expression(
    anndata_dict: Dict[str, ad.AnnData],
    gene_list: List[str],
    layer: str = "log_CPM",
    sample_list: Optional[List[str]] = None,
    save_output: Optional[str] = None,
    **kwargs: Optional[Dict],
) -> None:
    """
    Generate heatmaps of gene expression for selected samples and genes using scanpy's spatial function.

    Parameters
    ----------
        anndata_dict (Dict[str, ad.AnnData]): Dictionary of AnnData objects.
        gene_list (List[str]): List of genes to visualize.
        sample_list (Optional[List[str]]): List of sample IDs to visualize. If None, visualize all samples.

    Returns
    -------
        None: Displays the heatmaps.
    """
    samples_to_plot = (
        sample_list if sample_list is not None else list(anndata_dict.keys())
    )

    for sample_id in samples_to_plot:
        if sample_id not in anndata_dict:
            print(f"Sample {sample_id} not found in the dictionary.")
            continue

        adata = anndata_dict[sample_id]
        adata.X = adata.layers[layer]

        for gene in gene_list:
            if gene not in adata.var_names:
                print(f"Gene {gene} not found in sample {sample_id}.")
                continue

            try:
                if save_output is not None:
                    sc.pl.spatial(
                        adata,
                        color=gene,
                        title=f"{gene} expression in {sample_id}",
                        save=f"{sample_id}_{gene}_{save_output}",
                        show=False,
                        **kwargs,
                    )
                else:
                    sc.pl.spatial(
                        adata,
                        color=gene,
                        title=f"{gene} expression in {sample_id}",
                        show=True,
                        **kwargs,
                    )
            except Exception as e:
                print(f"Failed to plot gene {gene} for sample {sample_id}: {e}")


def plot_obsm(
    adat: ad.AnnData,
    obsm_slot: str,
    features: Union[list[str], None] = None,
    show: bool = True,
    save_output: Optional[str] = None,
    **kwargs: dict,
):
    """Plot obsm matrix as spatial plot.

    Parameters
    ----------
    adat: ad.AnnData
        Input Anndata object to use as input for plot
    obsm_slot : str
        Name of the obsm matrix to plot.
    features : Union[list[str], None]
        List of features to plot.
    show : bool
        Show the plot, by default True.
    **kwargs : dict
        Arguments to pass to scanpy.pl.spatial.
    """
    obsm_ad = convert_obsm_to_adata(adat, obsm_slot)
    if features is None:
        features = obsm_ad.var.index
    if save_output is not None:
        sc.pl.spatial(
            obsm_ad,
            color=features,
            save=f"{list(adat.uns['spatial'].keys())[0]}_{save_output}",
            show=show,
            **kwargs,
        )
    else:
        sc.pl.spatial(obsm_ad, color=features, show=show, **kwargs)


def get_slide_img(
    slide_paths: pd.DataFrame, slide_idx: int, bucket: str = MOSAIC_BUCKET
):
    s3 = boto3.client("s3")
    slide_path = str(slide_paths["path"].iloc[slide_idx])
    buffer = io.BytesIO(s3.get_object(Bucket=bucket, Key=slide_path)["Body"].read())
    with buffer as file_obj:
        slide = tiffslide.TiffSlide(file_obj)
        return slide.get_thumbnail(slide.level_dimensions[-2])
