import scanpy as sc
import sys
import numpy as np
import fireducks as pd
import anndata as ad

# Load packages and classes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tiffslide
import gget
from Ensembl_converter import EnsemblConverter
import tifffile
import zarr
import datetime
import argparse

import torch
import gc
import random as rd
import os, tqdm
import subprocess


# MosaicDataset and BruceDataset classes allow loading and visualisation of the different data sources
from gbmhackathon import MosaicDataset
from gbmhackathon.data.io.loaders import SingleCellLoader
from gbmhackathon.s3_loader import write_s3

# Helper functions
def remove_duplicate_var_indices(data: ad.AnnData | pd.DataFrame, mode: str) -> ad.AnnData | pd.DataFrame:
    print(f"\nDEBUG: Entering remove_duplicate_var_indices with mode: {mode}")
    # Extract the var index
    if mode == 'scRNA':
        var_index = data.var.index
        print("DEBUG: Mode is scRNA, extracted data.var.index")
    elif mode == 'bulk':
        var_index = data.index
        print("DEBUG: Mode is bulk, extracted data.index")
    else:
        raise ValueError(f"Wrong value for mode argument, either 'scRNA' or 'bulk', received : {mode}")

    # Find duplicates
    duplicates = var_index[var_index.duplicated(keep='first')]
    print(f"DEBUG: Found {len(duplicates)} duplicate variable indices.")

    # Drop duplicates
    data = data[:, ~var_index.isin(duplicates)]
    print(f"DEBUG: Shape of data after removing duplicates: {data.shape}")

    # Optionally print the number of removed variables
    print(f"Removed {len(duplicates)} variables with duplicate indices.")

    return data

def replace_nan_var_indices(data, mode):
    print(f"\nDEBUG: Entering replace_nan_var_indices with mode: {mode}")
    # Replace NaN in the var index with a placeholder
    if mode == "scRNA":
        print("DEBUG: Mode is scRNA, filling NaN in data.var.index")
        data.var.index = data.var.index.fillna('unknown')
    elif mode == 'bulk':
        print("DEBUG: Mode is bulk, filling NaN in data.index")
        data.index = data.index.fillna('unknown')
    else:
        raise ValueError(f"Wrong value for mode argument, either 'scRNA' or 'bulk', received : {mode}")
    # Ensure indices are unique after replacement
    if mode == "scRNA":
        if not data.var.index.is_unique:
            raise ValueError("Indices are still not unique after replacing NaN.")
        print("DEBUG: data.var.index is unique after filling NaN.")
    elif mode == 'bulk':
        if not data.index.is_unique:
            raise ValueError("Indices are still not unique after replacing NaN.")
        print("DEBUG: data.index is unique after filling NaN.")

    return data

def prepare_adata_for_selection(adata):
    print("\nDEBUG: Entering prepare_adata_for_selection")
    # Convert the AnnData matrix to a DataFrame
    X_df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )
    print(f"DEBUG: Converted AnnData to DataFrame with shape: {X_df.shape}")
    return X_df

def main_gene_selection(X_df, gene_list):
    print(f"\nDEBUG: Entering main_gene_selection with X_df shape: {X_df.shape} and gene_list length: {len(gene_list)}")
    """
    Describe:
        rebuild the input adata to select target genes encode protein
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    print(f"DEBUG: Number of genes to fill: {len(to_fill_columns)}")
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))),
                                 columns=to_fill_columns,
                                 index=X_df.index)
    print(f"DEBUG: Created padding DataFrame with shape: {padding_df.shape}")
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1),
                          index=X_df.index,
                          columns=list(X_df.columns) + list(padding_df.columns))
    print(f"DEBUG: Concatenated DataFrame shape: {X_df.shape}")
    X_df = X_df[gene_list]
    print(f"DEBUG: Selected genes, new DataFrame shape: {X_df.shape}")

    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    print(f"DEBUG: Created var DataFrame with shape: {var.shape}")
    return X_df, to_fill_columns, var

def create_folds(id_list, batch_size):
    print(f"\nDEBUG: Entering create_folds with id_list length: {len(id_list)} and batch_size: {batch_size}")
    n_batch = int(np.ceil(len(id_list) / batch_size))
    print(f"DEBUG: Number of batches: {n_batch}")
    id_set = set(id_list)
    memory = set()
    folds = []
    for batch in range(n_batch-1):
        fold_ids = rd.sample(list(id_set.difference(memory)), k=batch_size)
        folds.append(fold_ids)
        memory = memory.union(set(fold_ids))
        print(f"DEBUG: Created fold {batch+1} with {len(fold_ids)} IDs.")
    #Last batch
    remaining = list(id_set.difference(memory))
    remaining = rd.sample(remaining, k=len(remaining)) # Corrected shuffle
    folds.append(remaining)
    print(f"DEBUG: Created last fold with {len(remaining)} IDs.")
    print(f"DEBUG: Total number of folds created: {len(folds)}")
    return folds

def delete_dir(dir_path):
    print(f"\nDEBUG: Entering delete_dir with path: {dir_path}")
    # List all files in the directory
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)

            # Check if it is a file (not a subdirectory)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove the file
                print(f"Deleted file: {filename}")
        print(f"DEBUG: Finished deleting files in {dir_path}")
    else:
        print(f"DEBUG: Directory {dir_path} does not exist, skipping deletion.")

def infer_embeddings(files, mode):
    print(f"DEBUG: Entering infer_embeddings with {len(files)} files and mode: {mode}")
    """
    Run the embedding inference script on each chunk file using subprocess,
    ensuring the correct conda environment is used.

    Parameters:
    -----------
    files : list
        List of paths to chunk files to process
    mode: str
        String indicating either "scRNA" or "bulk"
    """
    # Method 1: Use the current Python executable path
    # This ensures we use the same Python that has scanpy installed
    python_executable = sys.executable
    print(f"\nDEBUG: Using Python executable: {python_executable}")
    if mode == "scRNA":
        input_type = "singlecell"
        print("DEBUG: Mode is scRNA, setting input_type to singlecell")
    elif mode == "bulk":
        input_type = "bulk"
        print("DEBUG: Mode is bulk, setting input_type to bulk")
    else:
        raise ValueError(f"Wrong value for mode argument, either 'scRNA' or 'bulk', received : {mode}")
    for i, chunk_path in enumerate(tqdm.tqdm(files, desc=f"Processing {mode} embeddings")):
        # Create the command using the correct Python path
        cmd = [
            python_executable,  # Use the current Python executable
            "../scFoundation/model/get_embedding.py",
            "--task_name", f"{mode}_chunk{i}_Inference",
            "--input_type", input_type,
            "--output_type", "cell",
            "--pool_type", "all"]
        if mode == "scRNA":
            cmd += ["--tgthighres", "a5"]
        cmd += [
            "--data_path", chunk_path,
            "--save_path", "./"]
        if mode == "scRNA":
            cmd += ["--pre_normalized", "T"] # for scRNA T = already normalized+log1p
        else:
            cmd += ["--pre_normalized", "F"] # for bulk F = Compute T and S tokens using log10(sum of expression)]

        cmd += ["--version", "rde"]
        print(f"DEBUG: Running subprocess command: {cmd}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=True
            )
            print(f"Successfully processed {chunk_path}")
            print(f"DEBUG: Subprocess stdout:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {chunk_path}: {e}")
            print(f"DEBUG: Subprocess stderr:\n{e.stderr}")

def create_chunks(mode: str, data: ad.AnnData | pd.DataFrame, gene_list: list, folder: str = "input_chunks", batch_size: int = 6000):
    print(f"\nDEBUG: Entering create_chunks with mode: {mode}, data shape: {data.shape}, gene_list length: {len(gene_list)}, folder: {folder}, batch_size: {batch_size}")
    if mode == 'scRNA':
        FOLDS = create_folds(list(data.obs_names), batch_size)
        print(f"DEBUG: Created {len(FOLDS)} folds for scRNA data.")
    else:
        FOLDS = create_folds(list(data.index), batch_size)
        print(f"DEBUG: Created {len(FOLDS)} folds for bulk data.")
    n_batch = 2 #len(FOLDS) # Process all folds

    files = []
    if folder in os.listdir():
        delete_dir(folder)
    # Create the output directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    print(f"DEBUG: Created output directory: {folder}")
    for batch in tqdm.tqdm(range(n_batch), desc="Creating data chunks"):
        print(f"DEBUG: Processing batch {batch+1}/{n_batch}")
        if mode == 'scRNA':
            chunked_data, _, _ = main_gene_selection(prepare_adata_for_selection(data[FOLDS[batch],:]), gene_list)
            print(f"DEBUG: Shape of chunked scRNA data: {chunked_data.shape}")
        else:
            chunked_data, _, _ = main_gene_selection(data.loc[FOLDS[batch],:], gene_list)
            print(f"DEBUG: Shape of chunked bulk data: {chunked_data.shape}")
        filename = f'{folder}/preprocessed_chunk{batch}.csv'
        chunked_data.to_csv(filename)
        files.append(filename)
        print(f"DEBUG: Saved chunk to: {filename}")
    print(f"DEBUG: Created {len(files)} chunk files.")
    return FOLDS, files

def transfer_obs(original_adata_obs, subsampled_adata):
    print("\nDEBUG: Entering transfer_obs")
    common_indices = subsampled_adata.obs.index.intersection(original_adata_obs.index)
    print(f"DEBUG: Number of common indices: {len(common_indices)}")
    subsampled_adata.obs = original_adata_obs.loc[common_indices].copy()
    print(f"DEBUG: Transferred {len(subsampled_adata.obs)} observations.")
    return subsampled_adata

def construct_final_emb(folds, adata_obs=None):
    embedding_files = [file for file in os.listdir() if file.endswith("embedding_a5_resolution.npy")]
    print(f"DEBUG: Found {len(embedding_files)} embedding files.")
    
    refdf_index = []
    for batch in range(len(embedding_files)):
        refdf_index += folds[batch]
    print(f"DEBUG: Retrieved {len(refdf_index)} references indices.")
    print(f"\nDEBUG: Entering construct_final_emb with {len(refdf_index)} reference indices and adata_obs present: {adata_obs is not None}")
    
    arrays = [np.load(file) for file in tqdm.tqdm(embedding_files, desc="Loading embeddings")]

    imputemb = np.concatenate(arrays, axis=0)
    print(f"DEBUG: Concatenated embedding array shape: {imputemb.shape}")

    if adata_obs is None: # Mode is bulk
        print("DEBUG: adata_obs is None, assuming bulk mode.")
        rdy_emb = pd.DataFrame(imputemb, index=refdf_index)
        print(f"DEBUG: Final bulk embedding DataFrame shape: {rdy_emb.shape}")
    else: # Mode is scRNA
        print("DEBUG: adata_obs is present, assuming scRNA mode.")
        imputeAdata = sc.AnnData(pd.DataFrame(imputemb, index=refdf_index))
        print(f"DEBUG: Initial imputeAdata shape: {imputeAdata.shape}")
        sc.pp.scale(imputeAdata)
        print("DEBUG: Scaled imputeAdata.")
        sc.tl.pca(imputeAdata)
        print("DEBUG: Performed PCA on imputeAdata.")
        sc.pp.neighbors(imputeAdata)
        print("DEBUG: Calculated neighbors for imputeAdata.")
        sc.tl.umap(imputeAdata)
        print("DEBUG: Performed UMAP on imputeAdata.")
        imputeAdata = transfer_obs(adata_obs, imputeAdata)
        print(f"DEBUG: Shape of imputeAdata after transferring obs: {imputeAdata.shape}")
        df = pd.DataFrame(imputeAdata.X)#.select_dtypes(include = ['number'])
        df['orig.ident'] = imputeAdata.obs.reset_index()['orig.ident']
        print(f"DEBUG: Created embedding DataFrame with shape: {df.shape}")

        # Aggregation using Mean
        rdy_emb = df.groupby('orig.ident').mean()
        print(f"DEBUG: Final scRNA embedding DataFrame shape after aggregation: {rdy_emb.shape}")
    return rdy_emb

def get_data(mode: str) -> ad.AnnData | pd.DataFrame:
    print(f"\nDEBUG: Entering get_data with mode: {mode}")
    # Note that it can take up to 12 minutes to load the single-cell data because it is heavy
    if mode == "scRNA":
        print("DEBUG: Loading single-cell data...")
        data = MosaicDataset.load_singlecell("pa-3dqtp2dd4t56b7jvg-bx3h881cqf4ushomn6mknw3cstocaeuw1b-s3alias")
        print(f"DEBUG: Loaded single-cell data with shape: {data.shape}")
        return data
    elif mode == "bulk":
        print("DEBUG: Loading bulk RNA data...")
        data = MosaicDataset.load_tabular()["bulk_rna"]['TPM counts']
        ensembl_ids = data.index.astype(str).tolist() # retrieve gene IDS
        
        data = data.T # to have patients as rows and genes as columns
        print(f"DEBUG: First 5 Ensembl IDs: {ensembl_ids[:5]}")
        print(f"DEBUG: Total number of Ensembl IDs: {len(ensembl_ids)}")
        # Create an instance of EnsemblConverter
        converter = EnsemblConverter(use_progress_bar=True)
        
        batch_size = 100
        all_gene_names = []
        for i in tqdm.tqdm(range(0, len(ensembl_ids), batch_size), desc="Querying gget in batches"):
            batch_ids = ensembl_ids[i:i + batch_size]
            print(f"DEBUG: Querying gget for Ensembl IDs batch {i//batch_size + 1} of size {len(batch_ids)}")
            try:
                # Convert Ensembl IDs to gene symbols
                result = list(converter.convert_ids(batch_ids)["Symbol"])
                # print(result)
                if result is not None:
                    all_gene_names.extend(result)
                    print(f"DEBUG: Received {len(result)} gene names for this batch.")
                else:
                    print(f"DEBUG: WARNING: ensembl converter did not return gene names for this batch. Filling with original IDs.")
                    all_gene_names.extend(batch_ids)
            except Exception as e:
                print(f"DEBUG: ERROR during ensembl converter for batch {i//batch_size + 1}: {e}")
                print("DEBUG: Filling this batch with original Ensembl IDs.")
                all_gene_names.extend(batch_ids)

        if len(ensembl_ids) == len(all_gene_names):
            data.index = all_gene_names
            print(f"DEBUG: Successfully mapped {len(ensembl_ids)} Ensembl IDs to gene names using batched queries.")
        else:
            print(f"DEBUG: WARNING: Number of input IDs ({len(ensembl_ids)}) does not match the number of gene names returned by batched gget ({len(all_gene_names)}).")
            print("DEBUG: Using original Ensembl IDs as index.")

        print(f"DEBUG: Loaded bulk RNA data with shape: {data.shape} and index: {data.index[:5].tolist()}")
        return data
    else:
        raise ValueError(f"Wrong mode provided: {mode}. Expected 'scRNA' or 'bulk'.")
        
def prepare_data(mode) -> tuple[list[str], list[str], pd.DataFrame | None]:
    print(f"\nDEBUG: Entering prepare_data with mode: {mode}")
    data = get_data(mode)
    print(f"DEBUG: Data loaded with shape: {data.shape}")
    data = remove_duplicate_var_indices(data, mode)
    print(f"DEBUG: Data after removing duplicate indices shape: {data.shape}")
    data = replace_nan_var_indices(data, mode)
    print(f"DEBUG: Data after replacing NaN indices shape: {data.shape}")

    adata_obs = None
    if mode == 'scRNA':
        adata_obs = data.obs
        print(f"DEBUG: Extracted adata.obs with shape: {adata_obs.shape}")
    else:
        print("DEBUG: Mode is bulk, adata_obs is set to None.")

    print("DEBUG: Reading gene list from ../scFoundation/OS_scRNA_gene_index.19264.tsv")
    gene_list_df = pd.read_csv('../scFoundation/OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])
    print(f"DEBUG: Loaded gene list with {len(gene_list)} genes.")

    print(f"DEBUG: Creating chunks with mode: {mode}, data shape: {data.shape}, gene list length: {len(gene_list)}")
    folds, files = create_chunks(mode, data, gene_list, folder="input_chunks", batch_size=6000)
    print(f"DEBUG: Created {len(files)} chunk files.")
    print(f"DEBUG: Number of folds created: {len(folds)}")
    del data
    print(f"DEBUG: Deleted data")
    gc.collect()
    print("DEBUG: Requested garbage collection")
    return files, folds, adata_obs

def pipeline(mode: str, save: bool = False, save_name: str = "pipeline_emb_V1", folder: str = "embedding_V1") -> dict | None:
    print(f"\nDEBUG: Entering pipeline with mode: {mode}, save: {save}, save_name: {save_name}, folder: {folder}")
    files, folds, data_obs = prepare_data(mode)
    print(f"DEBUG: prepare_data returned {len(files)} files, {len(folds)} reference indices, and adata_obs is {'not None' if data_obs is not None else 'None'}.")

    infer_embeddings(files, mode)
    print("DEBUG: infer_embeddings completed.")

    rdy_emb_df = construct_final_emb(folds, adata_obs = data_obs)
    print(f"DEBUG: construct_final_emb returned embedding DataFrame with shape: {rdy_emb_df.shape if isinstance(rdy_emb_df, pd.DataFrame) else 'not a DataFrame'}.")

    settings = {
        "pipeline":f"{mode} Embedding Pipeline",
        "model": "scFoundation",
        "date": datetime.datetime.now().isoformat(),
    }
    data = {
        patient_id: torch.tensor(rdy_emb_df.loc[patient_id].values)
        for patient_id in rdy_emb_df.index
    }
    print(f"DEBUG: Created output data dictionary with {len(data)} entries.")

    output = {"settings": settings, "data": data}
    print(f"DEBUG: Created output dictionary with settings and data.")
    if save:
        write_s3(obj=output, save_name=save_name, folder=folder)
        print("DEBUG: Output saved.")
    return output

if __name__ == "__main__":
    
    print("\n[!] WARNING [!]")
    print("Before running this pipeline you MUST do the installation of scFoundation, there are several steps needed, check the notebook analysis/pipeline_scRNA.ipynb and the file local_attention in it")
    print("[!] WARNING [!]")
    
    parser = argparse.ArgumentParser(description="scRNA or Bulk pipeline")
    parser.add_argument("--mode", type=str, help="scRNA or bulk mode")
    parser.add_argument("--save_results", type=int, help="Save or not (1 or 0)")
    parser.add_argument("--save_name", type=str, default="pipeline_emb_V1", help="Name for saving results")
    parser.add_argument("--folder", type=str, default="embedding_V1", help="Abstra bucket folder where to save the results")
    parser.add_argument("--threads", type=int, default=2, help="number of threads for pytorch to use")

    args = parser.parse_args()
    print(f"DEBUG: torch originally used {torch.get_num_threads()}")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    print(f"DEBUG: Number of threads was set to {args.threads} and torch can use up to {torch.get_num_threads()} (general) and {torch.get_num_interop_threads()} (interop) threads.")
    output = pipeline(args.mode, save = args.save_results, save_name = args.save_name, folder = args.folder)
