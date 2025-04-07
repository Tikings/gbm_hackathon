# Load packages and classes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import tiffslide
import seaborn as sns
import gget, os, datetime
import tifffile
import zarr
from typing import Dict, Tuple, Optional, Union

import torch

from tqdm import tqdm
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from torch.utils.data import Dataset, DataLoader

# MosaicDataset and BruceDataset classes allow loading and visualisation of the different data sources
from gbmhackathon import MosaicDataset

os.chdir(os.path.dirname(os.path.abspath(__file__))) # Set working directory to be the folder where the script is

def remove_duplicates(df):
    df = df.loc[:,~df.columns.duplicated()] #remove duplicated columns
    df = df.loc[~df.index.duplicated(),:] #remove duplicated rows
    return df
    
def retrieve_useful_cols(df):
    query = np.sum(df, axis=0)
    return list(set(query[query > 0].index))

def get_data() -> Dict:
    """Load and return the WES dataset using MosaicDataset."""
    source_dict_mosaic = MosaicDataset.load_tabular()
    return source_dict_mosaic["wes"]
    
def prepare_data():
    source_dict_mosaic = MosaicDataset.load_tabular()
    
    onc = remove_duplicates(source_dict_mosaic["wes"]["WES CNV oncogenic"])
    dele = remove_duplicates(source_dict_mosaic["wes"]["WES CNV deletion"])
    amp = remove_duplicates(source_dict_mosaic["wes"]["WES CNV amplification"])
    mut = remove_duplicates(source_dict_mosaic["wes"]["WES mutations"])
    
    useful_cols = {}
    for key, df in {'onc':onc, 'del':dele, 'amp':amp, 'mut':mut}.items():
        useful_cols[key] = retrieve_useful_cols(df)
    
    total = []
    col_set = set()
    for key in useful_cols.keys():
        # print(len(useful_cols[key]))
        total += useful_cols[key]
        col_set = col_set.union(set(useful_cols[key]))
    uniques, counts = np.unique(total, return_counts=True)
    counts_dico = dict(zip(list(uniques), list(counts)))
    duplicated = [col for col in counts_dico.keys() if counts_dico[col] > 1]
    # print(f"Total number of useful columns from all dataframes {len(total)}\nNumber of unique columns {len(col_set)}.\nThere are {len(duplicated)} duplicated columns")

    
    def update_name(col_name, df_name):
        if col_name in duplicated:
            return f"{col_name}.{df_name}"
        return col_name
    
    #Update name with identifier
    onc_kept_cols = [update_name(col, 'onc') for col in useful_cols['onc']]
    # Retrieve kept columns
    onc = onc[useful_cols['onc']]
    # Update column names accordingly
    onc.columns = onc_kept_cols
    
    del_kept_cols = [update_name(col, 'del') for col in useful_cols['del']]
    # Retrieve kept columns
    dele = dele[useful_cols['del']]
    # Update column names accordingly
    dele.columns = del_kept_cols
    
    amp_kept_cols = [update_name(col, 'amp') for col in useful_cols['amp']]
    # Retrieve kept columns
    amp = amp[useful_cols['amp']]
    # Update column names accordingly
    amp.columns = amp_kept_cols
    
    mut_kept_cols = [update_name(col, 'mut') for col in useful_cols['mut']]
    # Retrieve kept columns
    mut = mut[useful_cols['mut']]
    # Update column names accordingly
    mut.columns = mut_kept_cols
    
    rdy_onc = onc[onc_kept_cols]
    rdy_del = dele[del_kept_cols]
    rdy_amp = amp[amp_kept_cols]
    rdy_mut = mut[mut_kept_cols]
    
    X_df = pd.concat([rdy_onc, rdy_del, rdy_amp, rdy_mut], axis=1)
    
    path = "../analysis/wes_selected_features.pkl"
    with open(path, "rb") as f:
       selected_all = pkl.load(f)
    X_final_df = X_df[selected_all]
    return X_final_df.astype(int)

def load_data(
    data: Optional[Union[str, Dict[str, torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """
    Load preprocessed WES data from a saved file or dictionary.

    Args:
        data: File path or preloaded dictionary (should be the output of pipeline_WES).

    Returns:
        Dictionary containing processed data tensors with keys:
            - "settings": computation settings.
            - "data": dictionary mapping patient IDs to embeddings.
            - "dataset": additional metadata (id2row, features, targets, X, Y).
    """
    if data is None:
        try:
            data = torch.load("WES_data.pt")
        except FileNotFoundError:
            raise FileNotFoundError(
                "No data provided and loading WES_data.pt failed."
            )
    elif isinstance(data, str):
        data = torch.load(data)
    return data
    
def pipeline_wes(save: bool = False, save_path: str = "wes_data.pt", verbose: bool = False,
    ) -> Dict[str, Union[Dict, Dict[str, torch.Tensor]]]:

    wes_emb_df = prepare_data()

    id2row = {patient_id: i for i, patient_id in enumerate(wes_emb_df.index)}
    dataset = {
        "id2row": id2row,
        "features": list(wes_emb_df.columns),
    }
    
    settings = {
        "model": "WES Embedding Pipeline",
        "date": datetime.datetime.now().isoformat(),
    }

    # Create dictionary mapping each patient_id to its embedding vector.
    data = {
        patient_id: torch.tensor(wes_emb_df.loc[patient_id].values)
        for patient_id in wes_emb_df.index
    }

    output = {"settings": settings, "data": data, "dataset": dataset}

    if save:
        if verbose:
            print(f"Saving data at {save_path}..")
        torch.save(output, save_path)
    return output

def get_emb(
    patient_id: str, data: Optional[Union[str, Dict[str, torch.Tensor]]] = None
) -> torch.Tensor:
    """
    Retrieve the WES embedding for a specific patient.

    Requires `pipeline_WES` to have been run first.

    Args:
        patient_id: The patient identifier (e.g., "HK_G_***").
        data: Preprocessed data dictionary or file path to the saved output from pipeline_WES.

    Returns:
        Torch tensor representing the embedding for the specified patient.
    """
    data = load_data(data)
    return data["data"][patient_id]


def get_batch(
    patient_ids: list, data: Optional[Union[str, Dict[str, torch.Tensor]]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve WES embeddings and corresponding targets for a batch of patients.

    Requires `pipeline_WES` to have been run first.

    Args:
        patient_ids: List of patient identifiers.
        data: Preprocessed data dictionary or file path to the saved output from pipeline_WES.

    Returns:
        Tuple containing:
            - A torch tensor of shape (batch_size, num_features) with the patient embeddings.

    Raises:
        ValueError: If any patient ID is not found in the data.
    """
    data = load_data(data)
    id2row = data["dataset"]["id2row"]

    tensor_indices = []
    for patient_id in patient_ids:
        if patient_id not in id2row:
            raise ValueError(f"Patient ID {patient_id} not found in data.")
        tensor_indices.append(id2row[patient_id])

    selection_mask = torch.LongTensor(tensor_indices)
    return data["data"][selection_mask, :]


def get_all_embeddings(
    data: Optional[Union[str, Dict[str, torch.Tensor]]] = None, format: str = "dict"
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Retrieve all patient embeddings.

    Requires `pipeline_WES` to have been run first.

    Args:
        data: Preprocessed data dictionary or file path to the saved output from pipeline_WES.
        format: Output format. Options:
            - "dict": Returns a dictionary mapping patient IDs to their embeddings.
            - "tensor": Returns a single torch tensor of shape (num_patients, num_features).

    Returns:
        - If format is "dict": Dictionary {patient_id: torch.Tensor(embedding)}.
        - If format is "tensor": A torch tensor containing embeddings for all patients.

    Raises:
        ValueError: If an invalid format is specified.
    """
    data = load_data(data)
    if format == "dict":
        return data["data"]
    elif format == "tensor":
        # Convert dictionary of embeddings to a tensor by stacking along the first dimension.
        embeddings = list(data["data"].values())
        return torch.stack(embeddings)
    else:
        raise ValueError(f"Invalid format '{format}'. Use 'dict' or 'tensor'.")

if __name__ == "__main__":
    # Example usage:
    # To run inference for all patients and generate the embeddings:
    result = pipeline_wes(verbose=False, save=False, save_path="wes_data.pt")
    print("Pipeline complete.")
