#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clinical Embeddings Pipeline

This script processes clinical data to generate embeddings for patient samples.

It includes:
- Data loading and preprocessing
- Dimensionality reduction via MCA
- Imputation and normalization of numerical features
- Functions to retrieve patient embeddings

Functions that require `pipeline_clinical` to have been run first are explicitly documented.

Output Structure:
    - "settings": Dictionary detailing the settings used for computing the embeddings (including computation date).
    - "data": Dictionary mapping patient IDs (e.g., "HK_G_***") to their corresponding embedding (torch.Tensor).
    - "dataset": Additional dictionary containing metadata such as id2row mapping, feature names, target names, and full tensors of features and targets.

Dependencies:
    - torch
    - pandas
    - numpy
    - prince
    - sklearn
    - gbmhackathon (for MosaicDataset)

Usage:
    Run `pipeline_clinical()` to preprocess the data and save the embeddings.
"""

# Original Owkin Imports
import gget
import numpy as np
import pandas as pd
import tifffile
import tiffslide
import zarr
from gbmhackathon import BruceDataset, MosaicDataset

# Custom imports
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
import prince
import torch
import datetime
import re
from typing import Dict, Tuple, Optional, Union


def identify_sample_id(sample_id: str) -> bool:
    """Identify if a sample is recurrent based on its ID."""
    return sample_id.endswith("b")


def correct_patient_id(sample_id: str) -> str:
    """Correct patient ID by standardizing the numeric part of recurrent samples."""
    match = re.match(r"(.*)b$", sample_id)
    if match:
        base_id = match.group(1)
        numeric_part = re.search(r"(\d+)", base_id)
        if numeric_part:
            num_str = numeric_part.group(0)
            new_num = str(int(num_str) - 1).zfill(len(num_str))
            corrected_id = base_id.replace(num_str, new_num)
            return corrected_id
    return sample_id[:-1]


def get_recurrent_info(sample_id: str) -> int:
    """Return 1 if the sample is recurrent, otherwise 0."""
    return 1 if sample_id.endswith("b") else 0


def get_data() -> pd.DataFrame:
    """Load and return the clinical dataset using MosaicDataset."""
    source_dict_mosaic = MosaicDataset.load_tabular()
    return source_dict_mosaic["clinical"]["processed gbm clinical"]


def load_data(
    data: Optional[Union[str, Dict[str, torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    """
    Load preprocessed clinical data from a saved file or dictionary.

    Args:
        data: File path or preloaded dictionary (should be the output of pipeline_clinical).

    Returns:
        Dictionary containing processed data tensors with keys:
            - "settings": computation settings.
            - "data": dictionary mapping patient IDs to embeddings.
            - "dataset": additional metadata (id2row, features, targets, X, Y).
    """
    if data is None:
        try:
            data = torch.load("clinical_data.pt")
        except FileNotFoundError:
            raise FileNotFoundError(
                "No data provided and loading clinical_data.pt failed."
            )
    elif isinstance(data, str):
        data = torch.load(data)
    return data

def impute_col(col):
    if any(t in str(col.dtype) for t in ['int', 'float']):
        return col.fillna(-1)
    else:
        return col.fillna("Unk")
        
def prepare_data() -> Tuple[pd.DataFrame, list, list, list]:
    """
    Prepare clinical data by handling inconsistencies, defining features, and imputing missing values.

    Returns:
        Tuple containing:
            - Processed DataFrame with corrections and missing values handled.
            - List of categorical feature names.
            - List of numerical feature names.
            - List of target column names.
    """
    gbm_df = get_data()
    gbm_df["corrected_patient_id"] = gbm_df.index.map(correct_patient_id)
    gbm_df["recurrent_sample"] = gbm_df.index.map(get_recurrent_info)

    NUM_TARGETS = [
        "os_years",
        "pfs_years",
        "largest_diameter_of_the_primary_tumour_mm_duplicated_0",
    ]
    CAT_TARGET = ["recurrent_sample"]
    TARGETS = NUM_TARGETS + CAT_TARGET

    cat_features = [col for col in gbm_df.columns if gbm_df[col].nunique() <= 10]
    not_features = [
        "cohort_code",
        "cancer_indication",
        "sample_source",
        "sample_origin",
        "tumour_resection_chronology",
    ]
    cat_features = [col for col in cat_features if col not in not_features + TARGETS]

    num_features = [
        col for col in gbm_df.columns if col not in cat_features + not_features + TARGETS
    ]
    num_features.remove("corrected_patient_id")
    
    num_features.remove("patient_id")

    # Replace categorical NAs with a new modality: -1 for numeric ordinal features and "Unk" for string categorical
    gbm_df[cat_features] = gbm_df[cat_features].apply(impute_col, axis=0)
    return gbm_df, cat_features, num_features, TARGETS


def pipeline_clinical(
    n_components : int = 10, verbose: bool = False, save: bool = False, save_path: str = "clinical_data.pt"
) -> Dict[str, Union[Dict, Dict[str, torch.Tensor]]]:
    """
    Process clinical data, compute embeddings, and save the output.

    The output structure is:
        - "settings": Dictionary detailing the computation settings and the date.
        - "data": Dictionary mapping each patient ID (e.g., "HK_G_***") to its corresponding embedding (torch.Tensor).
        - "dataset": Additional metadata including id2row mapping, feature names, target names, and full tensors for features (X) and targets (Y).

    Args:
        n_components: Number of components to keep for Multi Component Analysis.
        verbose: If True, prints progress messages.
        save: If True, saves the output to the specified path.
        save_path: File path to save the processed data.

    Returns:
        Dictionary with keys "settings", "data", and "dataset" as described above.
    """
    if verbose:
        print("Preparing Data...")
    gbm_df, cat_features, num_features, targets = prepare_data()

    cat_gbm_df = gbm_df[cat_features]
    num_gbm_df = gbm_df[num_features]
    target_df = gbm_df[targets]

    if verbose:
        print("Imputing numerical features...")
    knn = KNNImputer()
    imputed_num_gbm_df = pd.DataFrame(
        knn.fit_transform(num_gbm_df), columns=num_features, index=num_gbm_df.index
    )

    if verbose:
        print("Scaling numerical features...")
    scaler = StandardScaler()
    norm_num_gbm_df = pd.DataFrame(
        scaler.fit_transform(imputed_num_gbm_df),
        columns=num_features,
        index=num_gbm_df.index,
    )

    if verbose:
        print("Multi Component Analysis..")
    mca = prince.MCA(n_components=n_components, random_state=6262)
    mca_gbm_df = mca.fit_transform(cat_gbm_df.astype("category"))

    if verbose:
        print("Retrieving column contributions..")
    cat_col_contributions = mca.column_contributions_
    cat_col_contributions["feature"] = ['_'.join(mod.split('_')[:-1])[:-1] for mod in list(cat_col_contributions.index)]
    
    mca_gbm_df.columns = [f'C{i}' for i in range(mca_gbm_df.shape[1])]
    
    complete_df = norm_num_gbm_df.join(mca_gbm_df)
    
    if verbose:
        print("Imputing targets..")
    # Replace NAs in target
    imputed_targets = pd.DataFrame(
        knn.fit_transform(target_df), columns=targets, index=target_df.index
    )

    if verbose:
        print("Scaling targets..")
    # Normalize targets
    norm_target_df = pd.DataFrame(
        scaler.fit_transform(imputed_targets[[col for col in targets if col != "recurrent_sample"]]),
        columns=[col for col in targets if col != "recurrent_sample"],
        index=target_df.index,
    )
    norm_target_df = pd.concat([norm_target_df, imputed_targets[["recurrent_sample"]]], axis=1)

    id2row = {patient_id: i for i, patient_id in enumerate(gbm_df.index)}
    dataset = {
        "id2row": id2row,
        "cat_features": cat_features,
        "num_features": num_features,
        "features": list(complete_df.columns),
        "targets": targets,
        'mca_contributions':cat_col_contributions,
        "X": torch.tensor(complete_df.values),
        "Y": torch.tensor(norm_target_df.values),
        "imputed_Y": torch.tensor(imputed_targets.values),
        "original_Y": torch.tensor(target_df.values),
    }

    settings = {
        "model": "Clinical Embedding Pipeline",
        "date": datetime.datetime.now().isoformat(),
    }

    # Create dictionary mapping each patient_id to its embedding vector.
    data = {
        patient_id: torch.tensor(complete_df.loc[patient_id].values)
        for patient_id in complete_df.index
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
    Retrieve the clinical embedding for a specific patient.

    Requires `pipeline_clinical` to have been run first.

    Args:
        patient_id: The patient identifier (e.g., "HK_G_***").
        data: Preprocessed data dictionary or file path to the saved output from pipeline_clinical.

    Returns:
        Torch tensor representing the embedding for the specified patient.
    """
    data = load_data(data)
    return data["data"][patient_id]


def get_batch(
    patient_ids: list, data: Optional[Union[str, Dict[str, torch.Tensor]]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve clinical embeddings and corresponding targets for a batch of patients.

    Requires `pipeline_clinical` to have been run first.

    Args:
        patient_ids: List of patient identifiers.
        data: Preprocessed data dictionary or file path to the saved output from pipeline_clinical.

    Returns:
        Tuple containing:
            - A torch tensor of shape (batch_size, num_features) with the patient embeddings.
            - A torch tensor of shape (batch_size, num_targets) with the corresponding target values.

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
    return data["dataset"]["X"][selection_mask, :], data["dataset"]["Y"][
        selection_mask, :
    ]


def get_all_embeddings(
    data: Optional[Union[str, Dict[str, torch.Tensor]]] = None, format: str = "dict"
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Retrieve all patient embeddings.

    Requires `pipeline_clinical` to have been run first.

    Args:
        data: Preprocessed data dictionary or file path to the saved output from pipeline_clinical.
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
    result = pipeline_clinical(verbose=False, save=False, save_path="clinical_data.pt")
    print("Pipeline complete.")
