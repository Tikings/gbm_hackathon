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
from sklearn.impute import KNNImputer
import prince
import torch
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
        numeric_part = re.search(r"(\\d+)", base_id)
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
        data: File path or preloaded dictionary.

    Returns:
        Dictionary containing processed data tensors.
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


def prepare_data() -> Tuple[pd.DataFrame, list, list, list]:
    """
    Prepare clinical data by handling inconsistencies, defining features, and imputing missing values.

    Returns:
        Tuple containing the processed DataFrame, categorical features, numerical features, and targets.
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
    cat_features = [col for col in cat_features if col not in not_features]

    num_features = [
        col for col in gbm_df.columns if col not in cat_features + not_features
    ]
    num_features.remove("corrected_patient_id")

    gbm_df.fillna("Unk", inplace=True)
    return gbm_df, cat_features, num_features, TARGETS


def pipeline_clinical(
    verbose: bool = False, save: bool = False, save_path: str = "clinical_data.pt"
) -> Dict[str, torch.Tensor]:
    """
    Process clinical data, compute embeddings, and save the output.

    Args:
        verbose: Print progress messages if True.
        save: Save the processed data if True.
        save_path: File path for saving the data.

    Returns:
        Dictionary containing processed data tensors.
    """
    if verbose:
        print("Preparing Data...")
    gbm_df, cat_features, num_features, targets = prepare_data()

    cat_gbm_df = gbm_df[cat_features]
    num_gbm_df = gbm_df[num_features]
    target_df = gbm_df[targets]

    knn = KNNImputer()
    imputed_num_gbm_df = pd.DataFrame(
        knn.fit_transform(num_gbm_df), columns=num_features, index=num_gbm_df.index
    )
    scaler = StandardScaler()
    norm_num_gbm_df = pd.DataFrame(
        scaler.fit_transform(imputed_num_gbm_df),
        columns=num_features,
        index=num_gbm_df.index,
    )

    mca = prince.MCA(n_components=10, random_state=6262)
    mca_gbm_df = mca.fit_transform(cat_gbm_df.astype("category"))

    complete_df = norm_num_gbm_df.join(mca_gbm_df)

    id2row = {patient_id: i for i, patient_id in enumerate(gbm_df.index)}
    output = {
        "id2row": id2row,
        "X": torch.tensor(complete_df.values),
        "Y": torch.tensor(target_df.values),
    }

    if save:
        torch.save(output, save_path)
    return output


def get_emb(
    patient_id: str, data: Optional[Union[str, Dict[str, torch.Tensor]]] = None
) -> torch.Tensor:
    """
    Retrieve clinical embedding for a specific patient.

    Requires `pipeline_clinical` to have been run first.

    Args:
        patient_id: Patient identifier.
        data: Preprocessed data dictionary or path to saved file.

    Returns:
        Torch tensor representing the patient's embedding.
    """
    data = load_data(data)
    return data["X"][data["id2row"][patient_id]]


def get_batch(
    patient_ids: list, data: Optional[Union[str, Dict[str, torch.Tensor]]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve clinical embeddings and targets for a batch of patients.

    Requires `pipeline_clinical` to have been run first.

    Args:
        patient_ids: List of patient identifiers.
        data: Preprocessed data dictionary or path to a saved file.

    Returns:
        Tuple containing:
        - Torch tensor of shape (batch_size, num_features) with patient embeddings.
        - Torch tensor of shape (batch_size, num_targets) with corresponding target values.

    Raises:
        ValueError: If any patient ID is not found in the data.
    """
    data = load_data(data)
    id2row = data["id2row"]

    tensor_indices = []
    for patient_id in patient_ids:
        if patient_id not in id2row:
            raise ValueError(f"Patient ID {patient_id} not found in data.")
        tensor_indices.append(id2row[patient_id])

    selection_mask = torch.LongTensor(tensor_indices)
    return data["X"][selection_mask, :], data["Y"][selection_mask, :]


def get_all_embeddings(
    data: Optional[Union[str, Dict[str, torch.Tensor]]] = None, format: str = "tensor"
) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Retrieve all patient embeddings, either as a dictionary or a single tensor.

    Requires `pipeline_clinical` to have been run first.

    Args:
        data: Preprocessed data dictionary or path to a saved file.
        format: Output format. Options:
            - "dict" (default): Returns a dictionary mapping patient IDs to embeddings.
            - "tensor": Returns a single torch.Tensor containing all embeddings.

    Returns:
        - If format="dict": Dictionary {patient_id: torch.Tensor(embedding)}.
        - If format="tensor": A single torch.Tensor of shape (num_patients, num_features).

    Raises:
        ValueError: If an invalid format is specified.
    """
    data = load_data(data)

    if format == "dict":
        return {
            patient_id: get_emb(patient_id, data)
            for patient_id in data["id2row"].keys()
        }
    elif format == "tensor":
        return data["X"]
    else:
        raise ValueError(f"Invalid format '{format}'. Use 'dict' or 'tensor'.")


if __name__ == "__main__":
    # Example usage:
    # To run inference for all patients:
    result = pipeline_clinical(verbose=False, save=False, save_path="clinical_data.pt")
    print("Pipeline complete.")
