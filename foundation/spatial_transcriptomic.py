"""Analyse de spatial transcriptomique --> Visium data."""

#############################################################
############# Importation des librairies ####################
#############################################################

from pathlib import Path
from typing import Dict, List, Optional, Union
import pickle as pkl

import novae
import torch
import anndata as ad

from gbmhackathon.utils.visium_functions import normalize_anndata_wrapper
from gbmhackathon import MosaicDataset


##################################################
############# Chargement des données #############
##################################################


def get_data(patient: List[str] = None,
             resolution = "hires",
             radius_spatial_neighbors : int = 300,
             target_sum : int = 1e6
             ) -> Dict[str, ad.AnnData]:
    """Get data from the data-center, apply normalisation and compute spatial neighbors

    Args:
        patient (List[str], optional): List of patient to analyse, if None retrieves all the patient data. Defaults to None.
        resolution (str, optional): _description_. Defaults to "hires".
        radius_spatial_neighbors (int, optional): _description_. Defaults to 300.
        target_sum (int, optional): _description_. Defaults to 1e6.

    Returns:
        Dict[str, ad.AnnData]: dictionnary containing the AnnData object (values) of given patients (keys)
    """

    # Loading Raw Data from Mosaic Dataset
    data = MosaicDataset.load_visium(sample_list= patient, resolution = resolution)

    # Nomalization using their function
    norm_data = normalize_anndata_wrapper(data, target_sum = target_sum)

    ## Computing the spatial neighbors for further analysis
    list_patient, list_annData = list(norm_data.keys()), list(norm_data.values())
    novae.utils.spatial_neighbors(list_annData, radius=radius_spatial_neighbors, technology ="visium")

    return dict(zip(list_patient, list_annData))


def setup_novae(model_type = "MISC-Lab/novae-human-0") -> novae.model.Novae :
    """Retrieving the foundation model from hugging face

    Args:
        model_type (str, optional): Model id on hugging face (see https://huggingface.co/MICS-Lab for the other models). Defaults to "MISC-Lab/novae-human-0".

    Returns:
        novae.model.Novae: Foundation model
    """
    ### Loading the model from HF
    return novae.model.from_pretrained(model_type)

def get_emb(patient : str,
            dict_annData : Dict[str, ad.AnnData],
            model : novae.model.Novae) -> torch.Tensor :
    """Compute the embegging for a patient

    Args:
        patient (str): Id of the patient
        dict_annData (Dict[str, ad.AnnData]): Output of the get_data function
        model (novae.model.Novae): model retrieved from hugging face.

    Returns:
        torch.Tensor: Computed representation of the patient spatial transcriptomic data
    """
    #getting the patient data
    data_patient = dict_annData[patient]

    # Computing representations
    model.compute_representations(data_patient)

    return torch.from_numpy(data_patient.obsm["novae_latent"])



def get_all_embeddings(dict_annData : Dict[str, ad.AnnData],
                       model : novae.model.Novae) -> Dict[str, torch.Tensor]:
    """Compute all the representations of the data

    Args:
        dict_annData (Dict[str, ad.AnnData]): Output of the get_data function
       model (novae.model.Novea): Foundation model retrieved from Hugging Face

    Returns:
        Dict[str, torch.Tensor]: dictionnary containnig the embeddings computed by the model (values) for
        each patiert (keys)
    """

    # Converting to list all the AnnData files
    list_patients, list_anndata = dict_annData.keys(), list(dict_annData.values())

    # Computing all the representations
    model.compute_representations(list_anndata)

    # Retreiving all the representations computed
    list_latent = [torch.from_numpy(anndata.obsm["novae_latent"]) for anndata in list_anndata]

    return dict(zip(list_patients, list_latent))




###############################################
############## Full pipeline ##################
###############################################

def pipeline_novae(patient: List[str] = None,
                   resolution = "hires",
                   radius_spatial_neighbors : int = 300,
                   target_sum : int = 1e6,
                   model_type = "MISC-Lab/novae-human-0",
                   ) -> Dict[str, torch.Tensor]:

    #Getting all the data
    dict_annData = get_data(patient = patient,
                            resolution = resolution,
                            radius_spatial_neighbors = radius_spatial_neighbors,
                            target_sum = target_sum)
    #Getting the model
    model = setup_novae(model_type = model_type)

    #Compute embeddings
    return get_all_embeddings(
        dict_annData=dict_annData,
        model = model)

###########################################
############## Save data ##################
###########################################

# def save_novae(saving_folder : Path,
#                dict_emb : Dict[str,torch.Tensor],
#                name : str) -> None :
#     """To save the embeddings

#     Args:
#         saving_folder (Path): Path of the folder in which to save the data
#         dict_emb (Dict[str,torch.Tensor]): Dict of the embeddings
#         name (str): name of the file
#     """

#     # Getting the date
#     date = dict_emb["settings"]["date"]

#     # Saving the model
#     with open(saving_folder / f"{name}_{date}.pkl", "wb") as f:
#         pkl.dump(dict_emb, f)
#     print(f"Model saved at {saving_folder / f"{name}_{date}.pkl"}")
