import pickle
import numpy as np
import zarr 
import torch 
from PIL import Image
import torch.utils
from torch.utils.data import Dataset, DataLoader
import tifffile
import tiffslide
from transformers import AutoImageProcessor, AutoModel

import os
import anndata as ad
import scanpy as sc
import pandas as pd
import json

from matplotlib.pyplot import imread
import liana as li
import decoupler as dc
import omnipath
import novae

from gbmhackathon.utils.visium_functions import normalize_anndata_wrapper
from gbmhackathon import MosaicDataset

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 1200

device = torch.device("cuda")


def retrieve_visium_embedding(
    sample_list = None,
    novae_model = "MICS-Lab/novae-human-0",
    radius_spatial_n = 200, 
    resolution = "hires"
):
    #Function that retrieves for all the spatial transcriptomic data (visium) the embedded cell types using the novae fundation model

    # Loading the needed data
    visium_dict = MosaicDataset.load_visium(
        sample_list=sample_list,
        resolution=resolution)
    
    # Normalization using the provided tools
    visium_obj = normalize_anndata_wrapper(visium_dict, target_sum=1e6)

    # Getting key, values of the dict (to work w/ novae)
    list_ids = list(visium_obj.keys())
    list_anndata = list(visium_obj.values())

    print(f"Considered list id : \n {list_ids}")

    print("___ Computing Spatial Neighbors ___")
    novae.utils.spatial_neighbors(list_anndata, radius=radius_spatial_n, technology ="visium")
    print("DONE")

    print("___ Importation of the pretrained model ___")
    model = novae.Novae.from_pretrained(novae_model)
    print(model)
    print("DONE")

    print("__ Computing representation __")
    model.compute_representations(list_anndata, accelerator = "cuda", num_workers = 4)
    print("DONE")
    
    list_latent = [anndata.obsm["novae_latent"] for anndata in list_anndata]
    print(f"Size latent space = {list_latent[0].shape[1]}")

    ### Add a function to normalize the different keys ?
    dict_latent = dict(zip(list_ids, list_latent))

    return dict_latent
    








class custom_dataset(Dataset): 
            
    def __init__(self, df_he, embed_phikon=None,embed_novae=None,emb_c=None,df_target,df_emb_clinical) : 
           # self.optimus=list_optimus_path## liste de fichiers 
            self.df_he=df_he
            self.processor_phikonv2= AutoImageProcessor.from_pretrained("owkin/phikon-v2")
            
            self.phikonv2 = AutoModel.from_pretrained("owkin/phikon-v2")
            self.phikonv2.to(device)
            self.phikonv2.eval()
            
            #### Chargement des données phikon
            if embed_phikon is  None : 
                    self.matrix_embedds=torch.empty(self.df_he.shape[0],1024) 
                    for idx in range(self.df_he.shape[0]): 
                        id_patient = self.df_he.index[idx]
            
                      
                        slide = tiffslide.TiffSlide(self.df_he.loc[id_patient]["path"])
                    
                      
                        image = slide.read_region((0, 0), 0, slide.dimensions)
                        image = image.convert("RGB")
                    
                    
                        input_phikon = self.processor_phikonv2(image,return_tensors="pt")
                        input_phikon = {key: value.to(device) for key, value in input_phikon.items()}
                    
                        
                        with torch.inference_mode():
                            outputs = self.phikonv2(**input_phikon)
                            features = outputs.last_hidden_state[:, 0, :]  
                        self.matrix_embedds[idx,:]=features
                        print(idx)
            else : 
                        self.matrix_embedds=embed_phikon
            ### Chargement des embeddings novae
            if embed_novae is None : 
                print("aucun embed_novae donnée, mais aucune méthode implémernée pour l'obtenir") 
                pass
            else : 

                self.embed_novae=embed_novae
                
            ###Chargement des données emb_c

            if emb_c is None : 
                raise Exception("Emb_c n'est pas donné en entrée, mais aucune méthode implémentée dans le dataset") 
            else: 
                self.emb_c=emb_c

        
                            
    def __len__(self) : 
            return self.df_he.shape[0] -7 ###
            
        
    def __getitem__(self, idx):
         id_patient = self.df_he.index[idx]
         if id_patient in ["HK_G_034a", "HK_G_040a","HK_G_095a","HK_G_096b"] or id_patient in ["HK_G_071a","HK_G_092b","HK_G_104a"]: 
             #Ici, problème : pas de correspondance entre les id d'embeddings, à corriger 
             
             
             return None
         else : 
                 
             embedding_image=self.matrix_embedds[idx,:]

             ##Chargement des données novae
             if "{}_vis".format(id_patient) in self.embed_novae.keys() : 
                 
                 embedding_novae=self.embed_novae["{}_vis".format(id_patient)]
             elif id_patient in self.embed_novae.keys() : 
             
                  embedding_novae=self.embed_novae[id_patient]
             else : 
                 
                 print("!!!",id_patient)
                 raise Exception("impossible de trouver l'id patient")


             ##chargement des embeddings _c 
             
             if id_patient in self.emb_c.keys() : 
                 
                     embedding_c=self.emb_c[id_patient]
             else : 
                 print("###",id_patient)
                 return None
                                 
             return id_patient, embedding_image,embedding_novae,embedding_c



def custom_collate_fn(batch):
    """
    Collate function qui ignore les éléments `None`.
    
    Args:
        batch (list): Liste des échantillons retournés par le dataset pour un batch.
    
    Returns:
        list ou tuple : Le batch filtré sans les éléments `None`.
    """
   
    batch = [item for item in batch if item is not None]

   
    if len(batch) == 0:
        return None  # Vous pouvez aussi lever une exception si nécessaire.

    return batch


            

        