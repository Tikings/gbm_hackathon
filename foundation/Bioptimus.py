

import torch
import zarr
import numpy as np


def get_all_embeddings( dataframe : pd.DataFrame, aggreg : str="mean") -> dict : 

    """
    Pour les embeddings Bioptimus
    On suppose à nouveau qu'on a accès au même dataframe que lors du hackathon,voir point 6 du notebook de chargement des données (que tu peux trouver dans Data_Loader_Adrien.ipynb)  
    ça j'ai jamais fait, tout copié sur le notebook d'exploration puis quelques ajouts (concaténation) : j'ai aucune idée de si ça marche ou non 

    Args 
    ____
    DataFrame : était initialement récupéré avec : source_dict_mosaic["he"]["H1 features"] 

    aggreg : fonction d'aggrégation pour aggréger toutes les tuiles
    """

    dict_result=dict()
    for index,row in dataframe.iterrows() :

         path=row["path"]
         subject_id=row["Subject Id"]
         h1_zarr = zarr.open(path,'r')
         h1_emb = h1_zarr["emb"][:]
         
         if aggreg == "mean" : 
              

            h1_aggreg=np.mean(h1_emb,axis=1)
            print("!!!!!!!! après aggrégation, la taille du vecteur aggrégé est de : {} vérifier que c'est cohérent (doit correspond à (1,nb_features)) ".format(h1_aggreg.shape))

        ### Fais toi plaise pour tenter d'autres aggrégations stv

         dict_result[subject_id]=h1_aggreg
    return dict_result
    













