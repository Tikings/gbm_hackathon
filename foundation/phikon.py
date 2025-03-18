

import torch 
import tifffile
import tiffslide
from transformers import AutoImageProcessor, AutoModel
import pandas as pd


#### Admettons tu possèdes déja le dict que nous obtenions à au hackathon



def setup_model(device : str) -> tuple : 
     
     """
     Fonction qui permet de récupérer le modèle phikonv2

     --> Voir https://huggingface.co/owkin/phikon-v2
     
     """

     processor_phikonv2= AutoImageProcessor.from_pretrained("owkin/phikon-v2")
     phikonv2 = AutoModel.from_pretrained("owkin/phikon-v2")
 
     phikonv2.to(device) ### Cette étape peut être bloquante, je l'avais insérée dans le code mais elle n'est pas dans la doc
     phikonv2.eval()

     return processor_phikonv2,phikonv2





def get_emb(slide : tiffslide.TiffSlide, processor:AutoImageProcessor, model : AutoModel) :
     
    """
     Fonction qui permet de récupérer l'embedding phikon pour chaque slide
     --> Voir https://huggingface.co/owkin/phikon-v2
     
     
    """

    ## Les deux lignes suivantes étaient dans utils_gbm.py : fichier qui n'a jamais tourné, mais si j'ai mis ça, c'est que j'avais aussi dû l'utiliser pour process les embeddings (fichier perdu)
    
    image = slide.read_region((0, 0), 0, slide.dimensions)
    image = image.convert("RGB")


    #### Code plus classique, voir site en haut


    inputs = processor(image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape

    assert features.shape == (1, 1024)  

    return features 



def get_all_embeddings( dataframe : pd.DataFrame) -> dict : 
    """
    POUR PHIKON
    ___________
    Le dataframe attendu ici est un dataframe formulé comme lors du hackathon, voir point 6 du notebook de chargement des données (que tu peux trouver dans Data_Loader_Adrien.ipynb)  

    Y'a de grandes chances pour qu'en fait tu n'aies pas les données de la même manière --> Dis moi et je corrige tout ça. 

    
    Le DataFrame : était récupéré avec : source_dict_mosaic["he"]["HE files"] 
    

    """
    processor_phikon,phikon=setup_model()


    dict_final=dict()

    for index,row in dataframe.iterrows(): 

            path=row["path"]
            subject_id=row["Subject Id"] ### Attention : je prends Subject Id et pas Patient ID, voir ce que les autres prennent dans leur dictionnaire final
            slide=tiffslide.TiffSlide(path)

            embedding=get_emb(slide,processor=processor_phikon,model=phikon)

            dict_final[subject_id]=embedding


    return dict_final



















    




