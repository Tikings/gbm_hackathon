import torch 
import gc
from tqdm import tqdm
import tiffslide
from transformers import AutoImageProcessor, AutoModel
import pandas as pd
from gbmhackathon.utils.he_functions import get_tif_bytes_io
from gbmhackathon import MosaicDataset


def load_data() : 
    return MosaicDataset.load_tabular()["he"]["HE files"]


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
    # Preprocessing the image
    image = slide.read_region((0, 0), 0, slide.dimensions)
    image = image.convert("RGB")

    # Inference
    inputs = processor(image, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]  # (1, 1024) shape

    # Check
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
    # Loading model
    processor_phikon,phikon=setup_model()

    # Formatting dataframe
    df = dataframe.reset_index(inplace=False)

    emb_dict = dict()

    for _,row in tqdm(df.iterrows()): 

            # Getting path
            path=row["path"]
            subject_id=row["Subject Id"] 
            
            # Loading file from s3
            bio = get_tif_bytes_io(slide_path=path)
            slide=tiffslide.TiffSlide(bio)

            # Computing embedding
            embedding=get_emb(slide,processor=processor_phikon,model=phikon)
            
            del bio 
            gc.collect()

            emb_dict[subject_id]=embedding

    return emb_dict


def pipeline_phikon(device : str = "cpu", verbose : bool = False):
    
    if verbose : 
        print("Loading data...")
    # Loading data
    dataframe = load_data()

    if verbose :
        print("Processing embeddings...")

    emb_dict = get_all_embeddings(dataframe)

    return emb_dict
