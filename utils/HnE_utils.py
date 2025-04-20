import h5py 
import numpy as np
import os 


def get_files(path : str) -> dict :

    """
    Permet de récupérer les objets h5py pour chaque PATCH    
    """ 

    with h5py.File(path, 'r') as f:
        coords=f['coords'][:]
        features=f["features"][:]

    return {"coords" : coords, "features" : features}


def aggregate(features : np.ndarray,type_ : str = "mean") -> np.ndarray : 
    
    """
    Permet d'aggréger tous les embeddings de chaque patchs pour obtenir un vecteur unique/slide
    """
    if type_ =="mean" :
         return np.mean(features,axis=0)
    
    else :

        raise Exception("Le choix d'aggrégation ne correspond à aucune implémentation")
    

def get_embedding_matrix(folder_path : str, aggreg : str="mean") -> tuple:

    """
    Permet, à partir d'un dossier, de récupérer tous les embeddings, aggrégs, de toutes les slides.
    
    """ 
    list_slides=[i for i in os.listdir(folder_path) if i.endswith(".h5")]

    dict_name_pos=dict()

    for i,name in enumerate(list_slides): 

        if i==0: 

            matrix=aggregate(features=get_files( os.path.join(folder_path,name) )["features"],type_=aggreg)
        
            dict_name_pos[i]=name.split(".h5")[0]

        else : 
            try : 
                matrix=np.vstack((matrix,aggregate(features=get_files( os.path.join(folder_path,name) )["features"],type_=aggreg)))
            except ValueError : 
                 print("Certainement un problème de taille d'embeddings durant la concaténation")
            
            dict_name_pos[i]=name.split(".h5")[0]

    return matrix,dict_name_pos 




def plot_rapidos() : 

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt 
    print(mat.shape) 
    t=TSNE(n_components=2,perplexity=30) 
    transformed_data=t.fit_transform(mat)

    plt.scatter(transformed_data[:,0],y=transformed_data[:,1])
    plt.display()







