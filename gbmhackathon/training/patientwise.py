"""
Define all necessary functions to allow correct patient wise representation learning with MLFLOW logging
"""

from gbmhackathon.s3_loader import load_s3
from pathlib import Path
from torch.utils.data import Dataset
import re
from typing import List, Dict

ABSTRA_PROJECT_STORAGE_BUCKET = "s3://abstra-project-storage-lttemftb/1b75dc89-ad27-4a65-9e7f-877d1b4f36fc"
PATTERN_PATIENT = "(HK_G_[0-9]{3}(a|b))"
class PatientLearningDataset(Dataset):
    def __init__(self,
                 name_emb : Dict[str, str],
                 folder_name : str,
                 root_s3 : Path = ABSTRA_PROJECT_STORAGE_BUCKET, 
                 ):
        self.root = root_s3
        self.name_emb = name_emb
        self.folder = folder_name
        
        self.dict_emb = {}
        for key in self.name_emb.keys(): 
            self.dict_emb[key] = self._load_pickle(
                self.root, 
                self.folder,
                self.name_emb[key])
            if key == "clinical" : 
                self.dict_emb[key] = self._load_pickle(
                            self.root, 
                            self.folder_name,
                            self.name_emb[key]
                            )["data"] # Other structure for this dataset

        # Retrieving patient list
        set_patient = set()
        for emb_dict in self.dict_emb.values() : 
            set_patient.update(list(emb_dict.keys()))
        patients = list(set_patient) 

        index_patient = list(range(len(patients)))
        self.ind2patient = dict(zip(index_patient, patients))

    @staticmethod
    def _load_pickle(self, root, folder_name,  emb_name):
        path = str(root + "/" + folder_name + "/" + emb_name)
        dict_patient = load_s3(path)
        formated_dict = self.format_dict_keys(dict_patient, PATTERN_PATIENT)
        return formated_dict

    @staticmethod
    def format_dict_keys(dict_, pattern):
        new_dict = {}
        for keys, value in dict_.items():
            matcher = re.match(pattern, keys)
            if matcher : 
                new_key = matcher.group(0)
                new_dict[new_key] = value
        return new_dict
    
    def __len__(self):
        return len(self.ind2patient)
    
    def __getitem__(self, idx): 
        patient = self.ind2patient[idx]
        return dict([
            (key, self.dict_emb[key][patient])
            for key in self.dict_emb.keys()
        ])

    
