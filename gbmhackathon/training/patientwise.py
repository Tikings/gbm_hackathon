"""
Define all necessary functions to allow correct patient wise representation learning with MLFLOW logging
"""

from gbmhackathon.s3_loader import load_s3
from pathlib import Path
from torch.utils.data import Dataset
import torch
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
            if key in ["clinical","wes"] :
                self.dict_emb[key] = self._load_pickle(
                            self.root, 
                            self.folder,
                            self.name_emb[key]
                            )["data"] # Other structure for this dataset

        self.dict_emb = dict([(key), self.format_dict_keys(self.dict_emb[key],
                                                          PATTERN_PATIENT)]
                             for key in self.dict_emb.keys())

        self.size_emb = {}
        for key in self.dict_emb.keys():
            shape = list(self.dict_emb[key].values())[0].shape
            self.size_emb[key] = shape

        # Retrieving patient list
        set_patient = set()
        for emb_dict in self.dict_emb.values() : 
            set_patient.update(list(emb_dict.keys()))
        patients = list(set_patient) 

        index_patient = list(range(len(patients)))
        self.ind2patient = dict(zip(index_patient, patients))

    def _load_pickle(self, root, folder_name,  emb_name):
        path = str(root + "/" + folder_name + "/" + emb_name)
        dict_patient = load_s3(path)
        return dict_patient

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
        dict_patient = {}
        list_available = []
        patient = self.ind2patient[idx]
        for key in self.dict_emb.keys() :
            if patient in self.dict_emb[key].keys():
                dict_patient[key] = self.dict_emb[key][patient]
                list_available.append(1)
            else :
                dict_patient[key] = torch.zeros(self.size_emb[key])
                list_available.append(0)
        return patient, dict_patient, torch.Tensor(list_available).to(torch.int8)