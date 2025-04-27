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
            if key in ["clinical","wes","hne"] :
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

def collate_patient_wise(batch): 
    list_patient = [patient[0] for patient in batch]
    list_dict_tensor = [patient[1] for patient in batch]
    modalities = list(list_dict_tensor[0].keys())
    # The order of the rows are the same as those of the dictionnary with the embeddings
    list_available = [patient[2] for patient in batch]
    dict_batched = {}
    for mod in modalities:
        mod_list = []
        # BEFORE, wrong logic to batch modality embeddings across patient
        # for dic in list_dict_tensor:
        #     print(dic)
        #     dict_batched[mod] = torch.stack([dic[mod] for _ in dic.keys()])

        # AFTER
        for dic in list_dict_tensor:
            mod_list.append(dic[mod])
        if len(mod_list[0].size()) == 2 and mod_list[0].size(0) > 1:
            aggregate_tiles = []
            for tensor in mod_list:
                # print(tensor.size())
                aggregate_tiles.append(tensor.mean(dim=0).squeeze())
                # print(tensor.mean(dim=0).squeeze().size())
            dict_batched[mod] = torch.stack(aggregate_tiles).type(torch.float32)
        else:
            dict_batched[mod] = torch.stack(mod_list).type(torch.float32)
    available_mod_tensor = torch.stack(list_available).type(torch.float32)
    return list_patient, modalities, dict_batched, available_mod_tensor