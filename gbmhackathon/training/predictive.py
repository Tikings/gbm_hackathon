"""
Define all necessary functions to learn downstream predictive tasks
"""

from gbmhackathon.s3_loader import load_s3
from pathlib import Path
from torch.utils.data import Dataset
import torch
import re, gc
import random as rd
from typing import List, Dict, Optional, Union
from copy import deepcopy
from itertools import product

ABSTRA_PROJECT_STORAGE_BUCKET = "s3://abstra-project-storage-lttemftb/1b75dc89-ad27-4a65-9e7f-877d1b4f36fc"
PATTERN_PATIENT = "(HK_G_[0-9]{3}(a|b))"
POSSIBLE_CLF_TARGETS = ["recurrent_sample", "mgmt_promoter_methylation"]

class PredictiveLearningDataset(Dataset):
    def __init__(self,
                 name_emb : Dict[str, str],
                 folder_name : str,
                 root_s3 : Path = ABSTRA_PROJECT_STORAGE_BUCKET, 
                 device: Optional[Union[str, torch.device]] = None,
                 dropout: float = 0,
                 ):
        self.root = root_s3
        self.name_emb = name_emb
        self.folder = folder_name
        assert isinstance(dropout, float), f"Dropout must be a float, current type is {type(dropout)}"
        assert dropout <= 1 and dropout >= 0, "Dropout parameter must be between 0 and 1 (both included)"
        self.dropout = dropout

        # Handle device selection - use CUDA if available, otherwise CPU
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print("cas 1 : device reconnu comme None")
        else:
            print("cas 2 : device reconnu : {} ".format(device))
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")

        self.dict_emb = {}
        for key in self.name_emb.keys(): 
            self.dict_emb[key] = self._load_pickle(
                self.root, 
                self.folder,
                self.name_emb[key])
            # Other structure for these modalities
            if key in ["clinical","wes","hne","bulk","scRNA"] :
                if key == "clinical":
                    clinical_data = self._load_pickle(
                            self.root, 
                            self.folder,
                            self.name_emb[key]
                            )
                    self.dict_emb[key] = clinical_data["data"]
                    id2row = clinical_data['dataset']['id2row']
                    Y = clinical_data["dataset"]['Y']
                    cat_features = clinical_data["dataset"]['cat_features']
                    targets = clinical_data["dataset"]['targets']
                    self.clf_targets_info = {clf_target:list(torch.unique(Y[:,targets.index(clf_target)])) for clf_target in set(targets).intersection(set(POSSIBLE_CLF_TARGETS))}
                    self.targets_names = clinical_data["dataset"]['targets']
                    self.dict_targets = {pid:Y[id2row[pid],:] for pid in id2row.keys()}
                else:
                    self.dict_emb[key] = self._load_pickle(
                            self.root, 
                            self.folder,
                            self.name_emb[key]
                            )["data"]

        self.dict_emb = dict([(key), self.format_dict_keys(self.dict_emb[key],
                                                          PATTERN_PATIENT)]
                             for key in self.dict_emb.keys())

        self.size_emb = {}
        for key in self.dict_emb.keys():
            shape = list(self.dict_emb[key].values())[0].shape
            self.size_emb[key] = shape

        # Retrieving patient list
        set_patient = set()
        for emb_dict in self.dict_emb.values(): 
            set_patient.update(list(emb_dict.keys()))
        patients = list(set_patient) 

        index_patient = list(range(len(patients)))
        self.ind2patient = dict(zip(index_patient, patients))

        if self.dropout > 0:
            self.augment_dataset()
            self.filter_dropout_samples()
            self.repair_indices()
        else:
            self.dataset_dropout_proportion = 0
            
        self.inputs = []
        self.targets = []
        self.store_all()
        
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

    def store_all(self):
        for idx in self.ind2patient.keys():
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
            self.inputs.append(tuple([patient, dict_patient, torch.Tensor(list_available).to(torch.int8), self.device]))
            self.targets.append(self.dict_targets[self.ind2patient[idx]])

    def augment_dataset(self):
        all_ids = list(self.ind2patient.keys())
        
        last_idx = all_ids[-1]
        new_idx = last_idx + 1
        for pidx in all_ids:
            patient, patient_mod_dict, available_mod_tensor, _ = self.getitem(pidx)
            modalities = patient_mod_dict.keys()
            if available_mod_tensor.sum() < len(modalities):
                patient_non_missing_modalities = [mod for mod_i,mod in enumerate(modalities) if available_mod_tensor[mod_i] == 1]
                # print(f"NON MISSING MODS: {patient_non_missing_modalities}")
                n_available = len(patient_non_missing_modalities)
                
                patient_missing_modalities = [mod for mod_i,mod in enumerate(modalities) if available_mod_tensor[mod_i] == 0]
                # print(f"MISSING MODS: {patient_missing_modalities}")
                n_missing = len(patient_missing_modalities)
    
                # generate possible augmentations (only on available modalities)
                possible_augmentations = [t for t in list(product([0,1], repeat=n_available)) if sum(t) > 0 and sum(t) != n_available]
        
                for aug in possible_augmentations:
                    knockout_mods = [patient_non_missing_modalities[i] for i in range(n_available) if aug[i] == 0]
                    # print(f"KNOCKOUT MODS: {knockout_mods}")
                    augmented_dict = deepcopy(patient_mod_dict)
    
                    # new id to identify augmented samples
                    augmented_patient_id = f"{patient}_d{'_'.join(knockout_mods)}"
                    # print(f"Augmented ID: {augmented_patient_id}")
                    self.ind2patient[new_idx] = augmented_patient_id
                    new_idx += 1
                    for mod in augmented_dict.keys():
                        # If we do not add the knockout mods, they will be treated as missing by __getitem__
                        if mod not in knockout_mods and mod not in patient_missing_modalities: 
                            self.dict_emb[mod][augmented_patient_id] = augmented_dict[mod]
                            self.dict_targets[augmented_patient_id] = self.dict_targets[patient]
                            
    def filter_dropout_samples(self):
        new_ids = [id for id in list(self.ind2patient.keys()) if 'd' in self.ind2patient[id]]
        N_all_ids = len(self.ind2patient.keys())
        original_len = len(new_ids)
        for id in new_ids:
            sample_id = self.ind2patient[id]
            if rd.random() > self.dropout:
                for mod in self.dict_emb.keys():
                    self.dict_emb[mod].pop(sample_id, None) # remove from modality dict (the key may or may not be there, we must use .pop(key, None) method
                    self.dict_targets.pop(sample_id, None)
                self.ind2patient.pop(id, None) # remove from id list, we know it is in the dict so we can use del
        gc.collect()
        new_len = len([id for id in list(self.ind2patient.keys()) if 'd' in self.ind2patient[id]])
        new_N_all_ids = len(self.ind2patient.keys())
        
        print(f"By keeping {(self.dropout*100):.2f}% of dropout augmented samples we went from:")
        print(f"{original_len} dropout samples ({(original_len*100/N_all_ids):.2f}% dropout in dataset) -- to --> {new_len} dropout samples ({new_len*100/new_N_all_ids:.2f}% dropout in dataset)")
        self.dataset_dropout_proportion = new_len/new_N_all_ids

    def repair_indices(self):
        current_ids = [id for id in self.ind2patient.keys()]
        correct_ids = [i for i in range(len(self.ind2patient))]
        repair_mapping = dict(zip(current_ids, correct_ids))

        self.ind2patient = {repair_mapping[id]: patient_id for id, patient_id in self.ind2patient.items()}
        
    def __len__(self):
        return len(self.ind2patient)
        
    def getitem(self, idx):
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
        return patient, dict_patient, torch.Tensor(list_available).to(torch.int8), self.device
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], self.targets_names, self.clf_targets_info
            

def batcher_graphs(
    batch_emb : List[torch.Tensor],
    batch_conn : List[torch.Tensor],
    device):
    all_emb = []
    all_conn = []
    batch = []
    offset = 0 

    for i, (emb, conn) in enumerate(zip(batch_emb, batch_conn)):
         all_emb.append(emb)
         all_conn.append(conn)
         batch.append(torch.full(emb.size[0]), i, dtype=torch.long)
         offset += emb.size(0)

    return (torch.cat(all_emb).to(device=device),
            torch.cat(all_conn, dim=1).to(device=device),
            torch.cat(batch).to(device=device))

def collate_predictive(batch): 
    targets_names = batch[0][2]
    clf_targets_info = batch[0][3]
    # Fetch target tensor per patients
    list_targets = [patient[1] for patient in batch]
    list_patient = [patient[0][0] for patient in batch]
    list_dict_tensor = [patient[0][1] for patient in batch]
    modalities = list(list_dict_tensor[0].keys())
    
    #print(f" Modalities available : {modalities}")
    
    # The order of the rows are the same as those of the dictionnary with the embeddings
    list_available = [patient[0][2] for patient in batch]

    # We presuppose that the device is the same for every embedding
    device = [patient[0][-1] for patient in batch][0]
    
    dict_batched = {}
    for mod in modalities:
        mod_list = []
        for dic in list_dict_tensor:
            # Ensure tensor is on the correct device
            tensor = dic[mod]
            if tensor.device != device:
                tensor = tensor.to(device)
            mod_list.append(tensor)
        if len(mod_list[0].size()) == 2 and mod_list[0].size(0) > 1:
            aggregate_tiles = []
            for tensor in mod_list:
                # print(tensor.size())
                aggregate_tiles.append(tensor.mean(dim=0).squeeze())
                # print(tensor.mean(dim=0).squeeze().size())
            dict_batched[mod] = torch.stack(aggregate_tiles).type(torch.float32)
        else:
            dict_batched[mod] = torch.stack(mod_list).type(torch.float32)
    device = dict_batched[list(dict_batched.keys())[0]].device
    available_mod_tensor = torch.stack(list_available).type(torch.float32).to(device)
    # class_0 = [(torch.tensor(0).unsqueeze(0) if c[-1] != 0 else torch.tensor(1).unsqueeze(0)) for c in list_targets]
    # class_1 = [(torch.tensor(0).unsqueeze(0) if c[-1] != 1 else torch.tensor(1).unsqueeze(0)) for c in list_targets]
    
    # class_0_targets_tensor = torch.stack(class_0).type(torch.float32).to(device)
    # # print(class_0_targets_tensor.size())
    # class_1_targets_tensor = torch.stack(class_1).type(torch.float32).to(device)
    
    # class_targets_tensor = torch.cat([class_0_targets_tensor, class_1_targets_tensor], dim=1).to(device)
    clf_targets_tensor = make_targets_tensor(list_targets, clf_targets_info, device)
    reg_targets_tensor = torch.stack([tensor[:-len(clf_targets_info.keys())] for tensor in list_targets]).type(torch.float32).to(device)
    targets_tensor = torch.cat([reg_targets_tensor, clf_targets_tensor], dim=1).to(device)
    return list_patient, modalities, dict_batched, available_mod_tensor, targets_tensor, targets_names

## SPATIAL GRAPHENCODER
# def collate_predictive(batch): 
#     targets_names = batch[0][2]
#     clf_targets_info = batch[0][3]
#     # Fetch target tensor per patients
#     list_targets = [patient[1] for patient in batch]
#     list_patient = [patient[0][0] for patient in batch]
#     list_dict_tensor = [patient[0][1] for patient in batch]
#     modalities = list(list_dict_tensor[0].keys())

#     # Removing connectivites of spatial embeddings from modalities
#     if "connectivities" in modalities :
#         modalities.remove("connectivities")
    
#     #print(f" Modalities available : {modalities}")
    
#     # The order of the rows are the same as those of the dictionnary with the embeddings
#     list_available = [patient[0][2] for patient in batch]

#     # We presuppose that the device is the same for every embedding
#     device = [patient[0][-1] for patient in batch][0]
    
#     dict_batched = {}
#     for mod in modalities:
#         if mod != "spatial" : 
#             mod_list = []
#             for dic in list_dict_tensor:
#                 # Ensure tensor is on the correct device
#                 tensor = dic[mod]
#                 if tensor.device != device:
#                     tensor = tensor.to(device)
#                 mod_list.append(tensor)
#             if len(mod_list[0].size()) == 2 and mod_list[0].size(0) > 1:
#                 aggregate_tiles = []
#                 for tensor in mod_list:
#                     # print(tensor.size())
#                     aggregate_tiles.append(tensor.mean(dim=0).squeeze())
#                     # print(tensor.mean(dim=0).squeeze().size())
#                 dict_batched[mod] = torch.stack(aggregate_tiles).type(torch.float32)
#             else:
#                 dict_batched[mod] = torch.stack(mod_list).type(torch.float32)
#         else :
#             spatial_stack = [dic["spatial"] for _ in dic.keys()] #! Check the dimension
#             connectivities_stack = [dic["connectivities"] for _ in dic.keys()]
#             dict_batched["spatial"] = batcher_graphs(spatial_stack, connectivities_stack)
#     device = dict_batched[list(dict_batched.keys())[0]].device
#     available_mod_tensor = torch.stack(list_available).type(torch.float32).to(device)
#     # class_0 = [(torch.tensor(0).unsqueeze(0) if c[-1] != 0 else torch.tensor(1).unsqueeze(0)) for c in list_targets]
#     # class_1 = [(torch.tensor(0).unsqueeze(0) if c[-1] != 1 else torch.tensor(1).unsqueeze(0)) for c in list_targets]
    
#     # class_0_targets_tensor = torch.stack(class_0).type(torch.float32).to(device)
#     # # print(class_0_targets_tensor.size())
#     # class_1_targets_tensor = torch.stack(class_1).type(torch.float32).to(device)
    
#     # class_targets_tensor = torch.cat([class_0_targets_tensor, class_1_targets_tensor], dim=1).to(device)
#     clf_targets_tensor = make_targets_tensor(list_targets, clf_targets_info, device)
#     reg_targets_tensor = torch.stack([tensor[:-len(clf_targets_info.keys())] for tensor in list_targets]).type(torch.float32).to(device)
#     targets_tensor = torch.cat([reg_targets_tensor, clf_targets_tensor], dim=1).to(device)
#     return list_patient, modalities, dict_batched, available_mod_tensor, targets_tensor, targets_names
    
def make_targets_tensor(list_targets: List, clf_target_info: Dict, device: Union[str, torch.device]):
    clf_targets = list(clf_target_info.keys())
    final_tensor = []
    for target in clf_targets:
        to_cat = []
        for i, u_val in enumerate(clf_target_info[target]):
            tensor = [(torch.tensor(0).unsqueeze(0) if c[-1] != u_val else torch.tensor(1).unsqueeze(0)) for c in list_targets]
            u_val_targets_tensor = torch.stack(tensor).type(torch.float32).to(device)
            to_cat.append(u_val_targets_tensor)
        
        clf_target_tensor = torch.cat(to_cat, dim=1).to(device)
        final_tensor.append(clf_target_tensor)
    return torch.cat(final_tensor, dim=1).type(torch.float32).to(device)
    
# def collate_predictive_colearning(batch): 
#     batch, y, targets_names = batch

#     list_targets = [y_tensor for y_tensor in y]
#     list_patient = [patient[0] for patient in batch]
#     list_dict_tensor = [patient[1] for patient in batch]
#     modalities = list(list_dict_tensor[0].keys())

#     auxilliary_targets = []
#     for patient in list_patient:
#         if 'd' in patient:
#             patient = patient[:patient.index('_d')]
#         auxilliary_targets.append(patient)
        
#     # Removing connectivites of spatial embeddings from modalities
#     if "connectivities" in modalities :
#         modalities.remove("connectivities")
    
#     #print(f" Modalities available : {modalities}")
    
#     # The order of the rows are the same as those of the dictionnary with the embeddings
#     list_available = [patient[2] for patient in batch]

#     # We presuppose that the device is the same for every embedding
#     device = [patient[-1] for patient in batch][0]
    
#     dict_batched = {}
#     for mod in modalities:
#         if mod != "spatial" : 
#             mod_list = []
#             for dic in list_dict_tensor:
#                 # Ensure tensor is on the correct device
#                 tensor = dic[mod]
#                 if tensor.device != device:
#                     tensor = tensor.to(device)
#                 mod_list.append(tensor)
#             if len(mod_list[0].size()) == 2 and mod_list[0].size(0) > 1:
#                 aggregate_tiles = []
#                 for tensor in mod_list:
#                     # print(tensor.size())
#                     aggregate_tiles.append(tensor.mean(dim=0).squeeze())
#                     # print(tensor.mean(dim=0).squeeze().size())
#                 dict_batched[mod] = torch.stack(aggregate_tiles).type(torch.float32)
#             else:
#                 dict_batched[mod] = torch.stack(mod_list).type(torch.float32)
#         else :
#             spatial_stack = [dic["spatial"] for _ in dic.keys()] #! Check the dimension
#             connectivities_stack = [dic["connectivities"] for _ in dic.keys()]
#             dict_batched["spatial"] = batcher_graphs(spatial_stack, connectivities_stack)

#     available_mod_tensor = torch.stack(list_available).type(torch.float32)
#     targets_tensor = torch.stack(list_targets).type(torch.float32)
#     return list_patient, modalities, dict_batched, available_mod_tensor, auxilliary_targets, targets_tensor, targets_names