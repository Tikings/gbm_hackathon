





from gbmhackathon.training.patientwise import *
from gbmhackathon.models.mme import MultiModalEncoder
from gbmhackathon.utils.loss_functions import InfoNCELoss, RegularizedInfoNCELoss, SmoothingFunction
from gbmhackathon.s3_loader import load_s3
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import os
from datetime import datetime
import json



class experiment : 

    def __init__(self,config) :

        """
        
        
        """


        self.config=config

        self.__setup_output_folder()

        self.device=self.config["global_settings"]["device"]
        
        self.dataset = PatientLearningDataset(self.config["modalities_data"]["modalities"], self.config["modalities_data"]["pkl_storage_folder"], device=self.device)
        self.dataloader = DataLoader(self.dataset, self.config["training"]["batch_size"], shuffle=True, collate_fn=collate_patient_wise, generator=torch.Generator(device=self.dataset.device))
        
        
        self.missing_mods=load_s3(self.config["modalities_data"]["missing_mods"])
        
        self.id2sample=self.dataset.ind2patient
        self.patient2id=self.__get_id2patient()

        self.input_size_dict=self.__get_InputSize()

        ##### ICI,certainement modifier les classes 
                
        hne_cfg = self.__adaptBaseConfig(self.config["architecture"]["mid_capacity_cfg"], self.input_size_dict["hne"])
        spatial_cfg = self.__adaptBaseConfig(self.config["architecture"]["small_capacity_cfg"], self.input_size_dict["spatial"])
        clinical_cfg = self.__adaptBaseConfig(self.config["architecture"]["small_capacity_cfg"], self.input_size_dict["clinical"])
        wes_cfg = self.__adaptBaseConfig(self.config["architecture"]["mid_capacity_cfg"], self.input_size_dict["wes"])

        
        self.mme_cfg={"hne_cfg":hne_cfg, "clinical_cfg":clinical_cfg, "wes_cfg":wes_cfg, "spatial_cfg":spatial_cfg}
        self.__replace_instance_modality_cfg()

        self.mme=MultiModalEncoder(**self.mme_cfg).to(self.device)





    def fit_mme(self) : 
        """
        """

        EPOCHFLAG = 0
        problematic_batch_patient_ids = []
        
        mme = self.mme
        mme = torch.jit.script(mme)

        loss_fn = RegularizedInfoNCELoss(list(self.config["modalities_data"]["modalities"].keys()),
                                        self.patient2id, **self.config["training"]["InfoNCE_Loss"])
        optimizer = Adam(
            mme.parameters(),
            lr=1e-3,
        )

        EPOCH_LOSSES = []
        for epoch in range(self.config["training"]["epochs"]):
            
            epoch_loss = []
            for i, batch in enumerate(self.dataloader):
               
                patient_ids, modalities, X_dict, avail_mods = batch
             
                encoded = mme(X_dict)
                NANFLAG = {}

                for mod in X_dict.keys():
                    if self.__check(X_dict[mod]):
                        NANFLAG[mod] = 1
                        
                    for key, tensor in dict(mme.wes_net.named_parameters()).items():
                        if self.__check(tensor):
                            NANFLAG[mod] = 1
                            break
                    if self.__check(encoded[mod]):
                        NANFLAG[mod] = 1
                if 1 in list(NANFLAG.values()):
                    
                    problematic_batch_patient_ids.append(patient_ids)
                    EPOCHFLAG = 1
                    break
                    raise ValueError(f"Found NANs in modality {[key for key in NANFLAG.keys() if NANFLAG[key] == 1]}")
                else:
                   
                    loss_batch = (encoded, patient_ids, avail_mods)
                    
                    loss = loss_fn(loss_batch)
                    optimizer.zero_grad()
                   
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())

                    
            print(f"Epoch {epoch} loss: {np.mean(epoch_loss):.4f}".upper())
            if EPOCHFLAG == 1:
                print("BREAK")
                break
            EPOCH_LOSSES.append(np.mean(epoch_loss))
        ###return EPOCH_LOSSES, mme



    

    def __get_id2patient(self) : 

        """
        map each patient (not a sample) to an id 
        """
        all_ids = list(self.missing_mods.keys())
        patient_map = {key: i for key, i in zip(all_ids, [k for k in range(1,len(all_ids)+1)])}
        for patient_id, idx in patient_map.items():
            if patient_id.endswith('b'): # pour les rechutes on met le même index que le sample original
                patient_map[patient_id] = idx - 1
        return patient_map
    
    def __get_InputSize(self) : 
        
        """
        
        """
        toy_batch = next(iter(self.dataloader))
        input_size_dict = {}
        raw_emb = toy_batch[2]
        for mod in raw_emb.keys():
            print(mod, raw_emb[mod].size())
            input_size_dict[mod] = raw_emb[mod].size(1)

        return input_size_dict
    
    def __replace_instance_modality_cfg(self) : 
        """
        
        """

        for key in self.mme_cfg.keys(): 
            if self.mme_cfg[key]["act_fn"]=="SmoothingFunction": 
                self.mme_cfg[key]["act_fn"]=SmoothingFunction


    def __adaptConfig(self) : 
        """
        REPRENDRE APRES INDICATIONS RAPH
        """

        for modality in self.config["architecture"].keys(): 
            pass

    
    def __adaptBaseConfig(self,config,input_size): 
        """
        REPRENDRE APRES INDICATIONS RAPH
        """

        base_copy=deepcopy(self.config["architecture"]["base_config"])
        base_copy["layers"] = [input_size] + base_copy["layers"]
        return base_copy


    def __check_inf(self,tensor):
        return torch.isinf(tensor).sum() != 0
    
    def __check_nan(self,tensor):
        return torch.isnan(tensor).sum() != 0

    def __check(self,tensor):
        return self.__check_inf(tensor) or self.__check_nan(tensor)

    def __setup_output_folder(self) : 

        """
        """

        output_folder=self.config["global_settings"]["output_dir"]
        assert os.path.exists(output_folder), "The output folder specified does not exist"

        name_folder="experiment_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M"))

        os.makedirs(os.path.join(output_folder,name_folder))
        
        self.experiment_dir=os.path.join(output_folder,name_folder)

        with open(os.path.join(self.experiment_dir,"config.json"), "w") as f:
            json.dump(self.config, f)


    
    def __verify_config(self): 
        """
        verify here whatever you want on config
        """



        pass


        

        

                                           