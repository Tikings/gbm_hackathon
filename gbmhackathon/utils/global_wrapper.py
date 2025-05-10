from gbmhackathon.training.patientwise import *
from gbmhackathon.models.mme import MultiModalEncoder, GBMNet
from gbmhackathon.utils.loss_functions import InfoNCELoss, RegularizedInfoNCELoss, SmoothingFunction
from gbmhackathon.utils.module_functions import instantiate
from gbmhackathon.s3_loader import load_s3
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import os
from datetime import datetime
import json
import pickle as pkl 
import torch.nn as nn
from typing import Dict
from gbmhackathon.utils.module_functions import enforce_signature_types




class MME_Global : 
    """Class to handle MultiModalEncoder experiments"""
    @enforce_signature_types
    def __init__(self,config: Dict, save: bool=True) :

        """
        
        
        """


        self.config=config
       
        self.save_output=save
        if self.save_output :
            self.__setup_output_folder()
        self.__format_config()
        self.__verify_config()
        self.device=self.config["global_settings"]["device"]
        
        
        self.dataset = PatientLearningDataset(self.config["MME_Model"]["modalities_data"]["modalities"], self.config["MME_Model"]["modalities_data"]["pkl_storage_folder"], device=self.device)
        self.dataloader = DataLoader(self.dataset, self.config["MME_Model"]["training"]["batch_size"], shuffle=True, collate_fn=collate_patient_wise, generator=torch.Generator(device=self.dataset.device))
        
        
        self.missing_mods=load_s3(self.config["MME_Model"]["modalities_data"]["missing_mods"])
        
        self.id2sample=self.dataset.ind2patient
        self.patient2id=self.__get_id2patient()

        self.input_size_dict=self.__get_InputSize() ### Voir pour modifier automatiquement les input size donnés en config

        ##### ICI,certainement modifier les classes 
        self.available_modalities=self.__get_available_modalities()
        assert len(self.available_modalities)>0, '0 modality detected, weird'
        self.mme_cfg=self.config["MME_Model"]["architecture"]["MME"]
        
        # if "hne" in self.available_modalities : 
        #     hne_cfg = self.__adaptBaseConfig(self.config["MME_Model"]["architecture"]["mid_capacity_cfg"], self.input_size_dict["hne"])
        #     self.mme_cfg["hne_cfg"]=hne_cfg

        # if "spatial" in self.available_modalities : 
        #      spatial_cfg = self.__adaptBaseConfig(self.config["MME_Model"]["architecture"]["small_capacity_cfg"], self.input_size_dict["spatial"])
        #      self.mme_cfg["spatial_cfg"]=spatial_cfg
        # if "clinical" in self.available_modalities : 
        #     clinical_cfg = self.__adaptBaseConfig(self.config["MME_Model"]["architecture"]["small_capacity_cfg"], self.input_size_dict["clinical"])
        #     self.mme_cfg["clinical_cfg"]=clinical_cfg
        
        # if "wes" in self.available_modalities : 
        #     wes_cfg = self.__adaptBaseConfig(self.config["MME_Model"]["architecture"]["mid_capacity_cfg"], self.input_size_dict["wes"])
        #     self.mme_cfg["wes_cfg"]=wes_cfg
        
       
        # self.__replace_instance_modality_cfg()

        self.mme=MultiModalEncoder(**self.mme_cfg).to(self.device)
        self.trained=False

        


    def fit_mme(self) : 
        """
        """

        EPOCHFLAG = 0
        problematic_batch_patient_ids = []
        
        mme = self.mme
        # mme = torch.jit.script(mme)

        regularized_loss_cfg = {"modalities":list(self.config["MME_Model"]["modalities_data"]["modalities"].keys()),
                                "patient_map":self.patient2id}
        regularized_loss_cfg.update(self.config["MME_Model"]["training"]["InfoNCE_Loss"])
        loss_fn = instantiate(regularized_loss_cfg, RegularizedInfoNCELoss)
        optimizer = Adam(
            mme.parameters(),
            lr=1e-3,
        )

        EPOCH_LOSSES = []
        for epoch in range(self.config["MME_Model"]["training"]["epochs"]):
            
            epoch_loss = []
            for i, batch in enumerate(self.dataloader):
               
                patient_ids, modalities, X_dict, avail_mods = batch
             
                encoded = mme(X_dict)
                NANFLAG = {}

                for mod in X_dict.keys():
                    if self.__check(X_dict[mod]):
                        NANFLAG[mod] = 1
                        
                    # for key, tensor in dict(mme.modality_net_map[list(mme.modality_net_map.keys())[0]].named_parameters()).items():
                    #     if self.__check(tensor):
                    #         NANFLAG[mod] = 1
                    #         break

                    first_net = list(mme.modality_net_map.children())[0]  # Récupère le premier module
                    for name, tensor in first_net.named_parameters():
                        if self.__check(tensor):
                            NANFLAG[mod] = 1
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
        
        self.mme=mme
        self.training={"epoch_losses" : EPOCH_LOSSES}
        self.trained=True
        



    def __save_model(self,name: str= None): 
        
        assert (self.trained==True), "The model should be trained before saving, but it is not "
        if name is None : 
            name="mme_model.pt"
        self.path_model=os.path.join(self.experiment_dir,name)
        torch.save(self.mme.state_dict(),self.path_model)
        




    def save(self,name:str=None):

        if name is None : 

            name="{}.pkl".format(self.__class__.__name__)
        
        self.__save_model()
        self.mme=None ### On sauvegarde le modèle dans un obet à part
        with open(os.path.join(self.experiment_dir,name), 'wb') as f:
                pkl.dump(self, f)



    def reload_model(self): 
        """
        Pour reload le modèle après réouverture de la classe à postériorie
        """

        self.mme=MultiModalEncoder(**self.mme_cfg).to(self.device)
        state_dict = torch.load(self.path_model)
        self.mme.load_state_dict(state_dict)



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

        for modality in self.config["MME_Model"]["architecture"].keys(): 
            pass

    
    def __adaptBaseConfig(self,config,input_size): 
        """
        REPRENDRE APRES INDICATIONS RAPH
        """

        base_copy=deepcopy(self.config["MME_Model"]["architecture"]["base_config"])
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

        name_folder="experiment_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        os.makedirs(os.path.join(output_folder,name_folder))
        
        self.experiment_dir=os.path.join(output_folder,name_folder)

        with open(os.path.join(self.experiment_dir,"config.json"), "w") as f:
            json.dump(self.config, f)

    def __get_available_modalities(self): 

        mod_cfg=self.config["MME_Model"]["modalities_data"]["modalities"]
        list_modalities=list()

        for modality in mod_cfg.keys(): 
            if modality not in ["pkl_storage_folder","missing_mods"] :
                list_modalities.append(modality)

        return list_modalities
    


    def __format_config(self) : 
        """
        Because some things needs to be formated (ex : function call from string)
        """


        for key in list(self.config["MME_Model"]["architecture"]["MME"].keys()): 
            value = self.config["MME_Model"]["architecture"]["MME"][key] ## On peut pas itérer avec items() puisquele dictionnaire peut changer de taille  
            
            if key.split("_")[0] not in self.config["MME_Model"]["modalities_data"]["modalities"].keys(): 
                del self.config["MME_Model"]["architecture"]["MME"][key]
                continue ### Pour éviter d'utiliser cette modalité dans la suite si on l'a pas inséré dans les data


            if self.config["MME_Model"]["architecture"]["MME"][key]["net_type"] == 'mlp' :
                    module_name,act_funct_name=value["net_config"]["act_fn"].rsplit(".",1)
                    module = __import__(module_name, fromlist=[act_funct_name])
                    self.config["MME_Model"]["architecture"]["MME"][key]["net_config"]["act_fn"]=getattr(module, act_funct_name)


                    module_name,normlayer_name=value["net_config"]["norm_layer"].rsplit(".",1)
                    module = __import__(module_name, fromlist=[normlayer_name])
                    self.config["MME_Model"]["architecture"]["MME"][key]["net_config"]["norm_layer"]=getattr(module, normlayer_name)
            
            self.config["MME_Model"]["architecture"]["MME"][key]["device"]=self.config["global_settings"]["device"]

 

    


    def __verify_config(self): 
        """
        verify here whatever you want on config
        IMPLEMENTER TOUTES LES VERIFS NECESSAIRES 
        """







class ModularModel : 


    def __init__(self,config): 
        """
        
        """

        self.config=config
        self.__verify_config()


    
    def __verify_config(self):

        """
        check whatever you want on config here
        typiquement : vérifier qu'il y'a bien une phase 1 si on n'a pas une phase 2
        """


        pass

    def __phase_1(self) : 
        """Instantiate of load a MME, fit it (constrastive phase) if required
        
        
        """

        self.architecture_config=self.config["Global_Architecture"]
        if  self.architecture_config["MME"]["path"] is None : 

                    MME=MME_Global(self.config,save=False)
                    MME.fit()
                    self.mme=MME.mme

        else : 
             
                raise "il faut implémenter le chargement d'un modèle déja entraîné"
        
        

    def __phase_2(self) :
        

        if self.config["Global_Architecture"]["head"]["type"]== "network" : 
             

             GBM_cfg=self.config["gbm_head"]|{"mme" : self.mme}
             GBM_cfg["mme_cfg"]=self.config["MME_Model"]["architecture"]["MME"]
             if self.config["Global_Architecture"]["MME"]["phase_2_training"] : 
                 GBM_cfg["freeze_mme"]=True
            
             self.GbmNet=instantiate(GBM_cfg,GBMNet)




    def fit(self): 
         
         pass
         
         

            














        

        
    
    

        

        

                                           