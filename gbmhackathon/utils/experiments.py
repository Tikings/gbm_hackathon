from gbmhackathon.utils import global_wrapper

from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from flatten_dict import flatten, unflatten


class training_experiment : 


    def __init__(self,config) : 




            self.config=config
            self.experiment_config=config["Experiments"]
            self.__setup_output_folder()
            self.result=dict()



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
        
        


    def Multiple_run_experiment(self) : 
         
        """
        
        """
        
        exp_folder=os.path.join(self.experiment_dir,"Multiple_run_experiment")
        if not os.path.exists(exp_folder) : 
             os.makedirs(exp_folder)

        
        config_exp=self.config["Experiments"]["Multiple_run_experiment"]["params"]
        epochs=self.config["MME_Model"]["training"]["epochs"]
        list_run_loss={}
        for run in range(config_exp["n_run"]) : 
             
             mod=global_wrapper.MME_Global(self.config,save=False)
             mod.fit_mme()

             list_run_loss[run]=mod.training["epoch_losses"]
            

        plt.figure(figsize=(8, 5))

        for run in range(len(list(list_run_loss.keys()))):
            plt.plot(range(1, epochs+1), list_run_loss[run], linewidth=1, c='grey', label=f'Run {run+1}')

        
        avg_curve = np.mean(np.stack(list(list_run_loss.values()), axis=0), axis=0)
        self.result["MultiRunResult"]={"list_run_loss" : list_run_loss,"average_loss" : avg_curve}
        
        plt.plot(range(1, epochs+1), avg_curve, linewidth=3, label='Average', zorder=10)   
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves for Each Run with Average')
        plt.legend()
        plt.grid(True)
        fig=plt.gcf()
        fig.savefig(os.path.join(exp_folder,'loss_curves.pdf'),dpi=1000)



                
             
             

class model_comparisons : 

    def __init__(self, config, overrides : dict) :
        """
        overrides should be a dictionary with the format "overrides_name : dict"
        """
        self.original_config=config
        self.__setup_output_folder()
        self.overrided_config=self.instantiate_overrides_config(overrides)




    def instantiate_overrides_config(self,overrides) :    
    
        assert type(overrides)==dict
        overrided_config=dict()

        for conf_name,conf in overrides.items() : 
             
            flatten_overr=flatten(conf,reducer='dot')
            original_flatten=flatten(self.original_config,reducer='dot')

            for overr_key,value in flatten_overr.items(): 
                 
                 assert (overr_key in original_flatten.keys()), "overrides keys not recognized"
                 original_flatten[overr_key]=value

                 overrided_config[conf_name]=unflatten(original_flatten,splitter="dot")

        return overrided_config
    

    def test_rapidos(self) : 
         
        for override,config in self.overrided_config.items(): 
              config["global_settings"]["output_dir"]=os.path.join(config["global_settings"]["output_dir"],override)
              exp=training_experiment(config=config)
              exp.Multiple_run_experiment()
              avg_loss=dict()
              avg_loss[override]=exp.result["MultiRunResult"]["average_loss"]


        max_avg_len=np.max([len(i) for i in avg_loss.values()])
        plt.figure(figsize=(8, 5))

        for conf in avg_loss.keys():
            tab=avg_loss[conf]
            if tab.shape[0] < max_avg_len : 
                tab=np.hstack((tab,np.full((max_avg_len-tab.shape[0]),fill_value=np.nan)))
            
            plt.plot(range(1, max_avg_len+1), tab, linewidth=1, label=" config : {}".format(conf))
        
        plt.savefig(os.path.join(self.experiment_dir,"comparison_loss.pdf"),dpi=1000)



      
    def __setup_output_folder(self) : 

        """
        """

        output_folder=self.original_config["global_settings"]["output_dir"]
        assert os.path.exists(output_folder), "The output folder specified does not exist"

        name_folder="experiment_{}".format(datetime.now().strftime("%Y-%m-%d_%H-%M"))

        os.makedirs(os.path.join(output_folder,name_folder))
        
        self.experiment_dir=os.path.join(output_folder,name_folder)
        

        with open(os.path.join(self.experiment_dir,"config.json"), "w") as f:
            json.dump(self.original_config, f)

        with open(os.path.join(self.experiment_dir,"overrided_config.json"), "w") as f:
            json.dump(self.overrided_config, f)
        

        
        






     
     
        

        






