from gbmhackathon.utils import global_wrapper

from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import numpy as np


class training_experiment : 


    def __init__(self,config) : 




            self.config=config
            self.experiment_config=config["Experiments"]
            self.__setup_output_folder()



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
        plt.plot(range(1, epochs+1), avg_curve, linewidth=3, label='Average', zorder=10)   
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves for Each Run with Average')
        plt.legend()
        plt.grid(True)
        fig=plt.gcf()
        fig.savefig(os.path.join(exp_folder,'loss_curves.pdf'),dpi=1000)


                
             
             

             

             

             
             


        






