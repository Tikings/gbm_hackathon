

import argparse
#import mlflow
import json
import models 
from flatten_dict import flatten
import torchvision.transforms as transforms
import torchvision
import torch
import training

### Finalement j'utilise pas hydra, argparse + json fonctionne 




def main(config : dict) : 

    ######################################################################################
    #### Un peu de blabla fourni par chatgpt, propre à CIFAR1O et on s'en fout un peu#####
    ######################################################################################
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisation
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["training"]["batch_size"], shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck') 
    
    #####################################################################################
    #####################################################################################
    #####################################################################################



    mlflow.set_experiment(config["MLFLow"]["experiment_name"])

    with mlflow.start_run(config["MLFLow"]["run_name"]) : 

        mlflow.log_param( flatten(config["training"]|config["model"],reducer="dot") )
        model=models.SimpleCNN(conv_size=config["model"]["conv_size"])

        training.train(model,config=config["training"],trainloader=trainloader)

    







    pass




parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='fichier config')
args = parser.parse_args()


with open(args.config, 'r') as f:
    config = json.load(f)


main(config)


















