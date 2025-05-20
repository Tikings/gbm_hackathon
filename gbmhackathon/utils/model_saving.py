import torch
import torch.nn as nn

import os
import pickle as pkl
from typing import Union, ClassVar, Tuple
from pathlib import Path


def save_model(model: nn.Module, path: str) -> Tuple[str, str]:
    # if there is a directory in the path, make sure the path is available for writing
    if '/' in path:
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
    
    extension_length = len(f".{path.split('.')[-1]}")
    cfg_path = f"{'/'.join(path.split('/')[:-1])}/cfg_{path.split('/')[-1]}"
    
    path = Path(path).resolve()
    cfg_path = Path(cfg_path).resolve()
    model.eval()
    assert 'config' in model.__dict__, "No self.config attribute found, there must be a self.config attribute"
    torch.save(model.state_dict(), path)
    
    with open(cfg_path, 'wb') as f:
        pkl.dump(model.config, f)
    print(f"\nModel saved at path: {path}")
    print(f"Model config saved at cfg_path: {cfg_path}")
    print(f"You can load it by doing loaded_model = load_model(model_class, path, cfg_path, device)") 
    return path, cfg_path

def load_model(model_class: ClassVar, path: str, cfg_path: str, device: Union[str, torch.device], training: bool = False) -> nn.Module:
    device = torch.device(device) if not isinstance(device, torch.device) else device
    with open(cfg_path, 'rb') as f:
        cfg = pkl.load(f)
        
    model = model_class(**cfg)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    # Set mode to eval or training
    model.training = training
    print(f"\nModel succesfully loaded. Model currently in {('inference' if not training else 'training')} mode.")
    return model