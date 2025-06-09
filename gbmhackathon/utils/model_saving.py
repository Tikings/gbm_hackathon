import torch
import torch.nn as nn

import os, io
import pickle as pkl
from typing import Union, ClassVar, Tuple, Dict
from pathlib import Path
from copy import deepcopy


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
    cfg = update_cfg_device(cfg, device)
    model = model_class(**cfg)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.device = torch.device(device) if not isinstance(device, torch.device) else device
    
    # Set mode to eval or training
    model.training = training
    print(f"\nModel succesfully loaded on device {model.device}. Model currently in {('inference' if not training else 'training')} mode.")
    return model

def update_cfg_device(cfg_dict: Dict, device: Union[str, torch.device]):
    for key, val in cfg_dict.items():
        if isinstance(val, dict):
            if 'device' in val.keys() and val['device'] != device:
                val['device'] = device
            elif 'device' not in val.keys():
                update_cfg_device(val, device)
    return cfg_dict

class CPU_Unpickler(pkl.Unpickler):
    """Solution to open the embedding file.
    from: https://stackoverflow.com/questions/56369030/runtimeerror-attempting-to-deserialize-object-on-a-cuda-device"""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)