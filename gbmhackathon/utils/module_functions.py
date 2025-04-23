import torch
import torch.nn as nn
import pickle as pkl
import inspect, warnings, json

from typing import Dict, ClassVar


class ParameterNotFoundWarning(UserWarning):
    """Warning when input config parameter not found in the class.__init__ parameters to instantiate"""

    pass


def instantiate(config: Dict, class_object: ClassVar):
    """Flexible instantiation of any class object with the specified config."""
    class_params = [
        param for param in inspect.signature(class_object.__init__).parameters.keys()
    ]

    instanciation_dict = {}
    for key in config.keys():
        if key in class_params:
            instanciation_dict[key] = config[key]
        else:
            warnings.warn(
                f"Parameter {key} not found in init arguments: {class_params}.",
                ParameterNotFoundWarning,
            )
    return class_object(**instanciation_dict)


def load(path_weights: str, path_config: str, module: ClassVar):
    with open(path_config, "rb") as f:
        if path_config.endswith(".json"):
            load_func = json.load
        elif path_config.endswith(".pt") or path_config.endswith(".pth"):
            load_func = torch.load
        else:
            load_func = pkl.load
        config = load_func(f)

    model = instantiate(config, module)
    model.load_state_dict(torch.load(path_weights, weights_only=True))
    model.eval()
    return model
