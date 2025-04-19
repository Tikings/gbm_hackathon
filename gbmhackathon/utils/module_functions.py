import torch
import torch.nn as nn
import inspect, warnings


class ParameterNotFoundWarning(UserWarning):
    """Warning when input config parameter not found in the class.__init__ parameters to instantiate"""

    pass


def instantiate(config, class_object):
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
