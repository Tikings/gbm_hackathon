import torch
import torch.nn as nn
import pickle as pkl
import inspect, warnings, json

from typing import get_type_hints, Union, Optional, Tuple, List, Dict, ClassVar

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


def enforce_signature_types(func):
    """
    Decorator that checks runtime argument types against the function's annotations.
    - If a parameter is annotated as float and receives an int, it coerces it to float.
    - Otherwise, raises TypeError on mismatch.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        # Skip 'self'
        for name, value in list(bound.arguments.items())[1:]:
            expected = hints.get(name)
            if expected is None:
                continue  # no type hint, skip

            # Handle Optional and Union types
            origin = getattr(expected, "__origin__", None)
            if origin is Union:
                valid_types = expected.__args__
            else:
                valid_types = (expected,)

            # Special-case: float expected but int provided → coerce
            if expected is float and isinstance(value, int):
                bound.arguments[name] = float(value)
                print(f"Coerced '{name}' with value {value} of type 'int' to 'float'.")
                continue

            # Now do the normal isinstance check
            if not isinstance(bound.arguments[name], valid_types):
                names = ", ".join(
                    getattr(t, "__name__", str(t)) for t in valid_types
                )
                raise TypeError(
                    f"Argument '{name}' to {func.__qualname__} "
                    f"expected type {names}, got {type(value).__name__}"
                )

        return func(self, *bound.args[1:], **bound.kwargs)

    return wrapper

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
