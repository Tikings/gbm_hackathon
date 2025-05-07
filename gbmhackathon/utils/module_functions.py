import torch
import torch.nn as nn
import pickle as pkl
import inspect, warnings, json, functools

from typing import get_type_hints, get_origin, get_args, Union, Optional, Tuple, List, Dict, ClassVar 

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
    Decorator that:
      - Coerces int→float when annotation is float.
      - Checks isinstance against annotation, unwrapping typing generics.
      - Raises TypeError on mismatch.
    """
    sig   = inspect.signature(func)
    hints = get_type_hints(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        # Skip 'self'
        for name, val in list(bound.arguments.items())[1:]:
            expected = hints.get(name)
            if expected is None:
                continue

            # Special-case float annotation + int passed → coerce
            if expected is float and isinstance(val, int):
                bound.arguments[name] = float(val)
                continue

            # Handle Optional[...] and Union[...] (including Optional = Union[..., None])
            origin = get_origin(expected)
            if origin is Union:
                # flatten Union args
                union_args = get_args(expected)
                # if NoneType present, include None
                valid_origins = []
                for arg in union_args:
                    o = get_origin(arg)
                    valid_origins.append(o or arg)
                valid_types = tuple(valid_origins)
            elif origin is not None:
                # typing.Dict, List, Tuple, etc. → use their origin (dict, list, tuple)
                valid_types = (origin,)
            else:
                # plain annotation like int, str, MyClass
                valid_types = (expected,)

            # Now do the check
            if not isinstance(bound.arguments[name], valid_types):
                exp_names = ", ".join(
                    getattr(t, "__name__", str(t)) for t in valid_types
                )
                raise TypeError(
                    f"{func.__qualname__} expected arg '{name}' "
                    f"to be {exp_names}, got {type(val).__name__}"
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
