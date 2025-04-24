import numpy as np

import torch
import torch.nn as nn
from typing import List, Dict, Callable, Iterable, Any

from gbmhackathon.utils.module_functions import instantiate


def correct_float_int(layer_list: List):
    """Identify ints which were passed as floating numbers"""
    layers = []
    for item in layer_list:
        if isinstance(item, float):
            item_str = str(item)
            decimal_str = item_str[item_str.index(".") + 1 :]
            decimal_length = len(decimal_str)
            if decimal_str == "0" * decimal_length:
                layers.append(int(item))
        else:
            layers.append(item)
    return layers


def iterable_types(iterable: Iterable):
    """Retruns a list of each element's type as strings"""
    out = []
    for element in iterable:
        out.append(str(type(element)))
    return out


def check_isin_item(item: str, pattern: str):
    """Check if a pattern is in another string"""
    return pattern in item


def check_instance(item: Any, instance: Any):
    """Check if an item is of the given instance"""
    return isinstance(item, instance)


def check_callable_item(item: Any, allow_none: bool = False):
    """Check if any input item is a callable (or either a callable or None if allow_none)"""
    if allow_none:
        return callable(item) or (item is None)
    return callable(item)


def check_func(iterable: Iterable, func: Callable, *args, **kwargs):
    """Applies func to any iterable and return results as a list"""
    return list(map(lambda p: func(p, *args, **kwargs), iterable))


def check_types_list(input_list: List, type_str: str):
    """Verifies that each element in the list is of the correct type"""
    input_list = correct_float_int(input_list)
    input_list_types = iterable_types(input_list)

    assert np.all(check_func(input_list_types, check_isin_item, "int")) == True, (
        f"All elements in layers argument must be integers. At least some were not: {input_list} -> {input_list_types}"
    )


class MLP(nn.Module):
    """Multi Layer Perceptron with dropout and normalizaiton layers that can be instantiated dynamically"""

    def __init__(
        self,
        layers: List[int],
        dropout: List[float] | float,
        act_fn: List[Callable | None] | Callable | None,
        norm_layer: List[Callable | None] | Callable | None,
    ):
        super().__init__()
        self.layers_arg = layers
        self.dropout_arg = dropout
        self.act_fn_arg = act_fn
        self.norm_layer_arg = norm_layer

        self.validate_params()

        module_list = nn.ModuleList()

        for i, layer_size in enumerate(layers):
            if i == 0:
                # Skip the input layer dimension
                pass
            else:
                module_list.append(nn.Linear(layers[i - 1], layer_size))

                if (
                    i != len(layers) - 1
                ):  # If we are on the output layer, prevent dropout
                    # Adding dropout
                    if isinstance(dropout, list):
                        if dropout[i - 1] != 0:
                            module_list.append(nn.Dropout(dropout[i - 1]))
                    else:
                        if dropout != 0:
                            module_list.append(nn.Dropout(dropout))

                # Adding Activation Function
                if isinstance(act_fn, list):
                    if act_fn[i - 1] is not None:
                        module_list.append(act_fn[i - 1]())
                else:
                    if act_fn is not None:
                        module_list.append(act_fn())

                # Adding Normalization Layer
                if isinstance(norm_layer, list):
                    if norm_layer[i - 1] is not None:
                        module_list.append(norm_layer[i - 1](layer_size))
                else:
                    if norm_layer is not None:
                        module_list.append(norm_layer(layer_size))

        self.network = module_list

        # weights and biases are initialized vy defaults, using appropriate initialize methods.
        # That's why we dont manually do it

    def forward(self, x):
        return self.network(x)

    def validate_params(self):
        """Checks the validity of the MLP init parameters"""
        # Integrity of the argument 'layers'
        assert len(self.layers_arg) > 1, (
            f"Not enough layers in layers argument, must be at least of length 2: {self.layers_arg}"
        )
        assert isinstance(self.layers_arg, list), (
            f"layers argument must be a List (of integers): {type(self.layers_arg)}"
        )
        check_types_list(self.layers_arg, "int")

        # Integrity of the argument 'dropout'
        assert isinstance(self.dropout_arg, list) or isinstance(
            self.dropout_arg, float
        ), (
            f"dropout argument must be either a List or a single float: {type(self.dropout_arg)}"
        )
        if isinstance(self.dropout_arg, list):
            check_types_list(self.layers_arg, "float")

            # when we define a dropout for output layer (-1 for input size) (it will be skipped during instantiation)
            # Appropriate definition (-1 for input size, -1 for output size)
            assert (
                len(self.dropout_arg) == len(self.layers_arg) - 1
                or len(self.dropout_arg) == len(self.layers_arg) - 2
            ), (
                f"Wrong dropout list length must be either len(layers) - 1 ({len(self.layers_arg) - 1}) or len(layers) - 2 ({len(self.layers_arg) - 2}) but found {len(self.dropout_arg)}"
            )
        # Integrity of the argument 'act_fn'
        assert (
            isinstance(self.act_fn_arg, list)
            or callable(self.act_fn_arg)
            or self.act_fn_arg is None
        ), (
            f"act_fn argument must be either a List or callable: {self.act_fn_arg}, {type(self.act_fn_arg)}"
        )
        if isinstance(self.act_fn_arg, list):
            check_func(self.act_fn_arg, check_callable_item, allow_none=True)

            # -1: when we define a act_fn for output layer (-1 for input size) (it will be skipped during instantiation)
            assert len(self.act_fn_arg) == len(self.layers_arg) - 1, (
                f"Wrong act_fn list length must be len(layers) - 1 ({len(self.layers_arg) - 1}) but found {len(self.act_fn_arg)}"
            )
        # Integrity of the argument 'norm_layers'
        assert (
            isinstance(self.norm_layer_arg, list)
            or callable(self.norm_layer_arg)
            or self.norm_layer_arg is None
        ), (
            f"norm_layer argument must be either a List, callable or None: {self.norm_layer_arg}, {type(self.norm_layer_arg)}"
        )
        if isinstance(self.norm_layer_arg, list):
            check_func(self.norm_layer_arg, check_callable_item, allow_none=True)

            # -1: when we define a norm_layer for output layer (-1 for input size) (it will be skipped during instantiation)
            assert len(self.norm_layer_arg) == len(self.layers_arg) - 1, (
                f"Wrong norm_layer list length must be len(layers) - 1 ({len(self.layers_arg) - 1}) but found {len(self.norm_layer_arg)}"
            )


class ModalityEncoder(nn.Module):
    """Defines a unified module for all Encoders."""

    def __init__(
        self,
        layers: List[int],
        dropout: List[float] | float,
        act_fn: List[Callable | None] | Callable | None,
        norm_layer: List[Callable | None] | Callable | None,
    ):
        super().__init__()
        self.mlp = MLP(layers, dropout, act_fn, norm_layer)

    def forward(self, x):
        return self.mlp(x)


@torch.jit.script  # For parallel processing
class MultiModalEncoder(nn.Module):
    """Global encoder that encompasses all 6 modalities"""

    def __init__(
        self,
        hne_cfg: Dict,
        spatial_cfg: Dict,
        sc_cfg: Dict,
        bulk_cfg: Dict,
        wes_cfg: Dict,
        clinical_cfg: Dict,
    ):
        super().__init__()
        # Store configs
        self.hne_cfg = hne_cfg
        self.spatial_cfg = spatial_cfg
        self.sc_cfg = sc_cfg
        self.bulk_cfg = bulk_cfg
        self.wes_cfg = wes_cfg
        self.clinical_cfg = clinical_cfg

        # Instantiate architecture
        self.hne_net = instantiate(self.hne_cfg, ModalityEncoder)
        self.spatial_net = instantiate(self.spatial_cfg, ModalityEncoder)
        self.sc_net = instantiate(self.sc_cfg, ModalityEncoder)
        self.bulk_net = instantiate(self.bulk_cfg, ModalityEncoder)
        self.wes_net = instantiate(self.wes_cfg, ModalityEncoder)
        self.clinical_net = instantiate(self.clinical_cfg, ModalityEncoder)

        self.modality_net_map = {
            "hne": self.hne_net,
            "spatial": self.spatial_net,
            "bulk": self.bulk_net,
            "scRNA": self.sc_net,
            "clinical": self.clinical_net,
            "wes": self.wes_net,
        }

    def forward(self, x: Dict[str, torch.Tensor]):
        futures_list = []
        for key in self.modality_net_map.keys():
            modality_tensor = x[key]
            futures_list.append(
                torch.jit.fork(self.modality_net_map[key], modality_tensor)
            )

        return {
            key: torch.jit.wait(future)
            for key, future in zip(self.modality_net_map.keys(), futures_list)
        }
