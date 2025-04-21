import numpy as np

import torch
import torch.nn as nn
from typing import List, Callable, Iterable, Any


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
    input_list = correct_float_int(input_list)
    input_list_types = iterable_types(input_list)

    assert np.all(check_func(input_list_types, check_isin_item, "int")) == True, (
        f"All elements in layers argument must be integers. At least some were not: {layers} -> {layer_types}"
    )


class MLP(nn.Module):
    def __init__(
        self,
        layers: List[int],
        dropout: List[float] | float,
        act_fn: List[Callable] | Callable,
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
                        if dropout[i] != 0:
                            module_list.append(nn.Dropout(dropout[i]))
                    else:
                        if dropout != 0:
                            module_list.append(nn.Dropout(dropout))

                # Adding Activation Function
                if isinstance(act_fn, list):
                    module_list.append(act_fn[i]())
                else:
                    module_list.append(act_fn())

                if i < len(norm_layer):
                    # Adding Normalization Layer
                    if isinstance(norm_layer, list):
                        if norm_layer[i] is not None:
                            module_list.append(norm_layer[i](layer_size))
                    else:
                        if norm_layer is not None:
                            module_list.append(norm_layer(layer_size))

    def validate_params(self):
        # Integrity of the argument 'layers'
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

            assert len(self.dropout_arg) >= len(layers_arg)
        # Integrity of the argument 'act_fn'
        assert isinstance(self.act_fn_arg, list) or callable(self.act_fn_arg), (
            f"act_fn argument must be either a List or callable: {self.act_fn_arg}, {type(self.act_fn_arg)}"
        )
        if isinstance(self.act_fn_arg, list):
            check_func(self.act_fn_arg, check_callable_item)

        # Integrity of the argument 'norm_layers'
        assert isinstance(self.norm_layer_arg, list) or callable(self.norm_layer_arg), (
            f"act_fn argument must be either a List or callable: {self.norm_layer_arg}, {type(self.norm_layer_arg)}"
        )
        if isinstance(self.norm_layer_arg, list):
            check_func(self.norm_layer_arg, check_callable_item, allow_none=True)


# class ModalityEncoder(nn.Module):
#     def __init__(self, layers: List[int], dropout: List[float] | float):
class HnEEncoder(nn.Module):
    """Encoder for HnE data"""

    pass


class SpatialEncoder(nn.Module):
    """Encoder for Spatial transcriptmic data"""

    pass


class BulkEncoder(nn.Module):
    """Encoder for BulkRNAseq data"""

    pass


class SingleCellEncoder(nn.Module):
    """Encoder for scRNAseq data"""

    pass


class WESEncoder(nn.Module):
    """Encoder for Whole Exome Sequencing data"""

    pass


class ClinicalEncoder(nn.Module):
    """Encoder for Clinical data"""

    pass


class MultiModalEncoder(nn.Module):
    """Global encoder that encompasses all 6 modalities"""

    pass
