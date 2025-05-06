import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention, global_mean_pool
from typing import List, Dict, Callable, Iterable, Any, Optional, Union
from torch.jit import Future
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from torch.nn.parallel import parallel_apply

from gbmhackathon.utils.module_functions import instantiate, enforce_signature_types

# Helper functions
def remove_field(cfg: Dict, flag: Union[str,List[str]]) -> Dict:
    match flag:
        case str():
            if flag not in cfg.keys():
                raise KeyError(f"Key {flag} not in dict keys: {cfg.keys()}")
            return {key:value for key, value in cfg.items() if key != flag}
        case list():
            for f in flag:
                if f not in cfg.keys():
                    raise KeyError(f"Key {f} not in dict keys: {cfg.keys()}")
            return {key:value for key, value in cfg.items() if key not in flag}
    
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
    @enforce_signature_types
    def __init__(
        self,
        layers: List[int],
        dropout: List[float] | float,
        act_fn: List[Callable | None] | Callable | None,
        norm_layer: List[Callable | None] | Callable | None,
        enable_residuals: bool = True,
    ):
        """
        Parameters:
        -----------
        layers : List[int]
            A sequence of layer sizes, including input and output dimensions.
            e.g. [in_dim, hidden1, hidden2, ..., out_dim]

        dropout : float or List[float]
            Dropout probability (0–1) applied after each hidden Linear layer.
            If a single float is given, the same rate is used everywhere;
            if a list, its length should match the number of hidden layers.

        act_fn : Callable or List[Callable or None]
            Activation function(s) to insert after each hidden layer.
            Can be a single callable (e.g. nn.ReLU) or a list of the same length
            as the hidden layers, with None to skip activation at specific layers.

        norm_layer : Callable or List[Callable or None]
            Normalization layer(s) to apply after activation.
            Accepts a layer constructor (e.g. nn.LayerNorm) or a matching list
            (with None entries to skip normalization).

        enable_residuals : bool (default: True)
            If True, automatically add skip‑connections between any two Linear layers
            that share the same dimensionality to help gradient flow.
        """
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

        self.network_layers = module_list
        # weights and biases are initialized vy defaults, using appropriate initialize methods.
        # That's why we dont manually do it

        self.enable_residuals = enable_residuals
        self.potential_res_dims: Dict[int, int] = {0:0}
        if self.enable_residuals:
            uniques, counts = np.unique(self.layers_arg, return_counts=True)
            self.potential_res_dims: Dict[int, int] = {dim:idx for idx, dim in enumerate(list(uniques[counts > 1]))}

        if len(self.potential_res_dims) < 1:
            self.enable_residuals = False
            print("No potential residual connections found")

    def forward(self, x):
        if self.enable_residuals:
            possible_residuals: Dict[int, torch.Tensor] = {0:torch.tensor(0)} # to keep TorchScript happy
            res_dim_idx: int = 0
            for layer in self.network_layers:
                if isinstance(layer, nn.Linear):
                    if x.size(1) in self.potential_res_dims:
                        res_dim_idx = self.potential_res_dims[x.size(1)] # get index of the dimension
                        if len(possible_residuals) > res_dim_idx:
                            x = x + possible_residuals[res_dim_idx]
                        elif possible_residuals == {0:0}:
                            possible_residuals = {len(possible_residuals)-1:x}
                        else:
                            possible_residuals.update({len(possible_residuals)-1:x}) # add idx/tensor pair
                x = layer(x)
        else:
            for layer in self.network_layers:
                # if isinstance(layer, nn.Linear):
                #     print(layer.weight.device, layer.bias.device)
                #     print(x.device)
                x = layer(x)
        return x

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


class DropPath(nn.Module):
    @enforce_signature_types
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.check_args()
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class AttentionBlock(nn.Module):
    @enforce_signature_types
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = self.dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv_bias = qkv_bias
        self.qkv = nn.Linear(dim, dim * 3, bias=self.qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

        self.norm = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path)

        self.check_args()
        
    def forward(self, x):
        # Pre-norm
        x_norm = self.norm(x)
        try:
            B, N, C = x_norm.shape
        except:
            B, N = x_norm.shape
            C = 1

        # QKV and split heads
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, C // self.num_heads + 1)
        q, k, v = qkv.unbind(dim=2)  # each shape (B, N, H, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Aggregate and project
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Residual + DropPath
        return x + self.drop_path(out)

class FeedForward(nn.Module):
    @enforce_signature_types
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.act = GEGLU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x_norm = self.norm(x)
        x_ff = self.fc1(x_norm)
        x_ff = self.act(x_ff)
        x_ff = self.fc2(x_ff)
        x_ff = self.drop(x_ff)
        return x + self.drop_path(x_ff)

        
class AttentionNetwork(nn.Module):
    @enforce_signature_types
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        """
        Parameters:
        -----------
        dim : int
            Dimensionality of the input and output features.

        depth : int
            Number of sequential Transformer-style attention blocks.

        num_heads : int
            Number of attention heads in each MultiHeadAttention block.

        mlp_ratio : float (default: 4.0)
            Expansion ratio for the hidden layer size in the MLP block relative to the input dimension.
            For example, if `dim=64` and `mlp_ratio=4.0`, the hidden layer in the MLP will have 64 * 4 = 256 units.

        qkv_bias : bool (default: True)
            If True, enables learnable bias for query, key, and value projections.

        attn_dropout : float (default: 0.0)
            Dropout applied to attention weights.

        proj_dropout : float (default: 0.0)
            Dropout applied after the output projection of the attention block.

        mlp_dropout : float (default: 0.0)
            Dropout applied after the activation in the MLP block.

        drop_path_rate : float (default: 0.1)
            Drop path probability used for stochastic depth regularization.
        """
        super().__init__()
        # stochastic depth scheduling
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # build layers
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                nn.ModuleList([
                    AttentionBlock(
                        dim=dim,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        attn_dropout=attn_dropout,
                        proj_dropout=proj_dropout,
                        drop_path=dpr[i],
                    ),
                    FeedForward(
                        dim=dim,
                        hidden_dim=int(dim * mlp_ratio),
                        dropout=mlp_dropout,
                        drop_path=dpr[i],
                    ),
                ])
            )

        # final normalization
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input tensor of shape [B, N, dim]  (batch, tokens, features)
        """
        for attn, ff in self.blocks:
            x = attn(x)
            x = ff(x)
        return self.norm(x)
        
class GraphEncoder(nn.Module):
    @enforce_signature_types
    def __init__(self,
                in_channels : int,
                hidden_channels : int,
                out_channels : int,
                dropout : float,
                mean_pool : bool = False, 
                activation_post_gat : Callable = F.relu,
                att_agg_activation : Callable = nn.ReLU,
                heads : int = 1,
                ):
        """
        Parameters:
        -----------
        in_channels : int
            Number of input features per node.

        hidden_channels : int
            Number of hidden features in the GAT layer (per head).

        out_channels : int
            Number of output features for the final representation.

        dropout : float
            Dropout probability applied after the GAT activation.

        mean_pool : bool (default: False)
            If True, use global mean pooling; otherwise, use attention-based pooling.

        activation_post_gat : Callable (default: F.relu)
            Activation function applied after the GAT layer.

        att_agg_activation : Callable (default: nn.ReLU)
            Activation function used inside the attention gate MLP for pooling.

        heads : int (default: 1)
            Number of attention heads in the GAT layer.
        """
        super().__init__()

        self.gat = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.dropout = nn.Dropout(dropout)

        self.pooling = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_channels * heads, hidden_channels),
                att_agg_activation(),
                nn.Linear(hidden_channels, 1)
            )
        ) if not mean_pool else global_mean_pool

        self.activation_post_gat = activation_post_gat
        self.fc = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, x, edge_index, batch):

        x = self.gat(x, edge_index)
        x = self.activation_post_gat(x)
        x = self.dropout(x)  
        x = self.pooling(x, batch)
        x = self.fc(x)

        return x

# class ModalityEncoder(nn.Module):
#     """Defines a unified module for all Encoders."""

#     def __init__(
#         self,
#         layers: List[int],
#         dropout: List[float] | float,
#         act_fn: List[Callable | None] | Callable | None,
#         norm_layer: List[Callable | None] | Callable | None,
#         enable_residuals: bool = True,
#         device: Optional[Union[str, torch.device]] = None,
#     ):
#         super().__init__()
#         # Handle device selection - use CUDA if available, otherwise CPU
#         if device is None:
#             self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         else:
#             self.device = torch.device(device)
        
#         print(f"Using device: {self.device}")
        
#         self.mlp: nn.Module = MLP(layers, dropout, act_fn, norm_layer, enable_residuals).to(self.device)
#     def forward(self, x):
#         return self.mlp(x)


    
class ModalityEncoder(nn.Module):
    """Defines a unified module for all Encoders."""
    @enforce_signature_types
    def __init__(
        self,
        config: Dict,
    ):
        super().__init__()
        # Handle device selection - use CUDA if available, otherwise CPU
        device: Union[str, torch.device] = config["device"] if "device" in config.keys() else None
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.net_type = config["net_type"]
        self.config = config
        self.net_config = self.config["net_config"]
        
        if self.net_type == "mlp":
            self.net: nn.Module = instantiate(self.net_config, MLP).to(self.device)
        elif self.net_type == "attention":
            self.net: nn.Module = instantiate(self.net_config, AttentionNetwork).to(self.device)
        elif self.net_type == "graph":
            self.net: nn.Module = instantiate(self.net_config, GraphEncoder).to(self.device)
        else:
            raise ValueError(f"Wrong 'net_type' argument value. Got {self.net_type} but must be either 'attention' or 'mlp' or 'graph'")
            
    def forward(self, x):
        return self.net(x)

class AuxilliaryClassifier(nn.Module):
    """Defines a unified module for auxilliary classifiers."""
    @enforce_signature_types
    def __init__(
        self,
        layers: List[int],
        dropout: List[float] | float,
        act_fn: List[Callable | None] | Callable | None,
        norm_layer: List[Callable | None] | Callable | None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        # Handle device selection - use CUDA if available, otherwise CPU
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.mlp: nn.Module = MLP(layers, dropout, act_fn, norm_layer).to(self.device)
    def forward(self, x):
        return self.mlp(x)

    
class MultiModalEncoder(nn.Module):
    """Global encoder that encompasses all 6 modalities"""
    @enforce_signature_types
    def __init__(
        self,
        hne_cfg: Dict | None = None,
        spatial_cfg: Dict | None = None,
        sc_cfg: Dict | None = None,
        bulk_cfg: Dict | None = None,
        wes_cfg: Dict | None = None,
        clinical_cfg: Dict | None = None,
    ):
        super().__init__()
        # Store configs
        self.hne_cfg = hne_cfg
        self.spatial_cfg = spatial_cfg
        self.sc_cfg = sc_cfg
        self.bulk_cfg = bulk_cfg
        self.wes_cfg = wes_cfg
        self.clinical_cfg = clinical_cfg

        self.modality_net_map = nn.ModuleDict()
        # Instantiate architecture
        if self.hne_cfg is not None:
            # self.hne_net = instantiate(self.hne_cfg, ModalityEncoder)
            self.hne_net = ModalityEncoder(config=self.hne_cfg)
            self.modality_net_map["hne"] = self.hne_net
            
        if self.spatial_cfg is not None:
            # self.spatial_net = instantiate(self.spatial_cfg, ModalityEncoder)
            self.spatial_net = ModalityEncoder(config=self.spatial_cfg)
            self.modality_net_map["spatial"] = self.spatial_net
            
        if self.sc_cfg is not None:
            # self.sc_net = instantiate(self.sc_cfg, ModalityEncoder)
            self.sc_net = ModalityEncoder(config=self.sc_cfg)
            self.modality_net_map["scRNA"] = self.sc_net
            
        if self.bulk_cfg is not None:
            # self.bulk_net = instantiate(self.bulk_cfg, ModalityEncoder)
            self.bulk_net = ModalityEncoder(config=self.bulk_cfg)
            self.modality_net_map["bulk"] = self.bulk_net
            
        if self.wes_cfg is not None:
            # self.wes_net = instantiate(self.wes_cfg, ModalityEncoder)
            self.wes_net = ModalityEncoder(config=self.wes_cfg)
            self.modality_net_map["wes"] = self.wes_net
            
        if self.clinical_cfg is not None:
            # self.clinical_net = instantiate(self.clinical_cfg, ModalityEncoder)
            self.clinical_net = ModalityEncoder(config=self.clinical_cfg)
            self.modality_net_map["clinical"] = self.clinical_net

    # def forward(self, x: Dict[str, torch.Tensor]):
    #     futures: Dict[str, Future[torch.Tensor]] = {}
    #     for name, net in self.modality_net_map.items():
    #         futures[name] = torch.jit.fork(net, x[name])

    #     outputs: Dict[str, torch.Tensor] = {}
    #     for name, fut in futures.items():
    #         outputs[name] = torch.jit.wait(fut)

    #     return outputs

    def forward(self, x: Dict[str, torch.Tensor]):
        outputs = {}
        for modality in self.modality_net_map.keys():
            outputs[modality] = self.modality_net_map[modality](x[modality])
        return outputs

    
class GBMNet(nn.Module):
    """Global model to learn predictive tasks"""
    @enforce_signature_types
    def __init__(
        self,
        head_cfg: Dict,
        include_mme: bool = False,
        hne_cfg: Dict | None = None,
        spatial_cfg: Dict | None = None,
        sc_cfg: Dict | None = None,
        bulk_cfg: Dict | None = None,
        wes_cfg: Dict | None = None,
        clinical_cfg: Dict | None = None,
    ):
        super().__init__()
        # Store configs
        self.head_cfg = head_cfg
        self.include_mme = include_mme

        if include_mme:
            self.hne_cfg = hne_cfg
            self.spatial_cfg = spatial_cfg
            self.sc_cfg = sc_cfg
            self.bulk_cfg = bulk_cfg
            self.wes_cfg = wes_cfg
            self.clinical_cfg = clinical_cfg

            global_mme_cfg = {"hne_cfg": self.hne_cfg,
                             "spatial_cfg":self.spatial_cfg,
                             "sc_cfg":self.sc_cfg,
                             "bulk_cfg":self.bulk_cfg,
                             "wes_cfg": self.wes_cfg,
                             "clinical_cfg": self.clinical_cfg}
            self.mme = MultiModalEncoder(**global_mme_cfg)
        else:
            # None for now but should be replaced with load_model(mme_path)
            self.mme = None

        self.head_net = MLP(**self.head_cfg)

    def forward(self, x: Dict[str, torch.Tensor]):
        if self.mme is not None:
            x = self.mme(x)
            x_list = []
            for pid in x[list(x.keys())[0]].size(0):
                patient_representation = torch.cat([x[mod][pid] for mod in x.keys()], dim=0).unsqueeze()
                x_list.append(patient_representation)
            x = torch.cat(x_list, dim=0)
        outputs = self.head_net(x)
        return outputs