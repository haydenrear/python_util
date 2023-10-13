import abc
import dataclasses
from typing import Optional

import torch.nn

from drools_py.configs.config_models import EdgeDim, NumAttnHeads, OutDim, DoSkipConnections, LayerNormEps
from drools_py.configs.config import Config


class ExternalTorchModuleConfig(Config, abc.ABC):
    pass


class MetaConfig(Config):

    def __init__(self, class_type: str):
        self.class_type = class_type

    @staticmethod
    def test_properties(**kwargs) -> dict:
        return MetaConfig.update_override(
            kwargs,
            MetaConfig("").to_dict()
        )


@dataclasses.dataclass
class LayerNormConfig(ExternalTorchModuleConfig):

    # TODO: option to make DeepNet
    def __init__(self, eps: LayerNormEps, size: EdgeDim):
        self.eps = eps
        self.size = size

    def to_layer_norm(self) -> torch.nn.LayerNorm:
        return torch.nn.LayerNorm(self.size.config_option,
                                  self.eps.config_option)

    @staticmethod
    def test_properties(**kwargs) -> dict:
        return LayerNormConfig.update_override(kwargs, LayerNormConfig(
            eps=LayerNormEps(1e-12),
            size=EdgeDim(10)
        ).to_self_dictionary())

    @classmethod
    def validation_properties(cls, **kwargs) -> dict:
        return LayerNormConfig.update_override(kwargs, LayerNormConfig(
            eps=LayerNormEps(1e-12),
            size=EdgeDim(2048)
        ).to_self_dictionary())

    @classmethod
    def layer_norm_config_dim_override(cls, dim: int = 512):
        return LayerNormConfig.build_validation_config(
            **LayerNormConfig(
                size=EdgeDim(dim),
                **LayerNormConfig.validation_properties_minus(["size"]),
            ).to_self_dictionary()
        )


class MultiHeadAttentionModuleConfig(ExternalTorchModuleConfig):

    def __init__(self,
                 embed_dim: EdgeDim,
                 kdim: EdgeDim,
                 vdim: EdgeDim,
                 n_attn_heads: NumAttnHeads):
        self.n_attn_heads = n_attn_heads
        self.vdim = vdim
        self.kdim = kdim
        self.embed_dim = embed_dim

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass


class LinearTorchModuleConfig(ExternalTorchModuleConfig):

    def __init__(self,
                 in_features: Optional[EdgeDim] = None,
                 out_features: Optional[OutDim] = None,
                 bias: Optional[DoSkipConnections] = None):
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass


class ProbabilityLayerTchModuleConfig(ExternalTorchModuleConfig):
    """
    Softmax
    """

    def __init__(self,
                 in_features: Optional[EdgeDim] = None):
        self.in_features = in_features

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass


class SequentialModuleConfig(ExternalTorchModuleConfig):

    def __init__(self, module_configs: list[Config]):
        self.module_configs = module_configs

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass


class ModuleListModuleConfig(ExternalTorchModuleConfig):

    def __init__(self, module_configs: list[Config]):
        self.module_configs = module_configs

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass


class ModuleDictModuleConfig(ExternalTorchModuleConfig):

    def __init__(self, module_configs: dict[str, Config]):
        self.module_configs = module_configs

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass


class ActivationFunctionModuleConfig(ExternalTorchModuleConfig):

    def __init__(self, in_features: EdgeDim):
        self.in_features = in_features

    @staticmethod
    def test_properties(**kwargs) -> dict:
        pass
