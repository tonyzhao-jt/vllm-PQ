from typing import Any, Dict, List, Optional, Union

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsConfig  # noqa: E501
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import \
    GPTQMarlinConfig
from vllm.model_executor.layers.quantization.gptq_marlin_24 import \
    GPTQMarlin24Config
from vllm.model_executor.layers.quantization.utils.gptq_utils import \
    get_dynamic_override
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, UnquantizedEmbeddingMethod)

logger = init_logger(__name__)


def layer_to_config(
    prefix: str,
    dynamic_value: Dict[str, Dict[str, Union[int, bool]]],
) -> Optional[QuantizationConfig]:
    # prefix here is the layer name
    bit_scheme = {
        # with GPTQMarlin24Config W4A16
        "4": {
            "weight_bits": 4,
            "group_size": 128,
            # for n GPTQ24
            "desc_act": True,
            "is_sym": True,
            "lm_head_quantized": False,
            "dynamic": {},
            "full_config": {},
        },
        "8": {
            "weight_bits": 8,
            "group_size": 128,
            # for n GPTQ24
            "desc_act": True,
            "is_sym": True,
            "lm_head_quantized": False,
            "dynamic": {},
            "full_config": {},
        },
        # with CompressedTensorsConfig W8A8
        "8-tc": {
            "ignore": ["lm_head"],
            "sparsity_scheme_map": {},
            "sparsity_ignore_list": [],
            "quant_format": "int-quantized",
            "target_scheme_map": {
                "Linear": {
                    "weights": QuantizationArgs(
                        num_bits=8,
                        type="int",
                        symmetric=True,
                        group_size=None,
                        trategy="channel",
                        block_structure=None,
                        dynamic=False,
                        actorder=None,
                        observer="minmax",
                        observer_kwargs={},
                    ),
                    "input_activations": QuantizationArgs(
                        num_bits=8,
                        type="int",
                        symmetric=True,
                        group_size=None,
                        strategy="token",
                        block_structure=None,
                        dynamic=True,
                        actorder=None,
                        observer=None,
                        observer_kwargs={},
                    ),
                }
            },
        },
    }
    q_cls = None
    bit = "16"
    if "bits" in dynamic_value:
        bit = dynamic_value["bits"]
    bit = str(bit)
    
    from vllm.platforms import current_platform
    capability_tuple = current_platform.get_device_capability()
    cap = capability_tuple.to_int()
    scheme = bit_scheme[bit] if bit in bit_scheme else None
    gptq_cls = GPTQConfig
    if bit in ["4", "8"]:
        if cap >= 80:
            gptq_cls = GPTQMarlinConfig
        else:
            scheme.pop('is_sym')
            scheme.pop('full_config')
            # scheme['desc_act'] = False
    
    if bit == "16":
        return None
    elif bit == '8-tc':
        q_cls = CompressedTensorsConfig(**scheme)
    elif bit == "8":
        q_cls = gptq_cls(**scheme)
    elif bit == "4":
        q_cls = gptq_cls(**scheme)
    else:
        raise ValueError(f"Unsupported bitwidth {bit}")

    return q_cls


class LLMPQQuantConfig(QuantizationConfig):
    def __init__(
        self,
        dynamic: Dict[str, Dict[str, Union[int, bool]]],
    ) -> None:
        """
        PQ fully relies on the config to provide
        """
        super().__init__()
        self.dynamic = dynamic

    @classmethod
    def get_name(cls) -> str:
        return "llmpq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQConfig":
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        embed_lmhead_bit = cls.get_from_keys_or(config, ["prepost"], default={})
        return cls(dynamic)

    # from compressed-tensors
    def get_cache_scale(self, name: str) -> Optional[str]:
        q_cls = self.layer_to_qconfig(name)
        logger.info(f"name {name}: {q_cls}")
        return q_cls.get_cache_scale(name) if q_cls else None

    def layer_to_qconfig(self, layer_name: str):
        if (
            get_dynamic_override(  # noqa: E712
                self, layer_name=layer_name  # noqa: E712
            )
            == False
        ):  # noqa: E712
            return None

        if layer_name:
            if "lm_head" in layer_name:
                return None
            dynamic_value = get_dynamic_override(self, layer_name)
            if not dynamic_value:
                return None
            logger.info(f"check layer {layer_name} {dynamic_value}")
            return layer_to_config(layer_name, dynamic_value)
        return None

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        assert self.dynamic, "PQ requires dynamic to be set"
        logger.info(f"{prefix}")
        if prefix and "embed" in prefix:
            return UnquantizedEmbeddingMethod()

        if prefix:
            if "lm_head" in prefix:
                return UnquantizedLinearMethod()
            q_cls = self.layer_to_qconfig(prefix)
            if q_cls:
                return q_cls.get_quant_method(layer, prefix)
            else:
                return UnquantizedLinearMethod()
        return None
