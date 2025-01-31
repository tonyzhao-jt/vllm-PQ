from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from vllm._custom_ops import (blockscale_interleave_fp4, cutlass_fp4_gemm,
                              quantize_to_fp4)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    apply_fp8_linear, cutlass_fp8_supported, requantize_with_max_scale)
from vllm.model_executor.parameter import (ModelWeightParameter,
                                           PerTensorScaleParameter)

logger = init_logger(__name__)

QUANT_ALGOS = ["FP8", "NVFP4"]
KV_CACHE_QUANT_ALGOS = ["FP8"]


class ModelOptFp8Config(QuantizationConfig):
    """Config class for ModelOpt FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
    ) -> None:
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning("Detected ModelOpt fp8 checkpoint. Please note that"
                           " the format is experimental and could change.")

    @classmethod
    def get_name(cls) -> str:
        return "modelopt"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelOptFp8Config":
        quant_config = cls.get_from_keys(config, ["quantization"])
        quant_method = quant_config["quant_algo"]
        if quant_method not in QUANT_ALGOS:
            raise ValueError(f"ModelOpt currently only supports: {QUANT_ALGOS}"
                             " quantizations in vLLM. Please check the "
                             "`hf_quant_config.json` file for your model's "
                             "quant configuration.")
        is_checkpoint_fp8_serialized = ("FP8" in quant_method)

        return cls(is_checkpoint_fp8_serialized)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import
        if isinstance(layer, LinearBase):
            return ModelOptFp8LinearMethod(self)
        elif isinstance(layer, Attention):
            return ModelOptFp8KVCacheMethod(self)
        return None


class ModelOptFp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: ModelOptFp8Config):
        super().__init__(quant_config)


class ModelOptFp8LinearMethod(LinearMethodBase):
    """Linear method for Model Optimizer static quantization.
    Supports loading FP8 checkpoints with static weight scale and
    activation scale. Future support might be added for dynamic 
    scales.

    Limitations:
    1. Only support per-tensor quantization due to torch._scaled_mm support.
    2. Only support float8_e4m3fn datatype 
        Args: quant_config: The ModelOpt quantization config.
    """

    def __init__(self, quant_config: ModelOptFp8Config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_checkpoint_fp8_serialized else
                        params_dtype)
        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=weight_dtype),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            weight_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                   weight_loader=weight_loader)
            weight_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("weight_scale", weight_scale)
            # INPUT SCALE
            scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                            weight_loader=weight_loader)

            scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        weight = layer.weight
        max_w_scale = layer.weight_scale.max()
        if not (layer.weight_scale == layer.weight_scale[0]).all():
            max_w_scale, weight = requantize_with_max_scale(
                layer.weight, layer.weight_scale, layer.logical_widths)
        layer.weight = Parameter(weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
        layer.input_scale = Parameter(layer.input_scale.max(),
                                      requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported)


class ModelOptNvFp4Config(QuantizationConfig):
    """Config class for ModelOpt FP4."""

    def __init__(self,
                 is_checkpoint_nvfp4_serialized: bool = False,
                 kv_cache_quant_algo: str = None,
                 group_size: int = None,
                 exclude_modules: List[str] = None) -> None:
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected ModelOpt NVFP4 checkpoint. Please note that"
                " the format is experimental and could change in future.")

            self.group_size = group_size
            self.kv_cache_quant_algo = kv_cache_quant_algo
            self.exclude_modules = exclude_modules

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_nvfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        # TODO : pavanimajety: Move to 100 after blackwell is more accessible
        return 89

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelOptNvFp4Config":
        quant_config = cls.get_from_keys(config, ["quantization"])
        quant_method = quant_config["quant_algo"]
        if quant_method not in QUANT_ALGOS:
            raise ValueError(f"ModelOpt currently only supports: {QUANT_ALGOS}"
                             " quantizations in vLLM. Please check the "
                             "`hf_quant_config.json` file for your model's "
                             "quant configuration.")
        is_checkpoint_nvfp4_serialized = ("NVFP4" in quant_method)
        kv_cache_quant_algo = quant_config["kv_cache_quant_algo"]
        group_size = quant_config["group_size"]
        exclude_modules = quant_config["exclude_modules"]
        if not (group_size and kv_cache_quant_algo and exclude_modules):
            raise ValueError("NVFP4 quantization requires group size and "
                             "kv_cache_quant_algo specified in "
                             "hf_quant_config.json")
        return cls(is_checkpoint_nvfp4_serialized, kv_cache_quant_algo,
                   group_size, exclude_modules)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import
        if isinstance(layer, LinearBase):
            return ModelOptNvFp4LinearMethod(self)
        elif isinstance(layer, Attention):
            return ModelOptFp8KVCacheMethod(self)
        return None


class ModelOptQuantizer(torch.nn.Module):
    """Class to load amax values for Model Opt checkpoints."""

    def __init__(self, _amax, **extra_weight_attrs):
        super().__init__()
        self._amax = _amax
        return

    def forward(self, x):
        return x


class ModelOptNvFp4LinearMethod(LinearMethodBase):
    """Linear method for Model Optimizer NVFP4.
    Supports loading NVFP4 checkpoints with the following structure:
    
    input_scale: torch.float32, scalar ,
    weight: NVFP4(represented as byte) Shape: [1, X, y/2]
    weight_scale: FP8-E4M3, Shape: [X, Y], aka per block scale,
    weight_scale_2: torch.float32, scalar,
    Args: quant_config: The ModelOpt quantization config.
    """

    def __init__(self, quant_config: ModelOptFp8Config):
        self.quant_config = quant_config
        self.cutlass_nvfp4_supported = cutlass_fp8_supported()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        if not self.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError("NVFP4 quantization was selected, "
                             " dynamic quantization is not supported.")
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        if (input_size_per_partition % 16 != 0):
            raise ValueError("Unsupported model when in features size is "
                             "not multiple of 16")
        # The nvfp4 weight is still represented as
        weight_dtype = (torch.float8_e4m3fn
                        if self.quant_config.is_checkpoint_nvfp4_serialized
                        else params_dtype)
        # Weight
        weight = ModelWeightParameter(
            data=torch.empty(
                # 2 fp4 data are packed in the input dimension
                layer.output_size_per_partition,
                layer.input_size_per_partition // 2,
                dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # Input Weight Scale
        input_scale = PerTensorScaleParameter(data=torch.empty(
            len(output_partition_sizes), dtype=torch.float32),
                                              weight_loader=weight_loader)
        layer.register_parameter("input_scale", input_scale)

        # Global Weight Scale
        weight_scale_2 = PerTensorScaleParameter(data=torch.empty(
            len(output_partition_sizes), dtype=torch.float32),
                                                 weight_loader=weight_loader)
        layer.register_parameter("weight_scale_2", weight_scale_2)

        # Per Block Weight Scale
        weight_scale = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition // self.quant_config.group_size,
            dtype=weight_dtype,
        ),
                                            input_dim=1,
                                            output_dim=0,
                                            weight_loader=weight_loader)

        layer.register_parameter("weight_scale", weight_scale)

    def swizzle_weight_scaling_factors(self,
                                       weight_scaling_factor: torch.tensor):
        return weight_scaling_factor

    def process_weights_after_loading(self, layer: Module) -> None:
        # global scales:
        input_scale_2 = layer.input_scale.max().to(torch.float32)
        weight_scale_2 = layer.weight_scale_2.max().to(torch.float32)

        layer.input_scale = Parameter(input_scale_2, requires_grad=False)

        layer.weight_scale_2 = Parameter(weight_scale_2, requires_grad=False)
        layer.alpha = Parameter(layer.input_scale * layer.weight_scale_2,
                                requires_grad=False)

        # swizzle the weight_scale_1
        # contracting dimension is input dimension
        # block_size = 16
        # w_indim  = input_size_per_partition // self.quant_config.group_size
        assert (layer.weight_scale.shape[1] %
                16 == 0), "Expected weight_scale.dim(1) to be divisible by 16"

        # blockscale_interleave_fp4 takes int8 as input
        int8_ws = layer.weight_scale.view(torch.int8).contiguous()
        swizzled_weight_scale = blockscale_interleave_fp4(int8_ws)
        swizzled_weight_scale = swizzled_weight_scale.reshape(
            int8_ws.shape).view(torch.float8_e4m3fn)
        layer.weight_scale_swizzled = Parameter(swizzled_weight_scale,
                                                requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_dtype = x.dtype

        # for input only the contracting dimension has a constraint.
        x_m, _ = x.shape
        w_n, _ = layer.weight.shape
        output_shape = [x_m, w_n]

        # quantize BF16 or FP16 to (FP4 and interleaved block scale)
        qinput, x_sf_1 = quantize_to_fp4(x, 1 / layer.input_scale)

        # validate dtypes of quantized input, weight and weight_sf
        assert (qinput.dtype == torch.uint8)
        assert (x_sf_1.dtype == torch.int32)
        assert (layer.weight.dtype == torch.uint8)
        assert (layer.weight_scale_swizzled.dtype == torch.float8_e4m3fn)
        assert (layer.alpha.dtype == torch.float32)

        # step 2 perform gemm
        workspace_bytes = 0
        workspace = torch.empty((workspace_bytes, ))

        out = cutlass_fp4_gemm(qinput, layer.weight.t(), x_sf_1,
                               layer.weight_scale_swizzled.view(torch.int32),
                               layer.alpha, workspace, workspace_bytes,
                               output_dtype)
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
