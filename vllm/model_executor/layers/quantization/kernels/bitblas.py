from typing import Optional, Tuple, List, Dict

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.bitblas_utils import (
    BITBLAS_SUPPORTED_GROUP_SIZES,
    BITBLAS_OPTIMIZE_FEATURES,
    apply_gptq_bitblas_linear,
    check_bitblas_supports_shape,
    bitblas_is_k_full,
    bitblas_make_empty_g_idx,
    bitblas_sort_g_idx,
    query_bitblas_supported_quant_types,
    unpack_gptq_qweight,
    unpack_gptq_qzeros,
)

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from .MPLinearKernel import MPLinearKernel, MPLinearLayerConfig

logger = init_logger(__name__)

class BitBLASLinearKernel(MPLinearKernel):

    OPT_FEATURES: List[int] = BITBLAS_OPTIMIZE_FEATURES
    ENABLE_TUNING: bool = True
    MATMUL_LAYOUT: str = "nt"
    BITBLAS_DTYPES: Dict[torch.dtype, str] = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.half: "float16",
        torch.int8: "int8",
    }
    bitblas_matmul: object = None

    def __init__(self,
        c: MPLinearLayerConfig,
        w_q_param_name: str,
        w_s_param_name: str,
        w_zp_param_name: Optional[str] = None,
        w_gidx_param_name: Optional[str] = None,
        bitblas_quant_config: QuantizationConfig = None,
    ):
        self.quant_config = bitblas_quant_config
        print(w_q_param_name, w_s_param_name, w_zp_param_name, w_gidx_param_name)
        super().__init__(c, w_q_param_name, w_s_param_name, w_zp_param_name,
                            w_gidx_param_name)

    def repack_bitblas_from_gptq(
        self,
        b_q_weight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
    ):
        from bitblas.quantization.utils import general_compress
        assert self.bitblas_matmul is not None, "bitblas_matmul is None"

        # qweight in gptq old quant linear stored with
        # (outfeatures, infeatures), should be transposed.
        qweight = b_q_weight.T.contiguous().view(
            self.quant_config.TORCH_BITBLAS_STORAGE_DTYPE)
        intweight = unpack_gptq_qweight(qweight,
                                   self.quant_config.weight_bits).contiguous()
        if self.bitblas_matmul.weight_transform is not None:
            qweight = self.bitblas_matmul.weight_transform(
                intweight.cpu()).cuda()
        # scales in gptq old quant linear stored with
        # (infeatures // group_size, outfeatures), should be transposed.
        scales = scales.T.contiguous()
        # qzeros should be de-quantized to int zeros.
        intzeros = unpack_gptq_qzeros(qzeros,
                                 self.quant_config.weight_bits).T.contiguous()
        zeros: Optional[torch.Tensor] = None
        if self.bitblas_matmul.config.zeros_mode == "original":
            zeros = intzeros.to(torch.float16).contiguous()
        elif self.bitblas_matmul.config.zeros_mode == "rescale":
            assert zeros is not None, "zeros should not be None"
            zeros[:, :] = intzeros.to(torch.float16)[:, :] * scales[:, :]
        elif self.bitblas_matmul.config.zeros_mode == "quantized":
            zeros = (torch.Tensor(
                general_compress(
                    intzeros.T.contiguous().cpu().numpy(),
                    self.quant_config.weight_bits,
                )).to(qweight.device).to(
                    self.quant_config.TORCH_BITBLAS_STORAGE_DTYPE).contiguous(
                    ))
        else:
            raise ValueError("Unsupported zeros type: {}".format(
                self.bitblas_matmul.config.zeros_mode))

        return qweight, scales, zeros

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @classmethod
    def can_implement(cls,
                      c: MPLinearLayerConfig) -> Tuple[bool, Optional[str]]:
        if c.zero_points:
            return False, "Zero points currently not supported by "\
                          " BitBLASLinearKernel. Will be added when AWQBitBLAS "\
                          "is migrated over to using MPLinearKernel backend"

        quant_types = query_bitblas_supported_quant_types(c.zero_points)
        if c.weight_type not in quant_types:
            return False, f"Quant type ({c.weight_type}) not supported by"\
                          f"  BitBLAS, supported types are: {quant_types}"

        if c.group_size not in BITBLAS_SUPPORTED_GROUP_SIZES:
            return False, f"Group size ({c.group_size}) not supported by "\
                            "BitBLAS, supported group sizes are: "\
                            f"{BITBLAS_SUPPORTED_GROUP_SIZES}"

        return check_bitblas_supports_shape(
            c.partition_weight_shape[1],  # out_features
            c.partition_weight_shape[0],  # in_features
            c.full_weight_shape[0],  # in_features
            c.group_size)

    # note assumes that
    #  `weight_packed` is: {input_dim = 0, output_dim = 1, packed_dim = 0}
    #  `weight_scale` is: {input_dim = 0, output_dim = 1}
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        device = getattr(layer, self.w_q_name).device
        c = self.config

        row_parallel = (c.partition_weight_shape[0] != c.full_weight_shape[0])
        self.is_k_full = bitblas_is_k_full(c.has_g_idx, row_parallel)

        # Default names since bitblas requires empty parameters for these,
        # TODO: remove this requirement from bitblas (allow optional tensors)
        if self.w_gidx_name is None:
            self.w_gidx_name = "g_idx"
        if self.w_zp_name is None:
            self.w_zp_name = "w_zp"

        if c.has_g_idx:
            g_idx, g_idx_sort_indices = bitblas_sort_g_idx(
                getattr(layer, self.w_gidx_name))
            self._transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(layer, self.w_gidx_name, bitblas_make_empty_g_idx(device))
            layer.g_idx_sort_indices = bitblas_make_empty_g_idx(device)

        if c.zero_points:
            raise NotImplementedError("Zero points not supported by BitBLAS")
        else:
            setattr(layer, self.w_zp_name, bitblas_make_empty_g_idx(device))

        # Newly generated tensors need to replace existing tensors that are
        # already registered as parameters by vLLM (and won't be freed)
        def replace_tensor(name, new_t):
            # It is important to use copy_() here since it ensures
            # the same buffer is reused
            getattr(layer, name).copy_(
                new_t.view(getattr(layer, name).dtype).view(
                    getattr(layer, name).shape))
            del new_t

        print("layer.qzeros", layer.qzeros)
        # Repack weights
        bitblas_qweight, bitblas_scales, bitblas_qzeros = (
            self.repack_bitblas_from_gptq(
                layer.qweight,
                layer.scales,
                layer.qzeros,
            ))
        replace_tensor(self.w_q_name, bitblas_qweight)
        replace_tensor(self.w_s_name, bitblas_scales)
        replace_tensor(self.w_zp_name, bitblas_qzeros)

    def configure_bitblas_matmul(
        self,
        infeatures: int,
        outfeatures: int,
        params_dtype: torch.dtype,
        bias: bool,
    ) -> None:
        enable_tuning = self.ENABLE_TUNING
        layout = self.MATMUL_LAYOUT
        bits = self.quant_config.weight_bits
        self._configure_bitblas_matmul(
            infeatures,
            outfeatures,
            params_dtype,
            enable_tuning,
            bias,
            layout,
            bits,
        )

    def _configure_bitblas_matmul(
        self,
        infeatures,
        outfeatures,
        params_dtype,
        enable_tuning,
        bias,
        layout,
        bits,
    ):
        from bitblas import MatmulConfig
        bitblas_dtype = self.BITBLAS_DTYPES[params_dtype]

        with_scaling = False
        with_zeros = False
        group_size = self.quant_config.group_size
        zeros_mode = self.quant_config.zeros_mode
        if self.quant_config.quant_method == "gptq":
            with_scaling = True
            with_zeros = True
            W_dtype = f"uint{bits}"
            if self.quant_config.is_sym:
                with_zeros = False
                W_dtype = f"int{bits}"
        else:
            raise ValueError(
                f"Unsupported quant_method {self.quant_config.quant_method}")

        matmul_config = MatmulConfig(
            M=self.OPT_FEATURES,
            N=outfeatures,
            K=infeatures,
            A_dtype=bitblas_dtype,
            W_dtype=W_dtype,
            out_dtype=bitblas_dtype,
            accum_dtype="int32" if bitblas_dtype == "int8" else bitblas_dtype,
            storage_dtype=self.quant_config.GPTQ_BITBLAS_STORAGE_DTYPE,
            with_scaling=with_scaling,
            with_zeros=with_zeros,
            group_size=group_size,
            with_bias=bias,
            layout=layout,
            zeros_mode=zeros_mode,
        )
        self.bitblas_matmul = self._get_or_create_bitblas_operator(
            matmul_config, enable_tuning)

    def _get_or_create_bitblas_operator(self, config, enable_tuning):
        from bitblas import Matmul, auto_detect_nvidia_target
        from bitblas.cache import get_database_path, global_operator_cache
        BITBLAS_DATABASE_PATH = get_database_path()
        BITBLAS_TARGET = auto_detect_nvidia_target()

        if global_operator_cache.size() == 0:
            global_operator_cache.load_from_database(BITBLAS_DATABASE_PATH,
                                                     BITBLAS_TARGET)

        bitblas_matmul = global_operator_cache.get(config)
        if bitblas_matmul is None:
            bitblas_matmul = Matmul(config,
                                    target=BITBLAS_TARGET,
                                    enable_tuning=False)
            if enable_tuning:
                bitblas_matmul.hardware_aware_finetune(topk=20)
                global_operator_cache.add(config, bitblas_matmul)
                global_operator_cache.save_into_database(
                    BITBLAS_DATABASE_PATH, BITBLAS_TARGET)
                TUNING_MESSAGE = (
                    f"BitBLAS Operator {config} tuned and saved to database.")
                logger.info(TUNING_MESSAGE)
            else:
                _message = f"BitBLAS Operator {config} created without tuning. "
                logger.info(_message)
        else:
            _message = f"BitBLAS Operator {config} retrieved from cache."
            logger.info(_message)
        return bitblas_matmul

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self.config
        w_q, w_s, w_zp, w_gidx = self._get_weight_params(layer)

        # `process_weights_after_loading` will ensure w_zp and w_gidx are not
        #  None for bitblas
        return apply_gptq_bitblas_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            g_idx=w_gidx,  # type: ignore
            g_idx_sort_indices=layer.g_idx_sort_indices,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            is_k_full=self.is_k_full,
            bias=bias
        )
