# SPDX-License-Identifier: Apache-2.0
"""Base class for all worker implementations."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.scheduler import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.v1.attention.backends.abstract import AttentionType

logger = init_logger(__name__)


class WorkerBase(ABC):
    """
    Abstract base class for worker implementations across 
    different hardware.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        """
        Initialize common worker components.
        
        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver 
            responsibilities
        """
        # Configuration storage
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        # Distributed settings
        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        self.device: Optional[torch.device] = None
        self.model_runner: Optional[nn.Module] = None
        self.init_gpu_memory: Optional[int] = None

    @abstractmethod
    def init_device(self) -> None:
        """Initialize hardware device and dependencies."""
        raise NotImplementedError

    @abstractmethod
    def load_model(self) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    @abstractmethod
    def get_kv_cache_spec(self) -> KVCacheSpec:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    @abstractmethod
    def initialize_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize key-value cache with given configuration."""
        raise NotImplementedError

    @abstractmethod
    def compile_or_warm_up_model(self) -> None:
        """Prepare model for execution through compilation/warmup."""
        raise NotImplementedError

    @abstractmethod
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        """Execute model with provided scheduler output."""
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Retrieve the underlying model implementation."""
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration."""
        return self.model_config.vocab_size

    @property
    def supports_memory_management(self) -> bool:
        """Indicate if worker supports advanced memory management."""
        return False

    @property
    def attention_type(self) -> "AttentionType":
        """Get attention implementation type from config."""
        return self.model_config.attention_type
