# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import List, Optional, Tuple

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.swa_cache_manager import SWACacheManager
from vllm.v1.request import Request


class PagedCacheManager:

    def __init__(self) -> None:
        self.kv_cache_manager = KVCacheManager()
        self.swa_cache_manager = SWACacheManager()

    def allocate_slots(
        self,
        request: Request,
        num_tokens: int,
        new_computed_blocks: Optional["AllocatedBlocks"] = None,
    ) -> Optional["AllocatedBlocks"]:
        raise NotImplementedError

    def free(
        self,
        request: Request,
    ):
        raise NotImplementedError

    def get_computed_blocks(
        self,
        request: Request,
    ) -> Tuple[Optional["AllocatedBlocks"], int]:
        raise NotImplementedError

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> int:
        raise NotImplementedError

    def reset_prefix_cache(self) -> bool:
        raise NotImplementedError


@dataclass(slots=True)
class AllocatedBlocks:

    new_blocks: List[KVCacheBlock]
    new_swa_blocks: List
