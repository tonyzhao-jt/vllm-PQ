# SPDX-License-Identifier: Apache-2.0

from vllm.v1.request import Request


class SWACacheManager:
    """Sliding window attention cache manager."""

    def __init__(
        self,
        sliding_window: int,
    ) -> None:
        self.sliding_window = sliding_window

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
    ):
        pass

    def free(
        self,
        request: Request,
    ):
        pass
