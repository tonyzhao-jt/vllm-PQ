# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreOutput, FinishReason


@dataclass
class SchedulerStats:
    """Stats associated with the scheduler."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0

    gpu_cache_usage: float = 0.0
    # gpu_prefix_cache_hit_rate: float = 0.0

    scheduled_new_reqs: Optional[List[str]] = None


@dataclass
class RequestStateStats:
    """Stats that need to be tracked across delta updates."""

    num_generation_tokens: int = 0
    arrival_time: float = 0.0
    first_scheduled_time: float = 0.0
    first_token_time: float = 0.0
    last_token_time: float = 0.0


@dataclass
class FinishedRequestStats:
    """Stats associated with a finished request."""

    finish_reason: "FinishReason"
    e2e_latency: float = 0.0
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    inference_time: float = 0.0
    decode_time: float = 0.0


class IterationStats:
    """Stats associated with a single set of EngineCoreOutputs."""

    def __init__(self, log_stats: bool):
        self.log_stats = log_stats
        self.iteration_timestamp = time.time()
        self.num_generation_tokens = 0
        self.num_prompt_tokens = 0
        self.finished_requests: List[FinishedRequestStats] = []
        self.time_to_first_tokens_iter: List[float] = []
        self.time_per_output_tokens_iter: List[float] = []
        self.queue_times_iter: List[float] = []
        self.prefill_times_iter: List[float] = []

    def _time_since(self, start: float) -> float:
        """Calculate an interval relative to this iteration's timestamp."""
        return self.iteration_timestamp - start

    def update_from_output(self, output: "EngineCoreOutput",
                           is_prefilling: bool, prompt_len: int,
                           request_state_stats: RequestStateStats):
        if not self.log_stats:
            return

        num_new_generation_tokens = len(output.new_token_ids)
        last_token_latency = self._time_since(
            request_state_stats.last_token_time)

        self.num_generation_tokens += num_new_generation_tokens
        if is_prefilling:
            # This relies on the invariant that EngineCore does
            # not stream outputs for partially completed prefills
            # (scheduler.update_from_output makes EngineCoreOutput
            # iff num_computed_tokens == num_tokens).
            assert (num_new_generation_tokens > 0)
            self.num_prompt_tokens += prompt_len
            request_state_stats.first_token_time = self.iteration_timestamp

            self.time_to_first_tokens_iter.append(last_token_latency)

            prefill_time = self._time_since(
                request_state_stats.first_scheduled_time)
            self.prefill_times_iter.append(prefill_time)
        else:
            self.time_per_output_tokens_iter.append(last_token_latency)

        request_state_stats.num_generation_tokens += num_new_generation_tokens
        request_state_stats.last_token_time = self.iteration_timestamp

    def update_from_newly_scheduled(self,
                                    request_state_stats: RequestStateStats):
        request_state_stats.first_scheduled_time = self.iteration_timestamp
        queue_time = self._time_since(request_state_stats.arrival_time)
        self.queue_times_iter.append(queue_time)

    def update_from_finished_request(self, finish_reason: "FinishReason",
                                     request_output: "RequestOutput",
                                     request_state_stats: RequestStateStats):
        e2e_latency = self._time_since(request_state_stats.arrival_time)
        inference_time = self._time_since(
            request_state_stats.first_scheduled_time)
        decode_time = self._time_since(request_state_stats.first_token_time)

        finished_req = \
            FinishedRequestStats(finish_reason=finish_reason,
                                 e2e_latency=e2e_latency,
                                 num_prompt_tokens=len(request_output.prompt_token_ids),
                                 num_generation_tokens=request_state_stats.num_generation_tokens,
                                 inference_time=inference_time,
                                 decode_time=decode_time)
        self.finished_requests.append(finished_req)
