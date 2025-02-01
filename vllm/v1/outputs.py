from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class SamplerOutput:

    # [num_reqs]
    sampled_token_ids: torch.Tensor

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: Optional[torch.Tensor]
    # [num_reqs, max_num_logprobs + 1]
    logprobs: Optional[torch.Tensor]
    # [num_reqs]
    sampled_token_ranks: Optional[torch.Tensor]


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use List instead.
@dataclass
class ModelRunnerOutput:

    # [num_reqs]
    req_ids: List[str]
    # req_id -> index
    req_id_to_index: Dict[str, int]

    # [num_reqs]
    sampled_token_ids: List[int]

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids_cpu: Optional[List[List[int]]]
    # [num_reqs, max_num_logprobs + 1]
    logprobs_cpu: Optional[List[List[float]]]
    # [num_reqs]
    sampled_token_ranks_cpu: Optional[List[int]]

    # req_id -> (prompt_logprobs_token_ids, prompt_logprobs,
    #            prompt_token_ranks)
    # [prompt_len, num_prompt_logprobs], [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor]]
