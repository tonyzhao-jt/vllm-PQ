import itertools
from typing import List, Tuple

import pytest
import torch

from tests.kernels.utils import override_backend_env_variable
from tests.v1.sample.utils import (
    assert_incr_detok_str_matches_non_incr_detok_str,
    compute_correct_cumulative_logprob, get_test_batch)
from vllm import SamplingParams

from ...conftest import VllmRunner

MODELS = ["facebook/opt-125m"]


def _repeat_logprob_config(
    test_prompts,
    logprob_prompt_logprob_list: List[Tuple],
) -> List[Tuple]:
    """Ensure each test prompt has a logprob config.
    
    A logprob config specifies the optional (i.e.
    may-be-`None`) number of sample logprobs and
    the optional number of prompt logprobs.

    If more test prompts than logprob configs are
    provided, the provided logprob configs are
    tiled to match the number of test prompts.

    If fewer test prompts than logprob configs
    are provided, the list of logprob configs
    is truncated to match the number of test
    prompts.

    Otherwise, the list of logprob configs
    is returned as-is.

    Args:
      test_prompts: list of prompts under test
      logprob_prompt_logprob_list: list of
                            (optional num sample logprob,
                             optional num prompt logprob)
                             tuples
    
    Returns:
      List of
      (optional num sample logprob,optional num prompt logprob)
      tuples which is either identical to
      `logprob_prompt_logprob_list`, or else repeats
      `logprob_prompt_logprob_list` enough times to match the
      number of `test_prompts`, or else is truncated to match
      the number of `test_prompts`
    """
    num_test_prompts = len(test_prompts)
    # Make sure there is a logprobs configuration for each test prompt
    logprob_prompt_logprob_list = list(
        itertools.islice(itertools.cycle(logprob_prompt_logprob_list),
                         num_test_prompts))
    # Now the number of prompts should match the number of sample params combos
    assert num_test_prompts == len(logprob_prompt_logprob_list)
    return logprob_prompt_logprob_list


def _test_case_get_logprobs_and_prompt_logprobs(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
    detokenize: bool,
    batch_logprobs_composition: str,
    max_num_batched_tokens: int,
    enable_prefix_caching: bool,
    example_prompts,
    monkeypatch,
) -> None:
    test_prompts = example_prompts
    override_backend_env_variable(monkeypatch, "FLASH_ATTN")

    max_num_seqs = 128
    max_num_batched_tokens = 128
    max_model_len = 128

    max_tokens = 5
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(
            test_prompts,
            max_tokens=max_tokens,
        )
        hf_logprobs = hf_model.generate_greedy_logprobs(
            test_prompts,
            max_tokens=max_tokens,
        )

    # Batch has mixed sample params
    # (different logprobs/prompt logprobs combos)
    logprob_prompt_logprob_list = get_test_batch(batch_logprobs_composition)

    # Ensure that each test prompt has a logprob config for testing
    logprob_prompt_logprob_list = _repeat_logprob_config(
        test_prompts, logprob_prompt_logprob_list)
    # Generate SamplingParams
    vllm_sampling_params = [
        SamplingParams(max_tokens=max_tokens,
                       logprobs=lp,
                       prompt_logprobs=plp,
                       temperature=0.0,
                       detokenize=detokenize)
        for lp, plp in logprob_prompt_logprob_list
    ]

    with vllm_runner(
            model,
            dtype=dtype,
            max_logprobs=7,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            enforce_eager=True,
            enable_prefix_caching=enable_prefix_caching,
    ) as vllm_model:
        vllm_results = vllm_model.model.generate(
            test_prompts, sampling_params=vllm_sampling_params)

    for vllm_result, hf_logprob, hf_output, logprob_prompt_logprob in zip(
            vllm_results, hf_logprobs, hf_outputs,
            logprob_prompt_logprob_list):

        # Extract request-level (prompt)logprobs config
        num_top_logprobs = logprob_prompt_logprob[0]
        num_top_prompt_logprobs = logprob_prompt_logprob[1]

        # Test whether sampled token output is consistent between vLLM and HF
        # vLLM prompt+completion should match HF output
        assert (vllm_result.prompt_token_ids +
                vllm_result.outputs[0].token_ids == hf_output[0])

        # Validate sample logprobs
        if num_top_logprobs is not None:
            assert num_top_logprobs is not None
            # Confirm that the structure of the sample logprobs in the result is
            # correct
            assert vllm_result.outputs[0].logprobs is not None
            assert len(vllm_result.outputs[0].logprobs) == max_tokens
            for logprobs, token_id in zip(vllm_result.outputs[0].logprobs,
                                          vllm_result.outputs[0].token_ids):
                assert logprobs is not None
                # If the output token is not included in the top X
                # logprob, it can return 1 more data
                assert (len(logprobs) == num_top_logprobs
                        or len(logprobs) == num_top_logprobs + 1)
                # But confirm that the output token ultimately does appear
                # among the logprobs
                assert token_id in logprobs
            output_text = vllm_result.outputs[0].text
            output_string_from_most_likely_tokens_lst: List[str] = []
            for top_logprobs in vllm_result.outputs[0].logprobs:
                top_logprob = next(iter(top_logprobs.values()))
                output_string_from_most_likely_tokens_lst.append(
                    top_logprob.decoded_token)

            if detokenize:
                output_string_from_most_likely_tokens = "".join(
                    output_string_from_most_likely_tokens_lst)
                assert_incr_detok_str_matches_non_incr_detok_str(
                    output_text, output_string_from_most_likely_tokens,
                    "The output text from the top logprob for each token "
                    "position should be the same as the output text in the "
                    "result.")
            else:
                assert output_text == ''
                assert output_string_from_most_likely_tokens_lst == (
                    [None] * max_tokens)

            # Compare vLLM sample logprobs to HF
            vllm_sample_logprobs = vllm_result.outputs[0].logprobs
            for i, top_logprobs in enumerate(vllm_sample_logprobs):
                for token_id, sample_logprob in top_logprobs.items():
                    logprob = sample_logprob.logprob
                    torch.testing.assert_close(
                        logprob,
                        hf_logprob[i][-1][token_id].item(),
                        atol=1e-2,
                        rtol=1e-2)
                    if detokenize:
                        assert isinstance(sample_logprob.decoded_token, str), (
                            "The token should be decoded by the time it is"
                            " returned to the user.")

            # At this point we know the sample logprobs are correct for this
            # request. Validate that cumulative_logprob is actually the sum.
            # For each request, assert that the returned cumulative logprob
            # matches the correct value, which is computed below.
            torch.testing.assert_close(
                vllm_result.outputs[0].cumulative_logprob,
                compute_correct_cumulative_logprob(vllm_result.outputs[0]),
                atol=1e-6,
                rtol=1e-6)
        else:
            # Logprobs disabled for this request; should be None
            assert vllm_result.outputs[0].logprobs is None

        # Validate prompt logprobs
        if num_top_prompt_logprobs is not None:
            # Confirm that structure of prompt logprobs in result is correct
            assert vllm_result.prompt_logprobs is not None
            # - The first prompt logprob is always None
            assert vllm_result.prompt_logprobs[0] is None
            # - Prompt logprobs are returned for all indices in
            #   the prompt
            assert len(vllm_result.prompt_logprobs) == len(
                vllm_result.prompt_token_ids)
            for prompt_logprobs, prompt_token_id in zip(
                    vllm_result.prompt_logprobs[1:],
                    vllm_result.prompt_token_ids[1:]):
                assert prompt_logprobs is not None
                # - If the prompt token is not included in the top X
                #   logprob, it can return 1 more data
                assert (len(prompt_logprobs) == num_top_prompt_logprobs
                        or len(prompt_logprobs) == num_top_prompt_logprobs + 1)
                # But confirm that the prompt token ultimately does appear
                # among the prompt logprobs
                assert prompt_token_id in prompt_logprobs
            # Compare prompt logprobs to HF
            # The first prompt logprob is always None, so we compare it from
            # 1:.
            vllm_prompt_logprobs = vllm_result.prompt_logprobs[1:]
            for i, vllm_prompt_logprob_dict in enumerate(vllm_prompt_logprobs):
                for token_id, logprob in vllm_prompt_logprob_dict.items():
                    torch.testing.assert_close(
                        logprob.logprob,
                        hf_logprob[0][i][token_id].item(),
                        atol=2e-2,
                        rtol=2e-2)
        else:
            assert vllm_result.prompt_logprobs is None


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype",
                         ["half"])  # needed for comparing logprobs with HF
@pytest.mark.parametrize("max_num_batched_tokens", [128, 256, 1024])
@pytest.mark.parametrize("batch_logprobs_composition",
                         ["NONE", "SAMPLE", "PROMPT", "SAMPLE_PROMPT"])
def test_get_logprobs_and_prompt_logprobs(
    hf_runner,
    vllm_runner,
    model: str,
    dtype: str,
    batch_logprobs_composition: str,
    max_num_batched_tokens: int,
    example_prompts,
    monkeypatch,
) -> None:
    """Test V1 Engine logprobs & prompt logprobs
    
    Exercise a variety of combinations of `logprobs` and `prompt_logprobs`
    settings and validate that
    * The generated logprobs and prompt logprobs are consistent with the
      configuration settings, in terms of whether or not the logprobs
      (of either type) were requested and how many were requested
    * The generated logprobs are consistent with the generated tokens
    * The generated (prompt)logprobs are consistent with HuggingFace
      (prompt)logprobs, as a reference

    batch_logprobs_composition controls the logprobs configurations for
    requests in the batch under test.

    Args:
      hf_runner
      vllm_runner
      model
      dtype
      batch_logprobs_composition: logprobs configuration for test batch
      max_num_batched_tokens: token budget for scheduling
      example_prompts
      monkeypatch
    """
    _test_case_get_logprobs_and_prompt_logprobs(
        hf_runner=hf_runner,
        vllm_runner=vllm_runner,
        model=model,
        dtype=dtype,
        detokenize=True,
        batch_logprobs_composition=batch_logprobs_composition,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=False,
        example_prompts=example_prompts,
        monkeypatch=monkeypatch)


def test_max_logprobs(monkeypatch):
    """vLLM v1 engine should fail a request with `logprobs > max_logprobs`
    
    Should also fail for `prompt_logprobs > max_logprobs`
    
    Args:
      monkeypatch
    """
    override_backend_env_variable(monkeypatch, "FLASH_ATTN")

    runner = VllmRunner("facebook/opt-125m",
                        max_logprobs=1,
                        enable_prefix_caching=False)
    vllm_sampling_params = SamplingParams(logprobs=1)
    # should pass
    runner.generate(["Hello world"], sampling_params=vllm_sampling_params)

    bad_sampling_params = SamplingParams(logprobs=2)
    with pytest.raises(ValueError):
        runner.generate(["Hello world"], sampling_params=bad_sampling_params)


@pytest.mark.parametrize("model", MODELS)
def test_none_logprobs(vllm_runner, model, example_prompts, monkeypatch):
    """Engine should return `logprobs` and `prompt_logprobs` as `None`
    
    Args:
      vllm_runner: vLLM engine runner fixture
      model: model name
      example_prompts: list of example prompts (test fixture)
      monkeypatch: supports editing env vars and rolling back changes
                   after the test
    """
    override_backend_env_variable(monkeypatch, "FLASH_ATTN")

    max_num_seqs = 256
    max_num_batched_tokens = None
    max_tokens = 5

    with vllm_runner(
            model,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=False,
    ) as vllm_model:
        sampling_params_logprobs_none = SamplingParams(max_tokens=max_tokens,
                                                       logprobs=None,
                                                       prompt_logprobs=None,
                                                       temperature=0.0)
        results_logprobs_none = vllm_model.model.generate(
            example_prompts, sampling_params=sampling_params_logprobs_none)

    for i in range(len(results_logprobs_none)):
        # Check sample logprobs are None
        assert results_logprobs_none[i].outputs[0].logprobs is None
        assert results_logprobs_none[i].outputs[0].cumulative_logprob is None
        # Check prompt logprobs are None
        assert results_logprobs_none[i].prompt_logprobs is None


@pytest.mark.parametrize("model", MODELS)
def test_zero_logprobs(vllm_runner, model, example_prompts, monkeypatch):
    """Engine should return sampled token and prompt token logprobs
    
    Args:
      vllm_runner: vLLM engine runner fixture
      model: model name
      example_prompts: list of example prompts (test fixture)
      monkeypatch: supports editing env vars and rolling back changes
                   after the test
    """
    override_backend_env_variable(monkeypatch, "FLASH_ATTN")

    max_num_seqs = 256
    max_num_batched_tokens = None
    max_tokens = 5

    with vllm_runner(
            model,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
    ) as vllm_model:
        sampling_params_logprobs_zero = SamplingParams(max_tokens=max_tokens,
                                                       logprobs=0,
                                                       prompt_logprobs=0,
                                                       temperature=0.0)
        results_logprobs_zero = vllm_model.model.generate(
            example_prompts, sampling_params=sampling_params_logprobs_zero)

    for i in range(len(results_logprobs_zero)):
        # Check that there is one sample logprob dict for each
        # sample token
        logprobs = results_logprobs_zero[i].outputs[0].logprobs
        prompt_logprobs = results_logprobs_zero[i].prompt_logprobs
        sampled_token_ids = results_logprobs_zero[i].outputs[0].token_ids
        prompt_token_ids = results_logprobs_zero[i].prompt_token_ids
        assert logprobs is not None
        assert len(sampled_token_ids) == len(logprobs)
        assert results_logprobs_zero[i].outputs[
            0].cumulative_logprob is not None
        # Check that there is one prompt logprob dict for each
        # prompt token
        assert prompt_logprobs is not None
        assert len(prompt_token_ids) == len(prompt_logprobs)
