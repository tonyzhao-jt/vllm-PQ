# Problem Statement

Ensure the v1 LLM Engine exposes a superset of the metrics available in v0.

# Objectives

- Archive parity of metrics between v0 and v1.
- The priority use case is accessing these metrics via Prometheus as this is what we expect to be used in production environments.
- Logging support - i.e. printing metrics to the info log - is provided for more ad-hoc testing, debugging, development, and exploratory use cases.

# Background

Metrics in vLLM can be categorized as follows:

1. Server-level metrics: these are global metrics that track the state and performance of the LLM engine. These are typically exposed as Gauges or Counters in Prometheus.
2. Request-level metrics: these are metrics that track the characteristics - e.g. size and timing - of individual requests. These are typically exposed as Histrograms in Prometheus, and are often the SLO that an SRE monitoring vLLM will be tracking.

The mental model is that the "Server-level Metrics" explain why the "Request-level Metrics" are what they are.

## v0 Metrics

In v0, the following metrics are exposed via a Prometheus-compatible `/metrics` endpoint using the `vllm:` prefix:

- `vllm:num_requests_running` (Gauge)
- `vllm:num_requests_swapped` (Gauge)
- `vllm:num_requests_waiting` (Gauge)
- `vllm:gpu_cache_usage_perc` (Gauge)
- `vllm:cpu_cache_usage_perc` (Gauge)
- `vllm:gpu_prefix_cache_hit_rate` (Gauge)
- `vllm:cpu_prefix_cache_hit_rate` (Gauge)
- `vllm:prompt_tokens_total` (Counter)
- `vllm:generation_tokens_total` (Counter)
- `vllm:request_success_total` (Counter)
- `vllm:request_prompt_tokens` (Histogram)
- `vllm:request_generation_tokens` (Histogram)
- `vllm:time_to_first_token_seconds` (Histogram)
- `vllm:time_per_output_token_seconds` (Histogram)
- `vllm:e2e_request_latency_seconds` (Histogram)
- `vllm:request_queue_time_seconds` (Histogram)
- `vllm:request_inference_time_seconds` (Histogram)
- `vllm:request_prefill_time_seconds` (Histogram)
- `vllm:request_decode_time_seconds` (Histogram)
- `vllm:request_max_num_generation_tokens` (Histogram)
- `vllm:num_preemptions_total` (Counter)
- `vllm:cache_config_info` (Gauge)
- `vllm:lora_requests_info` (Gauge)
- `vllm:tokens_total` (Counter)
- `vllm:iteration_tokens_total` (Histogram)
- `vllm:time_in_queue_requests` (Historgram)
- `vllm:model_forward_time_milliseconds` (Histogram
- `vllm:model_execute_time_milliseconds` (Histogram)
- `vllm:request_params_n` (Histogram)
- `vllm:request_params_max_tokens` (Histogram)
- `vllm:spec_decode_draft_acceptance_rate` (Gauge)
- `vllm:spec_decode_efficiency` (Gauge)
- `vllm:spec_decode_num_accepted_tokens_total` (Counter)
- `vllm:spec_decode_num_draft_tokens_total` (Counter)
- `vllm:spec_decode_num_emitted_tokens_total` (Counter)

These are documented under [Inferencing and Serving -> Production Metrics](https://docs.vllm.ai/en/stable/serving/metrics.html).

## Grafana Dashboard

vLLM also provides [a reference example](https://docs.vllm.ai/en/latest/getting_started/examples/prometheus_grafana.html) for how to collect and store these metrics using Prometheus and visualize them using a Grafana dashboard.

The subset of metrics exposed in the Grafana dashboard gives us an indication of which metrics are especially important:

- `vllm:e2e_request_latency_seconds_bucket` - End to end request latency measured in seconds
- `vllm:prompt_tokens_total` - Prompt Tokens/Sec
- `vllm:generation_tokens_total` - Generation Tokens/Sec
- `vllm:time_per_output_token_seconds` - Inter token latency (Time Per Output Token, TPOT) in second.
- `vllm:time_to_first_token_seconds` - Time to First Token (TTFT) latency in seconds.
- `vllm:num_requests_running` (also, `_swapped` and `_waiting`) - Number of requests in RUNNING, WAITING, and SWAPPED state
- `vllm:gpu_cache_usage_perc` - Percentage of used cache blocks by vLLM.
- `vllm:request_prompt_tokens` - Request prompt length
- `vllm:request_generation_tokens` - request generation length
- `vllm:request_success_total` - Number of finished requests by their finish reason: either an EOS token was generated or the max sequence length was reached
- `vllm:request_queue_time_seconds` - Queue Time
- `vllm:request_prefill_time_seconds` - Requests Prefill Time
- `vllm:request_decode_time_seconds` - Requests Decode Time
vllm:request_max_num_generation_tokens - Max Generation Token in Sequence Group

See [the PR which added this Dashboard](https://github.com/vllm-project/vllm/pull/2316) for interesting and useful background on the choices made here.

## Prometheus Client Library

Prometheus support was initially added [using the aioprometheus library](https://github.com/vllm-project/vllm/pull/1890), but a switch was made quickly to [prometheus_client](https://github.com/vllm-project/vllm/pull/2730). The rationale is discussed in both linked PRs.

### Multi-process Mode ==

In v0, metrics are collected in the engine core process and we use multi-process mode to make them available in the API server process. See [#7279](https://github.com/vllm-project/vllm/pull/7279).

### Built in Python/Process Metrics

The following metrics are supported by default by `prometheus_client`, but the are not exposed with multiprocess mode is used:

- `python_gc_objects_collected_total`
- `python_gc_objects_uncollectable_total`
- `python_gc_collections_total`
- `python_info`
- `process_virtual_memory_bytes`
- `process_resident_memory_bytes`
- `process_start_time_seconds`
- `process_cpu_seconds_total`
- `process_open_fds`
- `process_max_fds`

(Relevant because if we move away from multiprocess mode in v1, we get these back)

### v0 PRs and Issues

For background, these are some of the relevant PRs which added the v0 metrics:

- #1890
- #2316
- #2730
- #4464
- #7279

Also note the ["Even Better Observability"](https://github.com/vllm-project/vllm/issues/3616) feature where e.g. [a detailed roadmap was laid out](https://github.com/vllm-project/vllm/issues/3616#issuecomment-2030858781).

