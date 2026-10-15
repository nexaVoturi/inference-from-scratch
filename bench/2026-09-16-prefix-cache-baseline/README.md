# Prefix cache baseline — vllm-metal on M4 Pro

First measurement taken in this repository. Purpose was to validate the local stack,
not to produce a publishable result. Recorded because the method matters from day one.

## Disclosure block

```yaml
run:
  date: 2026-09-16
  runs: 1                     # single run — NOT publishable, see caveats
  cost_usd: 0.00              # local hardware

engine:
  name: vllm-metal
  version: 0.29.0.dev20260916013337
  vllm_core: 0.29.0 (V1 engine)
  commit: build dated 2026-09-16T01:33:37
  optimization_level: default (VLLM_COMPILE / inductor, cudagraph_mode=NONE)
  flags: vllm serve Qwen/Qwen3-0.6B --port 8000 --max-model-len 4096

model:
  id: Qwen/Qwen3-0.6B
  revision: c1899de289a04d12100db370d81485cdf75e47ca
  quantization: none (bfloat16)
  kv_cache_dtype: auto

hardware:
  gpu: Apple M4 Pro, 24GB unified memory (Metal backend)
  count: 1
  provider: local
  hourly_rate_usd: n/a

engine_config:
  enable_prefix_caching: true
  enable_chunked_prefill: true
  block_size: 16
  num_gpu_blocks: 8479
  kv_cache_size_tokens: 135664
  max_concurrency_at_4096: 33.12x

workload:
  shared_prefix_tokens: ~480
  total_prompt_tokens: 503
  output_tokens: 24
  concurrency_sweep: NONE — sequential single requests
```

## Result

| Run | Prompt tokens | Total latency (s) | Cumulative hits/queries | Hit rate |
|---|---|---|---|---|
| Cold, first sight | 503 | 4.69 | 0 / 503 | 0.0% |
| Same prefix, new suffix | 503 | 2.43 | 480 / 1006 | 47.7% |
| Same prefix, new suffix | 503 | 0.52 | 960 / 1509 | 63.6% |
| Different prefix | 23 | 0.42 | 960 / 1532 | 62.7% |

Prefix cache hit accounting is clean: 480 of 503 prompt tokens were served from cache
on repeat, and the unrelated prompt added queries without adding hits, which is the
expected behaviour.

## Caveats — why this is not publishable as-is

1. **The latency drop is confounded.** The first request includes `torch.compile` /
   inductor warmup. Attributing the 4.69s → 0.52s improvement to prefix caching alone
   would be wrong. Separating compilation from cache effects requires a warmup request
   before measurement begins.
2. **Total request latency, not TTFT.** These numbers include generation of 24 output
   tokens. TTFT was not isolated.
3. **n=1 per condition.** No repeats, no variance, no median.
4. **No concurrency sweep.** Sequential single requests only, which is a single
   operating point — exactly what `METHODOLOGY.md` says not to publish.

## What transfers

The **hit-rate accounting** transfers: 480/503 tokens served from cache for a shared
prefix is a property of the block manager, not of Apple Silicon.

The **latency figures do not transfer** to CUDA hardware and must not be quoted as
absolute performance.

## To redo properly

Warm up first, isolate TTFT, repeat n≥5, take medians, and sweep concurrency. That is
the week 4-5 exercise, not this one.
