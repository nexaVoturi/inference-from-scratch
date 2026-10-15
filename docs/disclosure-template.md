# Disclosure block

Copy this into every benchmark result, filled in. A result missing any field is not
publishable.

```yaml
run:
  date:                 # YYYY-MM-DD
  runs:                 # how many, and median or mean
  cost_usd:             # what this measurement cost to produce

engine:
  name:                 # vllm | sglang | vllm-metal | llama.cpp
  version:              # release tag
  commit:               # exact hash, or build date for vllm-metal
  optimization_level:   # -O0 | -O1 | -O2 | -O3 — mandatory, the number is meaningless without it
  flags:                # full serve command

model:
  id:                   # e.g. Qwen/Qwen3.5-2B
  revision:             # HF commit hash, not just the tag
  quantization:         # none | fp8-w8a8 | int4-w4a16 | mxfp4 | nvfp4
  quant_tool:           # llm-compressor version, or "as published"
  quant_source:         # which checkpoint it was produced from
  kv_cache_dtype:       # auto | fp8 | fp8_e4m3 | int8

hardware:
  gpu:                  # e.g. RTX 4090, or "Apple M4 Pro (Metal)"
  count:
  provider:             # runpod-community | vast | modal | prime-intellect | local
  region:
  hourly_rate_usd:      # and the date observed

workload:
  dataset:              # random | sharegpt | prefix_repetition | mooncake | burstgpt
  input_tokens:         # distribution, not just a mean
  output_tokens:
  concurrency_sweep:    # e.g. [1, 2, 4, 8, 16, 32, 64, 128] — never a single point

load_generator:
  tool:                 # guidellm | vllm bench serve | custom
  gil_bound_check:      # verified not client-bound? how?

transfers:              # from METHODOLOGY.md — which claims generalise beyond this hardware
does_not_transfer:      # and which explicitly do not
```
