# Build Your Own Qwen3 Inference Loop

A hands-on guide to understanding token generation, validating every step against
Hugging Face, and seeing how KV caching affects memory and latency.

This repository contains a small, explicit Qwen3-0.6B inference implementation. It
starts with a trusted Transformers reference, reproduces one transformer layer, adds
an autoregressive loop with a per-layer KV cache, and compares full and tiled
attention paths.

## Start here

- [Compendium: Build Your Own Qwen3 Inference Loop](https://nexavoturi.github.io/inference-from-scratch/docs/explainers/step1c-inference-loop-compendium.html)
- [Inside a transformer layer](https://nexavoturi.github.io/inference-from-scratch/docs/explainers/step1b-layer0-concepts.html)
- [Engine reproduction notes](engine/README.md)
- [Methodology and claim boundaries](METHODOLOGY.md)

## Reproduce the local result

From the repository root:

```text
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python engine/step1a_load.py
.venv/bin/python engine/step1b_layer0.py
.venv/bin/python engine/step1c_autoregressive.py --new-tokens 32
.venv/bin/python engine/step1d_kv_memory.py
.venv/bin/python engine/step1e_profile.py --repeats 3
```

The first model-loading command downloads the pinned Qwen3-0.6B snapshot when it is
not already present in the Hugging Face cache. Generated references and measurements
are saved under `engine/out/`.

## What the current run proves

- The manual 28-layer loop reproduces the Transformers reference sequence: **32/32 token IDs match**.
- Layer 0 hidden states are numerically aligned with Transformers:
  maximum absolute difference `0.031250`; mean absolute difference `0.000768`.
- The logical KV-cache cost is `112 KiB` per sequence token for this configuration.
- Full and tiled attention both preserve the observed token sequence.
- On the measured local CPU run, full attention took `22.6 ms/decode token` and tiled attention took `24.7 ms`.

## Claim boundaries

These are local educational measurements, not production benchmarks. They do not
establish CUDA performance, batching behavior, production throughput, or a universal
speedup from tiled attention. See `METHODOLOGY.md` for the disclosure rules.

## Repository structure

`engine/` contains the implementation and generated evidence. `docs/` contains the
explainers. `bench/` contains the first prefix-cache measurement and its caveats.
