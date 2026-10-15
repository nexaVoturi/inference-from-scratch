# Build Your Own Qwen3 Inference Loop

Autoregressive loop, contiguous KV cache, hand-written SDPA, tiled attention, arithmetic-intensity measurements. Weeks 1-3.

**Done when:** You can point at a profile and say why a decode step is memory-bound.

## Reproduce the local run

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

The first command that loads the model downloads the pinned Qwen3-0.6B snapshot
automatically through Hugging Face Hub. The scripts save the reference IDs and
summarised measurements under `engine/out/`.

## Checkpoints

- `step1a_load.py` loads the Qwen3-0.6B checkpoint, verifies the parameter-count derivation, and saves a Transformers reference generation.
- `step1b_layer0.py` implements one complete prefill layer and compares its hidden state with Transformers.
- `step1c_autoregressive.py` runs all 28 layers with an explicit contiguous per-layer K/V cache and greedy single-sequence decode. It checks generated token IDs against `engine/out/reference.json`:

  ```text
  .venv/bin/python engine/step1c_autoregressive.py --new-tokens 32
  ```

- `step1d_kv_memory.py` derives the logical contiguous K/V-cache footprint from
  the Qwen3 configuration and writes `engine/out/kv_cache_memory.csv`:

  ```text
  .venv/bin/python engine/step1d_kv_memory.py
  ```

- `step1e_profile.py` separates prompt prefill from one-token decode and
  compares full-score attention with the online-softmax tiled path:

  ```text
  .venv/bin/python engine/step1e_profile.py --repeats 3
  ```

The step 1c timing is a correctness measurement, not yet a benchmark. The current
checkpoint runs on CPU in this environment; CUDA comparison and hardware profiling
are still ahead. The hand-written full-score and tiled attention paths are included
in step 1e.

The step 1d memory numbers cover K/V tensors only. They exclude weights,
temporary activations, allocator overhead, and framework metadata.

The step 1e timings are local CPU/MPS orientation measurements. A tiled path
must match the reference tokens before its timing is meaningful.

Current CPU run (`3` repeats, `1` warmup, `32` generated tokens): full-score
attention took a median `22.6 ms/decode token`; tiled attention took `24.7 ms`.
Both produced `32/32` reference tokens. The tiled path is therefore retained
as a memory-movement experiment, not claimed as a speed improvement.

Results are not complete without a filled disclosure block. See `../docs/disclosure-template.md`.
