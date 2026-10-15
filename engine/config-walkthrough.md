# Qwen3-0.6B config walk-through

Date: 17 September 2026. Model revision `c1899de289a04d12100db370d81485cdf75e47ca`,
file `model.safetensors`, 1,503,300,328 bytes, all tensors BF16.

Derivations made before writing any engine code. Each ends in a number the engine
must reproduce. If the measurement disagrees, the derivation or the code is wrong,
and either is worth knowing.

## The config

| Field | Value | What it decides |
|---|---|---|
| `num_hidden_layers` | 28 | Loop count per forward pass |
| `hidden_size` | 1,024 | Residual stream width |
| `intermediate_size` | 3,072 | MLP inner width, 3x hidden |
| `num_attention_heads` | 16 | Query heads |
| `num_key_value_heads` | 8 | K and V heads. Grouped-query attention, 2 query heads per KV head |
| `head_dim` | 128 | Per-head width. Note 16 x 128 = 2,048, not 1,024: attention runs in a wider space than the residual stream and `o_proj` brings it back |
| `vocab_size` | 151,936 | Rows in the embedding and in `lm_head` |
| `max_position_embeddings` | 40,960 | Context limit the RoPE tables must cover |
| `rope_theta` | 1,000,000 | RoPE base. Lives under `rope_parameters` in transformers 5, not at the top level |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `tie_word_embeddings` | true | `lm_head` is the embedding matrix reused. The file still stores both copies |
| `attention_bias` | false | No bias vectors on q, k, v, o |
| `hidden_act` | silu | Gated MLP: `down(silu(gate(x)) * up(x))` |
| `dtype` | bfloat16 | 2 bytes per parameter |
| sliding window | off | Full causal attention at every layer |

Qwen3-specific: `q_norm` and `k_norm`, each `[128]`, are RMSNorms applied per head to
the query and key vectors **after projection and before RoPE**. Miss them and the
model produces text that is wrong in a way that is hard to notice at first.

## One layer's tensors

| Tensor | Shape | Parameters |
|---|---|---|
| `q_proj` | [2048, 1024] | 2,097,152 |
| `k_proj` | [1024, 1024] | 1,048,576 |
| `v_proj` | [1024, 1024] | 1,048,576 |
| `o_proj` | [1024, 2048] | 2,097,152 |
| `gate_proj` | [3072, 1024] | 3,145,728 |
| `up_proj` | [3072, 1024] | 3,145,728 |
| `down_proj` | [1024, 3072] | 3,145,728 |
| four norms | 1024 + 1024 + 128 + 128 | 2,304 |
| **layer total** | | **15,730,944** |

Attention is 6.29M of that, the MLP 9.44M. The MLP is 60% of the layer's weights,
which is also 60% of the bytes each decode step has to read.

## Parameter count

| Part | Parameters |
|---|---|
| 28 layers x 15,730,944 | 440,466,432 |
| `embed_tokens` [151936, 1024] | 155,582,464 |
| final norm | 1,024 |
| **unique total** | **596,049,920** |
| `lm_head` copy stored in the file | +155,582,464 |
| **on-disk total** | **751,632,384** |

The model is 0.6B by unique parameters and 0.75B by what the file stores. The
embedding is 26% of the unique count. A loader that honours the tie keeps one copy.

## Prediction 1: bytes read per decode step

A decode step reads every layer weight once and the full `lm_head` once. The embedding
lookup reads one row and is negligible.

```
(440,466,432 + 155,582,464) params x 2 bytes = 1,192,097,792 bytes = 1.19 GB
```

Apple's spec sheet gives the M4 Pro 273 GB/s of memory bandwidth. Treat that as a
ceiling to be verified, not a measurement.

```
1.19 GB / 273 GB/s = 4.37 ms per step  ->  about 229 tokens/s at batch 1
```

That is the bandwidth floor for one token, ignoring KV-cache reads, activations and
launch overhead. The hand-written engine should approach it from above and never
beat it. If it appears to, the timing is wrong.

## Prediction 2: KV cache per token

```
2 (K and V) x 28 layers x 8 KV heads x 128 head_dim x 2 bytes = 114,688 bytes = 112 KiB
```

| Context | KV cache |
|---|---|
| 1,024 tokens | 117 MB |
| 4,096 tokens | 470 MB |
| 40,960 tokens (max) | 4.70 GB |

Without grouped-query attention, at 16 KV heads, every figure doubles.

Cross-check against the running server. vLLM reported 8,479 blocks of 16 tokens, so
135,664 cacheable tokens, on 16 September. At 112 KiB per token that is 15.56 GB of
KV cache, which is plausible for a 24 GB machine after weights and activations at the
default memory fraction. The engine's measured memory-versus-context line must have
this slope.

## Prediction 3: FLOPs per decode token, and arithmetic intensity

Matmul FLOPs are 2 per parameter per token:

```
2 x 596,048,896 = 1.19 GFLOP per token, from weights
```

Attention over context adds, per layer, `QK^T` and `PV`, each `2 x 16 heads x 128 x seq`:

```
28 layers x 8,192 x seq  =  229k x seq FLOP    ->  0.94 GFLOP at seq 4,096
```

Arithmetic intensity at batch 1, short context: about 1.19 GFLOP over 1.19 GB, so
**roughly 1 FLOP per byte.** The M4 Pro GPU peaks somewhere near 9 TFLOPS in FP16
(Apple does not publish this; treat it as an estimate to bound, not to quote). Its
balance point is then about 9e12 / 273e9, or **roughly 33 FLOP per byte.** Decode sits
more than thirty times below the ridge. That is the whole reason decode is
memory-bound, and it is what the profile in step 4 has to show.

Prefill over 512 tokens reads the same weights once and does 512 times the matmul
work, so its intensity is near 512 FLOP per byte, well above the ridge. Same model,
same machine, opposite regime.

## What the engine must get right, in the order bugs usually appear

1. Chat template and tokenizer, so the prompt tokens match what vLLM sees.
2. RMSNorm in the pre-norm position: norm, then attention, add; norm, then MLP, add.
3. `q_norm` and `k_norm` per head, before RoPE.
4. RoPE with theta 1e6, applied to q and k, rotating pairs of dimensions.
5. GQA: repeat each KV head across its 2 query heads, or index instead of repeating.
6. Causal mask in prefill. No mask needed for a single-token decode step.
7. Gated MLP with SiLU on the gate branch only.
8. Tied `lm_head`.

## Numbers to reproduce

| Quantity | Predicted | Measured | Where |
|---|---|---|---|
| Unique parameters | 596,049,920 | | engine load |
| KV bytes per token | 114,688 | | step 2 memory plot |
| Decode step floor, batch 1 | 4.37 ms | | step 1 and 2 timing |
| Decode arithmetic intensity | ~1 FLOP/byte | | step 4 roofline |
| Prefill-512 arithmetic intensity | ~500 FLOP/byte | | step 4 roofline |
