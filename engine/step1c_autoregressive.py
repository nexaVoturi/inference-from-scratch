"""Step 1c: a complete, single-sequence autoregressive Qwen3 loop.

This keeps the layer math explicit, but adds the state that decode needs:
each layer stores its K/V tensors and appends one new position per step.

Run from the repository root:
    python3 engine/step1c_autoregressive.py --new-tokens 32

The reference IDs come from step 1a's Transformers generation. Matching token
IDs is the checkpoint; timing is reported for orientation, not as a benchmark.
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

from common import device, load_config, load_weights, snapshot_dir, sync


def rmsnorm(x, weight, eps):
    """Qwen3 RMSNorm: normalize in FP32, cast back, then scale."""
    dtype = x.dtype
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_cache(positions, dim, theta, dev, dtype):
    """Return cos/sin rows for the supplied absolute positions."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=dev).float() / dim))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    return x * cos.unsqueeze(0) + rotate_half(x) * sin.unsqueeze(0)


def attention_tiled(q, k, v, positions, causal, block_size):
    """Attention with an online softmax over contiguous K/V blocks.

    Unlike the full path, this never materializes the complete score or
    probability matrix. It is still an educational PyTorch implementation;
    the Python loop is deliberately visible so the memory trade-off is clear.
    """
    qf = q.float()
    kf = k.float()
    vf = v.float()
    q_len = q.shape[1]
    kv_len = k.shape[1]
    scale = q.shape[-1] ** -0.5

    running_max = torch.full(
        (q.shape[0], q_len, 1), float("-inf"), device=q.device
    )
    running_sum = torch.zeros_like(running_max)
    running_out = torch.zeros(
        (q.shape[0], q_len, v.shape[-1]), device=q.device, dtype=torch.float32
    )
    key_positions = torch.arange(kv_len, device=q.device)

    for start in range(0, kv_len, block_size):
        stop = min(start + block_size, kv_len)
        scores = (qf @ kf[:, start:stop].transpose(-1, -2)) * scale
        if causal:
            allowed = key_positions[start:stop].view(1, 1, -1) <= positions.view(1, -1, 1)
            scores = scores.masked_fill(~allowed, float("-inf"))

        block_max = scores.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(running_max, block_max)
        old_scale = torch.exp(running_max - new_max)
        old_scale = torch.where(torch.isfinite(running_max), old_scale, 0.0)
        block_exp = torch.exp(scores - new_max)
        running_sum = running_sum * old_scale + block_exp.sum(dim=-1, keepdim=True)
        running_out = (
            running_out * old_scale
            + block_exp @ vf[:, start:stop]
        )
        running_max = new_max

    return (running_out / running_sum).to(q.dtype)


class Qwen3Loop:
    """Smallest useful model wrapper: one sequence, no batching, no sampling."""

    def __init__(self, dev, attention="full", tile_size=128):
        self.dev = dev
        if attention not in {"full", "tiled"}:
            raise ValueError(f"unknown attention path: {attention}")
        self.attention = attention
        self.tile_size = tile_size
        self.cfg = load_config()
        self.w = load_weights(dev)
        self.hidden_size = self.cfg["hidden_size"]
        self.nh = self.cfg["num_attention_heads"]
        self.nkv = self.cfg["num_key_value_heads"]
        self.hd = self.cfg["head_dim"]
        self.eps = self.cfg["rms_norm_eps"]
        self.theta = 1_000_000.0
        self.layers = self.cfg["num_hidden_layers"]

    def layer(self, hidden, layer_idx, positions, cache):
        """Run one layer and return its output plus that layer's updated KV cache."""
        prefix = f"model.layers.{layer_idx}."
        residual = hidden
        normed = rmsnorm(hidden, self.w[prefix + "input_layernorm.weight"], self.eps)

        q = normed @ self.w[prefix + "self_attn.q_proj.weight"].T
        k = normed @ self.w[prefix + "self_attn.k_proj.weight"].T
        v = normed @ self.w[prefix + "self_attn.v_proj.weight"].T

        q = q.view(hidden.shape[0], self.nh, self.hd).transpose(0, 1)
        k = k.view(hidden.shape[0], self.nkv, self.hd).transpose(0, 1)
        v = v.view(hidden.shape[0], self.nkv, self.hd).transpose(0, 1)

        q = rmsnorm(q, self.w[prefix + "self_attn.q_norm.weight"], self.eps)
        k = rmsnorm(k, self.w[prefix + "self_attn.k_norm.weight"], self.eps)

        cos, sin = rope_cache(positions, self.hd, self.theta, self.dev, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            old_k, old_v = cache
            k = torch.cat((old_k, k), dim=1)
            v = torch.cat((old_v, v), dim=1)
        new_cache = (k, v)

        # GQA: materialize the two-query-head share used by this educational loop.
        k_heads = k.repeat_interleave(self.nh // self.nkv, dim=0)
        v_heads = v.repeat_interleave(self.nh // self.nkv, dim=0)
        if self.attention == "tiled":
            attn = attention_tiled(
                q,
                k_heads,
                v_heads,
                positions,
                causal=hidden.shape[0] > 1,
                block_size=self.tile_size,
            )
        else:
            scores = (q.float() @ k_heads.float().transpose(-1, -2)) / (self.hd**0.5)

            # Prefill has no cached prefix here, so the ordinary upper-triangle
            # mask is sufficient. A single decode query can attend to the complete cache.
            if hidden.shape[0] > 1:
                mask = torch.full(
                    (hidden.shape[0], k.shape[1]),
                    float("-inf"),
                    device=self.dev,
                ).triu(1)
                scores = scores + mask
            probs = scores.softmax(-1).to(q.dtype)
            attn = probs @ v_heads
        attn = attn.transpose(0, 1).reshape(hidden.shape[0], self.nh * self.hd)
        attn_out = attn @ self.w[prefix + "self_attn.o_proj.weight"].T

        hidden = residual + attn_out
        residual2 = hidden
        normed2 = rmsnorm(
            hidden, self.w[prefix + "post_attention_layernorm.weight"], self.eps
        )
        gate = normed2 @ self.w[prefix + "mlp.gate_proj.weight"].T
        up = normed2 @ self.w[prefix + "mlp.up_proj.weight"].T
        mlp_out = (
            F.silu(gate) * up
        ) @ self.w[prefix + "mlp.down_proj.weight"].T
        return residual2 + mlp_out, new_cache

    @torch.no_grad()
    def forward(self, token_ids, positions, cache=None):
        """Return logits for the final supplied position and updated layer caches."""
        hidden = F.embedding(token_ids, self.w["model.embed_tokens.weight"])
        next_cache = []
        for layer_idx in range(self.layers):
            old_cache = None if cache is None else cache[layer_idx]
            hidden, layer_cache = self.layer(
                hidden, layer_idx, positions, old_cache
            )
            next_cache.append(layer_cache)

        hidden = rmsnorm(hidden, self.w["model.norm.weight"], self.eps)
        logits = hidden @ self.w["lm_head.weight"].T
        return logits[-1], next_cache

    @torch.no_grad()
    def generate(self, prompt_ids, new_tokens):
        """Greedy-generate new tokens, retaining a per-layer KV cache."""
        prompt_len = prompt_ids.shape[0]
        positions = torch.arange(prompt_len, device=self.dev)
        logits, cache = self.forward(prompt_ids, positions)
        generated = []

        for step in range(new_tokens):
            token = logits.argmax().reshape(1)
            generated.append(int(token.item()))
            if step == new_tokens - 1:
                break
            position = torch.tensor([prompt_len + step], device=self.dev)
            logits, cache = self.forward(token, position, cache)

        return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-tokens", type=int, default=32)
    args = parser.parse_args()

    dev = device()
    reference = json.load(open("engine/out/reference.json"))
    prompt_ids = torch.tensor(reference["prompt_ids"], device=dev)
    expected = reference["ref_ids"][: args.new_tokens]

    loop = Qwen3Loop(dev)
    sync(dev)
    t0 = time.perf_counter()
    actual = loop.generate(prompt_ids, args.new_tokens)
    sync(dev)
    elapsed = time.perf_counter() - t0

    matched = actual == expected
    matching_tokens = sum(a == b for a, b in zip(actual, expected))
    print(f"device {dev}")
    print(
        f"manual loop: {len(actual)} new tokens in {elapsed:.2f}s "
        f"({elapsed / len(actual) * 1000:.1f} ms/token incl. prefill)"
    )
    print(f"token match: {matching_tokens}/{len(expected)}  exact={matched}")
    if not matched:
        first = next(i for i, (a, b) in enumerate(zip(actual, expected)) if a != b)
        print(f"first mismatch at generated index {first}: mine={actual[first]} ref={expected[first]}")
        print("mine", actual)
        print("ref ", expected)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
