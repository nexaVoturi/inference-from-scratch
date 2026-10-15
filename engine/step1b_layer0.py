"""Step 1b, checkpoint 1. Hand-written layer 0 only, diffed against
transformers' own post-layer-0 hidden state. Run from the repo root:
    python3 engine/step1b_layer0.py

Checklist from config-walkthrough.md, in the order bugs usually appear:
  RMSNorm (pre-norm) -> attention -> add; RMSNorm -> MLP -> add.
  q_norm/k_norm per head, before RoPE.
  RoPE theta 1e6, rotate-half convention: dim i pairs with dim i+64, not i+1.
  GQA: each of 8 KV heads serves 2 query heads (repeat_interleave order).
  Causal mask (this is prefill, seq=20 tokens).
  Gated MLP, SiLU on the gate branch only.
"""
import json
import torch
import torch.nn.functional as F
from common import load_config, load_weights, device, snapshot_dir

dev = device()
cfg = load_config()
w = load_weights(dev)

H = cfg["hidden_size"]
NH = cfg["num_attention_heads"]
NKV = cfg["num_key_value_heads"]
HD = cfg["head_dim"]
EPS = cfg["rms_norm_eps"]
THETA = 1_000_000.0


def rmsnorm(x, weight, eps=EPS):
    # fp32 normalize, cast down, THEN multiply by weight -- matches HF's
    # Qwen3RMSNorm exactly, not just "close enough."
    dtype = x.dtype
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cache(seq_len, dim, theta, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


prompt_ids = torch.tensor(
    json.load(open("engine/out/reference.json"))["prompt_ids"], device=dev
)
seq = prompt_ids.shape[0]

# --- embedding ---
hidden = F.embedding(prompt_ids, w["model.embed_tokens.weight"])

# --- layer 0 ---
p = "model.layers.0."
residual = hidden
normed = rmsnorm(hidden, w[p + "input_layernorm.weight"])

q = normed @ w[p + "self_attn.q_proj.weight"].T
k = normed @ w[p + "self_attn.k_proj.weight"].T
v = normed @ w[p + "self_attn.v_proj.weight"].T

q = q.view(seq, NH, HD).transpose(0, 1)
k = k.view(seq, NKV, HD).transpose(0, 1)
v = v.view(seq, NKV, HD).transpose(0, 1)

q = rmsnorm(q, w[p + "self_attn.q_norm.weight"])
k = rmsnorm(k, w[p + "self_attn.k_norm.weight"])

cos, sin = rope_cache(seq, HD, THETA, dev, q.dtype)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)

k = k.repeat_interleave(NH // NKV, dim=0)
v = v.repeat_interleave(NH // NKV, dim=0)

scores = (q.float() @ k.float().transpose(-1, -2)) / (HD**0.5)
mask = torch.full((seq, seq), float("-inf"), device=dev).triu(1)
probs = (scores + mask).softmax(-1).to(q.dtype)
attn = probs @ v
attn = attn.transpose(0, 1).reshape(seq, NH * HD)
attn_out = attn @ w[p + "self_attn.o_proj.weight"].T

hidden = residual + attn_out
residual2 = hidden
normed2 = rmsnorm(hidden, w[p + "post_attention_layernorm.weight"])
gate = normed2 @ w[p + "mlp.gate_proj.weight"].T
up = normed2 @ w[p + "mlp.up_proj.weight"].T
mlp_out = (F.silu(gate) * up) @ w[p + "mlp.down_proj.weight"].T
hidden = residual2 + mlp_out

# --- checkpoint: compare against transformers' own layer-0 output ---
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(snapshot_dir(), dtype=torch.bfloat16).to(dev).eval()
with torch.no_grad():
    out = model(prompt_ids.unsqueeze(0), output_hidden_states=True)
ref = out.hidden_states[1][0]

diff = (hidden.float() - ref.float()).abs()
print("mine  ", hidden[:3, :5])
print("theirs", ref[:3, :5])
print(f"max abs diff {diff.max().item():.6f}   mean abs diff {diff.mean().item():.6f}")
