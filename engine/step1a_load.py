"""Step 1a. Load weights and tokenizer, check the parameter prediction,
and produce a reference generation with transformers to compare against later.

Predictions checked (engine/config-walkthrough.md):
  unique parameters = 596,049,920
"""
import json, os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import MODEL_ID, snapshot_dir, load_config, load_weights, device, sync

dev = device()
print("device", dev)
cfg = load_config()
w = load_weights(dev)

# --- parameter count -------------------------------------------------------
on_disk = sum(t.numel() for t in w.values())
unique = on_disk - (w["lm_head.weight"].numel() if cfg["tie_word_embeddings"] else 0)
print(f"params on disk {on_disk:,}   unique {unique:,}   predicted 596,049,920")
assert unique == 596_049_920, "parameter prediction failed"

# --- tokenizer and chat template ------------------------------------------
tok = AutoTokenizer.from_pretrained(snapshot_dir())
msgs = [{"role": "user", "content": "Name the three largest cities in Australia."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=False)
print("prompt tokens", ids.shape[1])
print(repr(tok.decode(ids[0])))

# --- reference generation with transformers, greedy -----------------------
model = AutoModelForCausalLM.from_pretrained(snapshot_dir(), dtype=torch.bfloat16).to(dev).eval()
ids = ids.to(dev)
N = 32
with torch.no_grad():
    sync(dev); t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=N, do_sample=False)
    sync(dev); dt = time.perf_counter() - t0
gen = out[0, ids.shape[1]:].tolist()
print(f"transformers greedy: {N} tokens in {dt:.2f}s  ({dt/N*1000:.1f} ms/token incl. prefill)")
print(repr(tok.decode(gen)))

os.makedirs("engine/out", exist_ok=True)
json.dump({"prompt_ids": ids[0].tolist(), "ref_ids": gen}, open("engine/out/reference.json", "w"))
print("saved engine/out/reference.json")
