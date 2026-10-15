"""Shared helpers for the hand-written engine. No model code lives here."""
import glob, json, os, struct, time
import torch
from huggingface_hub import constants as hf_constants, snapshot_download

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def snapshot_dir() -> str:
    """Return the pinned model snapshot, downloading it when needed."""
    repo = "models--" + MODEL_ID.replace("/", "--")
    local = os.path.join(hf_constants.HF_HUB_CACHE, repo, "snapshots", REVISION)
    required = ("config.json", "model.safetensors", "tokenizer.json")
    if all(os.path.isfile(os.path.join(local, name)) for name in required):
        return local
    return snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        cache_dir=hf_constants.HF_HUB_CACHE,
    )


def load_config() -> dict:
    with open(os.path.join(snapshot_dir(), "config.json")) as f:
        return json.load(f)


def load_weights(device: str) -> dict[str, torch.Tensor]:
    """Load every tensor from model.safetensors onto `device`, bf16, keyed by name."""
    from safetensors.torch import load_file
    t0 = time.perf_counter()
    w = load_file(os.path.join(snapshot_dir(), "model.safetensors"), device=device)
    print(f"loaded {len(w)} tensors in {time.perf_counter()-t0:.2f}s")
    return w


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def sync(dev: str):
    if dev == "mps":
        torch.mps.synchronize()
