"""Step 1d: calculate contiguous KV-cache memory from the model config.

This is the memory arithmetic behind the cache used by step 1c. It reports
logical KV bytes only: model weights, temporary activations, allocator
overhead, and framework metadata are intentionally excluded.

Run from the repository root:
    python3 engine/step1d_kv_memory.py
"""
import argparse
import csv
from pathlib import Path

from common import load_config


def cache_bytes(config, sequence_length, batch_size, dtype_bytes=2):
    """Return logical bytes for K and V at one batch/sequence shape."""
    return (
        batch_size
        * config["num_hidden_layers"]
        * 2  # K and V
        * config["num_key_value_heads"]
        * config["head_dim"]
        * sequence_length
        * dtype_bytes
    )


def format_bytes(value):
    units = [(1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")]
    for scale, unit in units:
        if value >= scale:
            return f"{value / scale:.3f} {unit}"
    return f"{value} B"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq", type=int, default=None)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    config = load_config()
    max_seq = args.max_seq or config["max_position_embeddings"]
    lengths = [1, 128, 512, 2048, 4096, 8192, 16384, max_seq]
    lengths = sorted({n for n in lengths if n <= max_seq})

    per_token = cache_bytes(config, 1, args.batch_size)
    print(
        "KV cache: "
        f"layers={config['num_hidden_layers']} "
        f"kv_heads={config['num_key_value_heads']} "
        f"head_dim={config['head_dim']} "
        f"dtype=bfloat16 batch={args.batch_size}"
    )
    print(f"logical cache growth: {format_bytes(per_token)} per sequence token")
    print("sequence_length,logical_kv_bytes,logical_kv_gib")

    rows = []
    for length in lengths:
        value = cache_bytes(config, length, args.batch_size)
        row = {
            "sequence_length": length,
            "logical_kv_bytes": value,
            "logical_kv_gib": value / 1024**3,
        }
        rows.append(row)
        print(f"{length},{value},{row['logical_kv_gib']:.6f}")

    output = Path(__file__).resolve().parent / "out" / "kv_cache_memory.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
