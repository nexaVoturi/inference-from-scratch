"""Step 1e: separate prefill and decode timing for the manual engine.

This is a local timing harness, not a hardware benchmark. It compares the
full-score and tiled attention paths while checking both against the saved
Transformers token reference.

Run from the repository root:
    python3 engine/step1e_profile.py --repeats 3
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from common import device, sync
from step1c_autoregressive import Qwen3Loop


def timed_generate(loop, prompt_ids, new_tokens):
    prompt_len = prompt_ids.shape[0]
    positions = torch.arange(prompt_len, device=loop.dev)

    sync(loop.dev)
    t0 = time.perf_counter()
    logits, cache = loop.forward(prompt_ids, positions)
    sync(loop.dev)
    prefill_ms = (time.perf_counter() - t0) * 1000

    generated = []
    sync(loop.dev)
    t0 = time.perf_counter()
    for step in range(new_tokens):
        token = logits.argmax().reshape(1)
        generated.append(int(token.item()))
        if step == new_tokens - 1:
            break
        position = torch.tensor([prompt_len + step], device=loop.dev)
        logits, cache = loop.forward(token, position, cache)
    sync(loop.dev)
    decode_ms = (time.perf_counter() - t0) * 1000
    return generated, prefill_ms, decode_ms


def summarize(samples):
    return {
        "prefill_ms": statistics.median(x[1] for x in samples),
        "decode_ms": statistics.median(x[2] for x in samples),
        "decode_ms_per_token": statistics.median(x[2] for x in samples) / len(samples[0][0]),
        "total_ms": statistics.median(x[1] + x[2] for x in samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=128)
    args = parser.parse_args()
    if args.new_tokens < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("new-tokens, repeats, and warmup must be valid positive values")

    reference_path = Path(__file__).resolve().parent / "out" / "reference.json"
    reference = json.loads(reference_path.read_text())
    dev = device()
    prompt_ids = torch.tensor(reference["prompt_ids"], device=dev)
    expected = reference["ref_ids"][: args.new_tokens]
    results = {}

    for attention in ("full", "tiled"):
        loop = Qwen3Loop(dev, attention=attention, tile_size=args.tile_size)
        for _ in range(args.warmup):
            timed_generate(loop, prompt_ids, args.new_tokens)

        samples = [
            timed_generate(loop, prompt_ids, args.new_tokens)
            for _ in range(args.repeats)
        ]
        actual = samples[0][0]
        matching = sum(a == b for a, b in zip(actual, expected))
        exact = actual == expected
        results[attention] = {
            **summarize(samples),
            "token_match": f"{matching}/{len(expected)}",
            "exact": exact,
            "samples": [
                {"prefill_ms": p, "decode_ms": d}
                for _, p, d in samples
            ],
        }
        print(
            f"{attention}: prefill {results[attention]['prefill_ms']:.1f} ms, "
            f"decode {results[attention]['decode_ms_per_token']:.1f} ms/token, "
            f"total {results[attention]['total_ms']:.1f} ms, "
            f"match {matching}/{len(expected)}"
        )
        if not exact:
            raise SystemExit(f"{attention} attention failed the token reference")

    output = Path(__file__).resolve().parent / "out" / "profile.json"
    output.write_text(json.dumps({
        "device": dev,
        "prompt_tokens": len(reference["prompt_ids"]),
        "new_tokens": args.new_tokens,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "tile_size": args.tile_size,
        "results": results,
    }, indent=2) + "\n")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
