# Methodology

How measurements in this repository are taken, and what they can and cannot be used to claim.

## The disclosure set

Every published result states all of the following. A result missing any of these is
not publishable. The template is in `docs/disclosure-template.md`.

- Engine and exact commit hash or build date
- Model and exact quantization artifact, with provenance (which tool produced it, from which checkpoint)
- GPU SKU and count
- KV-cache dtype
- Input and output token distribution
- Concurrency sweep range, not a single point
- Region and instance type
- Run date
- Number of runs, and whether the reported figure is a median or a mean

## What transfers across hardware

These results can be stated generally, with the hardware noted as a caveat:

- Continuous-batching and scheduler behaviour
- Prefix and KV cache hit rates for a given workload shape
- KV-cache memory arithmetic
- Quantization quality degradation against bit width
- Speculative-decoding acceptance rates for a given draft/target pair
- Chunked-prefill effects on p99 TTFT
- Prompt-shape effects
- The shape of the throughput-latency curve

## What does not transfer

Never generalise these from a consumer GPU to datacentre hardware:

- Absolute tokens per second
- Optimal batch size
- Anything involving FP8 or FP4 on Ada against Hopper or Blackwell
- Multi-GPU scaling and NCCL behaviour
- Disaggregated prefill/decode
- Tokens-per-megawatt claims

When a result falls in this list, say so in the post. The credibility cost of being
caught over-claiming is much higher than the cost of stating a limit.

## Load generator correctness

Single-process asyncio load generators are limited by Python's GIL. vLLM's own bench
has been measured saturating at 146 QPS while targeting 1,000. Many published
high-concurrency numbers are measuring the client rather than the server.

Before any concurrency result above roughly 100 QPS, verify the load generator is not
the bottleneck and publish that verification alongside the result.

## Benchmarking other people's tools

1. Never benchmark an offering of mine against named competitors.
2. Give every named tool a regime where it wins. Neutrality is only credible when the
   results actually split.
3. Send drafts to every named vendor three to four weeks ahead, with a written feedback
   deadline. Run their tuned configuration and report it separately from the stable
   release, clearly labelled.
4. Put caveats above the fold, not in the README.
5. When wrong, publish the error and re-run everything affected, with a dated changelog.
   Never defend the original run.

## Cost figures

Cost per million tokens is stated with the GPU hourly rate used to compute it and the
date that rate was observed. Rates move. A cost figure without a date and a rate is
not a measurement.
