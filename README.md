# GLM-5.3 on 4× DGX Spark (GB10) — TensorFold TP4 recipe

**BertholomusAI recipe for the full GLM-5.3 (zai-org/GLM-5.3, 753B, `glm_moe_dsa`; not the Flash model) on four
NVIDIA DGX Spark / GB10 nodes, tensor-parallel 4, served by [TensorFold](https://github.com/ashhart/TensorFold).**

This repository is the recipe: the quantization formula, the launch scripts, the benchmark harness and the measured
numbers. The engine is TensorFold, written by **ashhart**. Our engine changes for this model live on a branch of our
fork, [bertholomus/TensorFold `glm-dsa-tp4`](https://github.com/bertholomus/TensorFold/tree/glm-dsa-tp4), and are
intended for upstream. See [Credits](#credits) for who made what.

> Hosts and addresses in this repository (`spark1`…`spark4`, `10.0.0.x`) are placeholders. Substitute your own.

| | |
|---|---|
| **Model** | GLM-5.3 by Z.ai, BF16 checkpoint, all 79 layers + the MTP layer |
| **Quant** | EXL3 3.0 bpw (exllamav3 1.5.3), formula below |
| **Engine** | TensorFold 0.6.0 + `glm-dsa-tp4` branch (24 commits) |
| **Hardware** | 4× GB10 (128 GB unified memory each), direct RoCE links, both 200G ports per node |
| **Parallelism** | TP4, one rank per node; ~70 GiB weights per rank |
| **Context** | 258,048 tokens |
| **Status** | Working, measured. Quality (KLD vs BF16) not yet measured — see [Quality](#quality) |

## Measured (2026-10-01, single stream, greedy)

All numbers come from the commit messages on the engine branch, where each change was gated on real runs.

- **Decode, MTP drafts depth 3:** 39.6 / 48.1 / 48.4 / 43.6 tok/s on the four `bench/tf_greedy.py` prompts
  (client-side, 300-token replies, 32K context). Output is token-identical to serial decoding on all four.
- **Decode, serial (no drafts):** 26.9 tok/s.
- **Prefill:** 339 tok/s on a 26,198-token prompt.
- **Long context (`--context 258048`):** needle found at 63,891 / 127,838 / 256,147 tokens; decode 32.1 / 33.1 /
  28.0 tok/s at those depths; prefill 292 / 266 / 213 tok/s (`bench/tf_needle.py`).
- **Gate (`bench/tf_bench.py`):** thinking answer correct, 13,265-token needle exact, tool call well-formed.

## The quant formula

Source: `zai-org/GLM-5.3` BF16. Converter: [exllamav3](https://github.com/turboderp-org/exllamav3) 1.5.3
(`d3739fd`), on one GB10, about 1.8 days.

```
python convert.py -i GLM-5.3-BF16 -o GLM-5.3-EXL3-3.0bpw -w work \
  -b 3.0 -hq -hb 6 -mb 4 -cb mul1
```

| Part | Bits |
|---|---|
| Routed experts (layers 3–77, 256 each) | 3.0 (uniform) |
| Attention, shared experts, dense layers 0–2 (`-hq`) | promoted, ~5 |
| MTP layer experts (`-mb`) | 4 |
| `lm_head` (`-hb`) | 6 |
| Whole model | 3.04 bpw average, 39 shards, 292.7 GB |

Per-rank fit at TP4 (from the real config, `tools/verify_shape_math.py` on the engine branch): 69.86 GiB of weights a
rank, which leaves room for a 258K-token cache at 96,640 bytes a slot.

## Engine work (on `glm-dsa-tp4`)

What the branch adds to TensorFold 0.6.0 for this model. Each item is one or more commits, with its before/after
numbers in the message.

- The `glm_moe_dsa` family (GLM-5.3 non-Flash) on CUDA: MLA with a rope-key cache, DSA sparse attention, the
  indexer, layer norms, routed + shared experts, MTP drafting (`mtp_decode`) bit-identical to serial.
- TP world size as a parameter (2, 4, 6), with balanced uneven partitioning for worlds that do not divide heads,
  experts or vocab.
- One-hop RDMA-write all-gather over both RoCE ports for decode partials (in-graph gather 34.5 → 23.0 µs).
- DSA token selection in bounded row blocks with a radix top-k (795 → 175 ms a layer at 130K tokens).
- CUDA graphs past the dense limit, a sparse-MLA kernel config change, a second stream for the key path and the shared
  expert in decode.
- Prompt-chunk expert tiles sized to the busiest expert (prefill 297 → 339 tok/s).
- Cache admission that counts this family's slot correctly (context 218K → 258K).

## Run it

1. Put the quantized checkpoint at `~/models/GLM-5.3-EXL3-3.0bpw` on all four nodes (each rank reads only its share).
2. On every node, set up the work directory with the engine branch and the container wrapper:
   ```
   mkdir -p ~/glm53-tf && cd ~/glm53-tf
   git clone -b glm-dsa-tp4 https://github.com/bertholomus/TensorFold
   cp /path/to/this/repo/scripts/tfrun.sh . && chmod +x tfrun.sh
   ```
3. On the machine that can ssh to all four, edit the top of `scripts/tp4_start.sh` (node names, master address,
   RoCE interface and HCA names) and run:

```
bash scripts/tp4_start.sh 32768 "--mtp-drafts 3"      # or 258048 for the long window
python bench/tf_bench.py http://127.0.0.1:18090       # on rank 0
```

Ranks run in throwaway `nvcr.io/nvidia/pytorch:26.07-py3` containers (`scripts/tfrun.sh`). The server on rank 0 is
OpenAI-compatible and has no authentication: keep it on loopback or put your own gateway in front.

## Quality

KL divergence against BF16 has **not** been measured yet for this quant, so this repository makes no quality claim
beyond the functional gate above. The KLD panel (exllamav3 `model_diff`, wikitext, 32 × 2048 tokens) is next, and a
mixed per-expert formula is being researched as a second candidate. Results will be added here.

## Credits

- **Z.ai** — GLM-5.3, the model and its weights. Use is subject to the GLM-5.3 license of the base model.
- **ashhart** — [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0): the engine, the universal EXL3
  experts kernel, the `glm5_next` (GLM-5.3-Flash) family our `glm_moe_dsa` work builds on, and the TP machinery.
- **turboderp** — [EXL3 / exllamav3](https://github.com/turboderp-org/exllamav3) (MIT): the quantization format and
  converter, and the `glm_moe_dsa` reference implementation the engine family was checked against.
- **Projects whose published results shaped this work:** jayleaton
  ([glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark), TensorFold on GB10), vcruz305
  (EXL3 kernel work on GB10). No code from them is included here.
- **NVIDIA** — the PyTorch container and NCCL the ranks run on.
- **BertholomusAI** (Albert Lee / [bertholomus](https://github.com/bertholomus)) — the quant formula and conversion,
  the `glm_moe_dsa` engine branch, the TP4 deployment and the measurements in this repository.

## License

Recipe scripts and documentation: Apache-2.0 (see `LICENSE`), matching TensorFold. The model weights remain under
Z.ai's GLM-5.3 license.
