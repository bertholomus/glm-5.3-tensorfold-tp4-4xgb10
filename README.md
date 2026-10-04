# GLM-5.3 on 4× DGX Spark (GB10) — TensorFold TP4 recipe

**BertholomusAI recipe for the full GLM-5.3 (zai-org/GLM-5.3, 753B, `glm_moe_dsa`; not the Flash model) on four
NVIDIA DGX Spark / GB10 nodes, tensor-parallel 4, served by [TensorFold](https://github.com/ashhart/TensorFold).**

This repository is the recipe: the quantization formula, the launch scripts, the benchmark harness and the measured
numbers. The engine is TensorFold, written by **ashhart**. Our engine changes for this model live on a branch of our
fork, [bertholomus/TensorFold `glm-dsa-tp4`](https://github.com/bertholomus/TensorFold/tree/glm-dsa-tp4), and are
intended for upstream. See [Credits](#credits) for who made what.

> **Before you run this.** For most people, [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is the better
> everyday model: faster, with native vision. We built this for one reason, quality: as close to the full model as four
> Sparks allow. This first release has **no vision**, and long prompts are slow (~400 tok/s prefill, ~76 s to the first
> token at 32K). **No DFlash:** the only DFlash drafter for the full GLM-5.3 is CC BY-NC-ND 4.0, so we draft with
> GLM-5.3's own MTP layer. That costs speed, not quality.

> Hosts and addresses in this repository (`spark1`…`spark4`, `10.0.0.x`) are placeholders. Substitute your own.

| | |
|---|---|
| **Model** | GLM-5.3 by Z.ai, BF16 checkpoint, all 79 layers + the MTP layer |
| **Quant** | EXL3 3.0 bpw (exllamav3 1.5.3), formula below |
| **Weights** | Our EXL3 3.0 bpw quant of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), GLM-5.3 License (Z.AI): [bertholomus/GLM-5.3-EXL3-3.0bpw](https://huggingface.co/bertholomus/GLM-5.3-EXL3-3.0bpw) |
| **Engine** | TensorFold 0.6.0 + `glm-dsa-tp4` branch (head `ddfad35`; release numbers measured on `3eb35dd`) |
| **Hardware** | 4× GB10 (128 GB unified memory each), direct RoCE links, both 200G ports per node |
| **Parallelism** | TP4, one rank per node; ~70 GiB weights per rank |
| **Context** | 1,048,576 tokens (the model's native window: Q5 cache + context parallelism over the 4 ranks) |
| **Quality** | KL 0.109 vs BF16, top-1 agreement 90.2 % (exllamav3 `model_diff`, wikitext-2, 32 × 2,048) — see [Quality](#quality) |

## Update 2026-10-04 (branch `glm-dsa-tp4` @ `ddfad35`): stability and cold start

Two engine fixes on top of `3eb35dd`. Output is unchanged: greedy equality 4/4 and every concurrent reply equals its
solo run.

- **RDMA fix (`cad0d2a`).** The decode all-gather matched receive completions to peers by QP number across both RDMA
  devices. When two ports gave the same number to different peers, a start could fail with
  `rdma gather: ibv_post_recv failed`. Completions are now matched on their own device only.
- **No first-burst penalty (`4016d27`).** Concurrent decoding now captures every round and draft-step CUDA graph at
  warm-up (50 graphs in 3.0 s) instead of during the first requests. The first 4 code requests after a start now run
  at the steady rate (82.3 tok/s HTTP, config A), where before they paid 36 graph captures. `TF_GLM_PRECAPTURE=0`
  restores the old behaviour.

The release numbers below are unchanged.

## Release numbers (2026-10-04, branch `glm-dsa-tp4` @ `3eb35dd`, 4× GB10, greedy, MTP-3)

Two configurations from the same tree. Every number below is an HTTP measurement with the tools in `bench/`
(prompt time included).

**A. 4 streams, replicated cache** (up to 65,536 tokens a stream):

```
TF_GLM_KV=q5 TF_GLM_EMBED_SPLIT=1 TF_GLM_MTP_REUSE=2 \
  bash scripts/tp4_start.sh 65536 "--mtp-drafts 3 --parallel 4"
```

- 4 concurrent code streams: **82 tok/s aggregate**, first token 0.59–0.60 s
- 4 concurrent chat streams: **71–73 tok/s aggregate**, first token 0.40–0.41 s
- Greedy equality 4/4 against the serial reference

**B. 1M window, 4 streams sharing one pool** (decode context parallelism over the four ranks):

```
TF_GLM_KV=q5 TF_GLM_EMBED_SPLIT=1 TF_GLM_DCP=4 TF_GLM_MTP_REUSE=2 TF_GLM_EXTENTS=1 \
  bash scripts/tp4_start.sh 1048576 "--mtp-drafts 3 --parallel 4"
```

- 4 concurrent code streams: **~64 tok/s aggregate**, first token 0.28–1.10 s
- 4 concurrent chat streams: **56 tok/s aggregate**, first token 0.22–0.82 s
- Single stream: 33–39 tok/s (36.7–43.3 without `--parallel`)
- Needle found at 1,039,064 tokens; prefill 402 tok/s at 32K, first token ~76 s at ~31K
- Every concurrent reply bit-identical to its solo run (8/8 at 1, 2 and 4 streams); greedy equality 4/4

Flags:
- `TF_GLM_KV=q5`: latent cache in exllamav3's 5-bit MLA format, indexer keys FP8 (KL vs a bf16 cache 0.0095 at 32k).
- `TF_GLM_EMBED_SPLIT=1`: the embedding split by vocabulary (1.33 GiB a rank back for cache).
- `TF_GLM_DCP=4`: decode context parallelism, each rank holds a quarter of every cache plane; the design follows
  TensorFold PR #159 by drowzeys, restated for this branch's kernels and cache formats.
- `TF_GLM_EXTENTS=1`: the 4 streams' windows as extents of one shared cache pool.
- `TF_GLM_MTP_REUSE=2`: the MTP draft layer reuses the main model's DSA token selection instead of rescoring.

Heat: the 1M configuration runs the GB10s hot under long prefills (we saw 86 °C on one node). Give every node good
airflow.

## Earlier served configuration (2026-10-03, `1efdb17`)

- **Decode, MTP-3:** 34.6 / 40.6 / 42.1 / 38.1 tok/s on the four `bench/tf_greedy.py` prompts.
- **Prefill:** 393 tok/s on a 129,837-token prompt (DCP4); 1,018 tok/s on 35,742 tokens with a replicated cache.
- **Long context:** needles found at 130,909 / 523,777 / 1,039,064 tokens at the 1M window.

## Measured (2026-10-01, single stream, greedy, 258K window, bf16 cache)

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

KL divergence against the BF16 model, exllamav3 1.5.3 `eval/model_diff.py` (the converter's own harness), wikitext-2
test split, 32 rows × 2,048 tokens (65,536 positions), BF16 activations, no cache quantization:

| | value |
|---|---|
| KL(P_BF16 ‖ P_quant) (model_diff's first line) | **0.109** |
| KL(P_quant ‖ P_BF16) | 0.123 |
| Top-1 agreement with BF16 | **90.2 %** |
| Perplexity, quant / BF16 | 2.911 / 2.750 (+5.9 %) |
| Per-token KL, median / p90 | 0.014 / 0.259 |

On the engine itself (the served prompt path scored against stored BF16 logits of the same panel): KL 0.111, top-1
90.2 %. A longer panel (4 × 16,384 wikitext tokens, so DSA's top-2,048 selection is exercised): KL 0.119.

Per-layer drift is smooth down the
stack (no damaged layer from the resumed conversion). A mixed per-expert formula is being researched as a second
candidate and will be measured on the same panel.

## Credits

- **Z.AI** — GLM-5.3, the model and its weights ([zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)). Our quant
  is a derivative of their BF16 weights and stays under the GLM-5.3 License, Copyright (c) 2026 Z.AI.
- **ashhart** — [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0): the engine, the universal EXL3
  experts kernel, the `glm5_next` (GLM-5.3-Flash) family our `glm_moe_dsa` work builds on, and the TP machinery.
- **turboderp** — [EXL3 / exllamav3](https://github.com/turboderp-org/exllamav3) (MIT): the quantization format and
  converter, and the `glm_moe_dsa` reference implementation the engine family was checked against.
- **Projects whose published results shaped this work:** jayleaton
  ([glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark), TensorFold on GB10), vcruz305
  (EXL3 kernel work on GB10). No code from them is included here.
- **drowzeys** — TensorFold PR #159 (decode context parallelism), whose design our `TF_GLM_DCP` follows.
- **NVIDIA** — the PyTorch container and NCCL the ranks run on.
- **BertholomusAI** (Albert Lee / [bertholomus](https://github.com/bertholomus)) — the quant formula and conversion,
  the `glm_moe_dsa` engine branch, the TP4 deployment and the measurements in this repository.

## License

Recipe scripts and documentation: Apache-2.0 (see `LICENSE`), matching TensorFold. The model weights remain under
Z.AI's GLM-5.3 License. Not affiliated with or endorsed by Z.AI, NVIDIA, the TensorFold authors or turboderp.
