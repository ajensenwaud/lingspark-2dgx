# Ling-3.1-flash in the patched engine

Reference for the `ling` CUDA family that `patches-zig/` adds to TensorFold v1.2.2 (the same text ships in the patched
tree as `docs/recipes/ling-3.1-flash-cuda.md`), followed by the switches LingSpark's scripts use.

The native CUDA build serves inclusionAI's Ling-3.1-flash (`bailing_moe_v3`: 63 KDA and 9 MLA layers, 512 routed
experts, an MTP layer) from bf16, FP8 block-128 and EXL3 checkpoints on one GPU, and from EXL3 packs on two ranks over
hosts. Rank 0 serves HTTP and rank 1 follows its steps, as [GLM-5.3-Flash on CUDA](https://github.com/ashhart/TensorFold/blob/v1.2.2/docs/recipes/glm-5.3-flash-cuda.md) does.

## Context past 262,144: YaRN

The checkpoint's native window is 262,144 positions (`max_position_embeddings`, rope theta 6,000,000). For 1,048,576,
give the config SGLang's override, as transformers reads it:

```json
"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144,
                 "rope_theta": 6000000, "partial_rotary_factor": 0.5},
"max_position_embeddings": 1048576
```

Only `yarn` is read; other rope types are refused by name. The MLA layers' 64 rope columns (and the MTP layer's) take
transformers' `_compute_yarn_parameters` frequencies (beta_fast 32, beta_slow 1, the ramp bounds truncated) and its
attention factor, 0.1 ln 4 + 1, on cos and sin; the softmax scale changes only when the config gives
`mscale_all_dim`, as Ling's modeling applies it. `TF_LING_YARN` picks when:

| `TF_LING_YARN` | rope |
| --- | --- |
| `auto` (default) | YaRN when `--context` is past the native window, the plain rope otherwise |
| `on` | YaRN at any context (needs `rope_scaling`; without it the 1M override above is assumed) |
| `off` | the plain rope; a `--context` past the native window is refused |

The plain rope is unchanged bit for bit when YaRN is off.

## Two ranks

`TF_LING_RANK` of `TF_LING_WORLD` (1 or 2) places each rank and `NCCL_COMM_ID` names rank 0's address and port; each
rank keeps a TCP line to rank 0 on the next port, and a rank that fails stops the other. Each rank loads half the KDA
and MLA heads, half of every MLP and expert width (EXL3's 128-wide blocks) and half the vocabulary of the head. The
latent cache is whole on both ranks up to the native window; past it each rank keeps every other 512-position chunk
(`TF_LING_KV_SPLIT`), so at 1M a rank holds half the cache (5.6 GiB instead of 11.25 on the full model). Each rank then
runs all 64 heads' attention over its own chunks (the absorbed queries gathered first), merges its chunks per head in
fp32, swaps the other rank's heads' partials, and merges the two as GLM's chunk merge does; the result differs from the
whole cache only in the order the chunks' softmax sums are added. Give both ranks the same arguments:

| `TF_LING_KV_SPLIT` | latent cache at two ranks |
| --- | --- |
| `auto` (default) | split when `--context` is past 262,144 |
| `on` | split at any context (two ranks only) |
| `off` | whole on both ranks |

```bash
# on rank 1
TF_LING_RANK=1 TF_LING_WORLD=2 NCCL_COMM_ID=192.0.2.1:29651 NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0 \
  tensorfold-native serve /models/ling-3.1-flash-exl3 --context 262144
# on rank 0 (192.0.2.1)
TF_LING_RANK=0 TF_LING_WORLD=2 NCCL_COMM_ID=192.0.2.1:29651 NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0 \
  tensorfold-native serve /models/ling-3.1-flash-exl3 --context 262144
```

Other formats and worlds other than 1 and 2 are refused at TP.

## Switches (environment, both ranks the same)

Defaults are what LingSpark serves; each switch turns one piece back off for a comparison.

| Variable | Default | What it does |
| --- | --- | --- |
| `TF_LING_GRAPHS` | on (`0`: off) | decode steps (verify windows and MTP draft chains) as CUDA graphs; replays are byte-equal to eager steps |
| `TF_LING_ROCE` | on (`0`: NCCL) | the per-layer all-gathers between the two ranks as main's one-shot RoCE gathers |
| `TF_LING_MTP_DRAFTS` | 3 | tokens the model's own MTP layer drafts a round (1 to the engine's maximum); drafted replies equal plain ones |
| `TF_LING_PROMPT_MLA` | expand (`absorb`: off) | prompt rows' MLA attention in the expanded form (kv_b over the cache rows, flash tiles) |
| `TF_LING_PROMPT_TILES` | on (`0`: off) | EXL3 prompt projections on main's tensor-core tiles |
| `TF_LING_KEEP_INPLACE` | auto (`on`, `off`) | a kept prompt state leaves its cache rows in place instead of copying them; auto does so past the native window, where a full copy does not fit |

## Kept prompt states

`--prompt-cache-gib N` keeps prompt states (the KDA layers' recurrent state plus the MLA and MTP layers' rows) so a
follow-up turn reads only its new tokens. With `--learn-conversations --spill-gib N` each kept state is also written
behind to disk through main's rows store, rank 0 holding both ranks' shares, and an evicted or restarted chat reads it
back instead of reading the prompt again. A state is staged whole in host memory on its way to disk, under a family
permit of 4 GiB at the native window and 2 GiB past it: at a 1M window, states of more than about 150K tokens stay in
memory only.
