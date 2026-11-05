# LingSpark

LingSpark is Ling-3.1-flash on 2x DGX Spark with TensorFold.

inclusionAI's Ling-3.1-flash-1019 (560B parameters, MoE, about 25B active per token; 72 layers, 63 of them linear
attention (KDA) and 9 full attention (MLA), plus an MTP layer) served across two DGX Sparks with tensor parallelism.
The model runs on TensorFold's Zig engine (v1.2.2, plus LingSpark's Ling family in `patches-zig/`) from our own EXL3
quantisation at 2.71 bits per weight (bpw) on average. Ling's own MTP layer drafts tokens for speculative decoding.

Text only. The default serves one conversation with the model's full 1M-token context window.

## Speed and quality

Speed, measured with [spark-bench](https://github.com/ajensenwaud/spark-bench) on the release configuration (1M
window, one conversation), both GPUs free, no other client, 512-token replies, thinking off:

| | LingSpark |
|---|---|
| Decode, 1 conversation, greedy (tok/s): chat / code / agent / report | 32.2 / 33.6 / 35.2 / 33.1 |
| Decode, 1 conversation at inclusionAI's sampling (temperature 1.0, top_p 0.95, top_k 20) | 31.0 / 35.2 / 33.6 / 31.7 |
| Prompt read (tok/s) at 4K / 16K / 64K / 128K tokens | 639 / 622 / 552 / 479 |
| First token after a 4K / 16K / 64K / 128K-token prompt (s) | 6.3 / 25.7 / 116 / 268 |
| Decode after a 4K / 16K / 64K / 128K-token prompt (tok/s) | 27.4 / 28.9 / 26.7 / 25.9 |
| Context per conversation | 1,048,576 tokens (1 slot) |

A 256K window (`CONTEXT=262144`) decodes about 8% faster: 34.3 / 36.5 / 39.1 / 35.4 tok/s for one conversation,
37-43 tok/s combined for two. Before the engine work below, the first working build decoded at 21-28 tok/s.

Quality, against Bertholomus AI's 3.03 bpw pack on the same engine and the same items (64K window, thinking on,
inclusionAI's sampling, 32K-token budget a reply):

| | **LingSpark 2.71 bpw (197 GB)** | Bertholomus 3.03 bpw (221 GB) |
|---|---|---|
| MMLU-Pro, 200 questions | **153 (76.5%)** | 152 (76.0%) |
| GSM8K, 500 questions | **488 (97.6%)** | 485 (97.0%) |
| KL divergence from the FP8 original (WikiText-2, 100 x 2,048 tokens) | 0.150 | 0.092 |
| Same top token as the FP8 original | 89.8% | 92.2% |
| Perplexity vs the FP8 original | +6.6% | +1.6% |

Paired on identical items (McNemar), neither task shows a measurable difference (p = 1.0 and 0.45); 200 questions
only detect gaps of about 5 points. Token by token our pack sits further from the original, the price of a pack
small enough for the 1M window.

Ling sometimes loops in its reasoning until the token limit ("Done. Final." over and over). LingSpark turns on
TensorFold's loop guard by default, which rescues some of those replies and changes nothing in the others; for agents,
also set a lower `max_tokens` than the 32K default.

Reading a new long prompt is slow: about 26 s at 16K tokens, 2 minutes at 64K, 4.5 minutes at 128K, 12.5 minutes at
250K and 85 minutes at 900K. Follow-up turns reuse the cached conversation and start in 7-8 s even at 880K-900K.

## What you need

- Two DGX Sparks joined by their CX7 ports, Docker with the NVIDIA runtime on both, SSH from the head (rank 0) to the
  worker (rank 1) without a password prompt. Hosts, addresses and network interface names go in `scripts/local.sh`
  (copy `scripts/local.sh.example`); every setting and its default is in `scripts/config.sh`.
- About 200 GB of free disk on each Spark for the pack, plus room for the images and the engine build, and up to
  64 GB on the head for saved conversations.
- Nothing else serving on either GPU: at 1M the serve leaves about 7 GiB free on each Spark.
- Ports: 8892 (API), 29651-29652 (between the ranks).

## Run it

On the head:

```bash
scripts/prepare.sh       # the engine: TensorFold v1.2.2 + patches-zig/, built in a container
scripts/fetch_pack.sh    # the 2.71 bpw pack from Hugging Face (197 GB, pinned revision) to both Sparks, verified
./start.sh               # OpenAI-compatible API on :8892, model id Ling-3.1-flash, 1M context
./stop.sh
```

The model weights can be downloaded from: [ajensenwaud/Ling-3.1-flash-1019-exl3-2.71bpw](https://huggingface.co/ajensenwaud/Ling-3.1-flash-1019-exl3-2.71bpw).

## How it was built

- Engine: TensorFold v1.2.2 had no served Ling family. LingSpark starts from Jürgen Schmied's Ling port (TensorFold
  pull request #874) and carries it onto v1.2.2 on TensorFold's shared EXL3 operators, then adds the two-Spark split,
  YaRN to 1M, the split full-attention cache, and kept prompt states in memory and on disk (26 patches). It was
  checked against exllamav3's reference forward within exllamav3's own noise; drafted replies equal plain ones.
- Speed: CUDA graphs for decode, RoCE gathers between the Sparks, the expanded form of the MLA prompt attention,
  tensor-core tiles for prompt projections, kept states left in place, and the 4-bit KDA attention.
  `docs/ENGINE.md` lists the engine's switches.
- Pack: exllamav3 v1.6.6 from inclusionAI's FP8 release. Routed experts at 3 bits in the first and last 17 MoE layers
  and 2 bits in the middle 34 (down projections 3 bits throughout), KDA attention 4, MLA attention, shared experts and
  head 6, the first four dense layers 4, the MTP layer 3: 2.71 bpw on average.

## Credits

Thanks to inclusionAI for Ling-3.1-flash-1019; Ash Hart ([@ashxhart](https://x.com/ashxhart)) and TensorFold's
contributors for [TensorFold](https://github.com/ashhart/TensorFold); Jürgen Schmied
([@jschmied](https://github.com/jschmied)) for the Ling port LingSpark's engine family starts from; turboderp for
[exllamav3](https://github.com/turboderp-org/exllamav3) and the reference Ling packs; Bertholomus AI for the 3.03 bpw
pack and recipe used as the comparison point; Mia AI Lab for the base image. Full list: [CREDITS.md](CREDITS.md).

## License

LingSpark's own code is under the BSD 3-Clause License (`LICENSE`). The engine patches in `patches-zig/` modify
TensorFold (Apache-2.0) and stay under its licence; `tools/requant/overlay/requant_dense.py` builds on exllamav3 and
stays under its MIT licence. The model and the pack are under inclusionAI's MIT licence, shipped with the pack.
