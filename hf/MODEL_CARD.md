---
license: mit
base_model: inclusionAI/Ling-3.1-flash-1019
base_model_relation: quantized
pipeline_tag: text-generation
library_name: exllamav3
tags:
- exl3
- tensorfold
- dgx-spark
- gb10
- bailing_hybrid
---

# Ling-3.1-flash-1019, EXL3 2.71 bpw

[inclusionAI/Ling-3.1-flash-1019](https://huggingface.co/inclusionAI/Ling-3.1-flash-1019) (560B MoE, about 25B
active) quantised to EXL3 at 2.71 bits per weight on average, made to run across two NVIDIA DGX Sparks (GB10) with
tensor parallelism and the model's full 1M-token context. The project that serves it:
[LingSpark](https://github.com/ajensenwaud/lingspark-2dgx).

| | |
|---|---|
| Weights | 197 GB, 29 shards (about half on each Spark at TP=2) |
| Bit rates | routed experts 3 bits in the first and last 17 MoE layers, 2 in the middle 34 (down projections 3 throughout); KDA attention 4; MLA attention, shared experts and head 6; dense layers 0-3 at 4; MTP layer 3 |
| Source | inclusionAI's FP8 release @ `dddfee4b` |
| Conversion | exllamav3 v1.6.6 (mul1 codebook, 250 x 2,048 calibration rows); KDA attention then requantised from 5 to 4 bits with every other tensor kept byte for byte |

## Quality

Against Bertholomus AI's 3.03 bpw pack, on the same engine and the same items (64K window, thinking on,
inclusionAI's sampling, 32K-token budget a reply):

| | **This pack, 2.71 bpw** | Bertholomus 3.03 bpw |
|---|---|---|
| MMLU-Pro, 200 questions | **153 (76.5%)** | 152 (76.0%) |
| GSM8K, 500 questions | **488 (97.6%)** | 485 (97.0%) |
| KL divergence from the FP8 original (WikiText-2, 100 x 2,048) | 0.150 | 0.092 |
| Same top token as the FP8 original | 89.8% | 92.2% |
| Perplexity vs the FP8 original | +6.6% | +1.6% |

Neither task shows a measurable difference on identical items (200 questions only detect gaps of about 5 points);
token by token this pack sits further from the original, the price of fitting the 1M window.

## Speed (two DGX Sparks, LingSpark's default 1M window, one conversation, 512-token replies, thinking off)

| | |
|---|---|
| Decode, greedy (chat / code / agent / report) | 32.2 / 33.6 / 35.2 / 33.1 tok/s |
| Decode at inclusionAI's sampling (temperature 1.0, top_p 0.95, top_k 20) | 31.0 / 35.2 / 33.6 / 31.7 tok/s |
| Prompt read at 4K / 16K / 64K / 128K tokens | 639 / 622 / 552 / 479 tok/s |
| First token after a 4K / 16K / 64K / 128K-token prompt | 6.3 / 25.7 / 116 / 268 s |
| Decode after a 4K / 16K / 64K / 128K-token prompt | 27.4 / 28.9 / 26.7 / 25.9 tok/s |
| Context | 1,048,576 tokens (YaRN past the native 262,144) |

A 256K window (`CONTEXT=262144`) decodes about 8% faster: 34.3 / 36.5 / 39.1 / 35.4 tok/s for one conversation,
37-43 tok/s combined for two.

Ling sometimes loops in its reasoning until the token limit; LingSpark turns on TensorFold's loop guard by default,
and agents should set a lower `max_tokens`. Reading a new long prompt is slow (about 2 minutes at 64K tokens, 4.5
minutes at 128K, 85 minutes at 900K); follow-up turns reuse the cached conversation and start in 7-8 s even at 900K.

## Use

With LingSpark (two Sparks): `scripts/prepare.sh`, `scripts/fetch_pack.sh`, `./start.sh`. The pack is a standard
exllamav3 v1.6.6 conversion of Ling-3.1-flash's architecture (`BailingMoeV3ForCausalLM`); it was evaluated only with
LingSpark's engine (TensorFold's Zig engine plus LingSpark's Ling family), not with exllamav3 itself.

## Acknowledgements

inclusionAI for the model; turboderp for exllamav3, which made this pack, and the reference Ling packs; Ash Hart and
TensorFold's contributors for TensorFold, the engine it was evaluated on; Jürgen Schmied for the Ling port LingSpark's
engine starts from; Bertholomus AI for the 3.03 bpw pack used as the comparison point; Mia's AI Lab for the two-Spark
recipe layout. Evaluation data: GSM8K (Cobbe et al., MIT), MMLU-Pro (TIGER-Lab, MIT), WikiText-2 (Merity et al.,
CC BY-SA 3.0). Full list in LingSpark's
[CREDITS.md](https://github.com/ajensenwaud/lingspark-2dgx/blob/main/CREDITS.md).

## Licence

MIT, as the source model: inclusionAI's licence text is in `LICENSE`.
