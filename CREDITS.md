# Credits

LingSpark stands on other people's work. Thank you to:

- **inclusionAI** for [Ling-3.1-flash-1019](https://huggingface.co/inclusionAI/Ling-3.1-flash-1019) (MIT), the model
  LingSpark serves, its block-FP8 release that our pack was quantised from, and its recommended sampling settings.
- **TensorFold** ([ashhart/TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0) by Ash Hart
  ([@ashxhart](https://x.com/ashxhart)) and its contributors. The engine, its Zig/CUDA build, the shared EXL3
  operators, CUDA graph cache, tensor-core prompt tiles, prompt cache and disk store, loop guard and two-rank serving
  are TensorFold's; LingSpark is a set of patches on TensorFold v1.2.2 (`patches-zig/`).
- **Jürgen Schmied** ([@jschmied](https://github.com/jschmied)), whose Ling port to TensorFold
  ([pull request #874](https://github.com/ashhart/TensorFold/pull/874)) is the starting code of LingSpark's engine
  family. Patch 0001 in `patches-zig/` is his work under his name; the later patches build on it.
- **turboderp** for [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT): the EXL3 format, the quantiser
  that made our pack (v1.6.6, with its native Ling-3.1 support), the reference forward the engine was checked
  against, and the KL/perplexity method. Also for
  [turboderp/Ling-3.1-flash-exl3](https://huggingface.co/turboderp/Ling-3.1-flash-exl3), the reference packs whose
  layout our pack's matrix count was checked against. `tools/requant/overlay/requant_dense.py` is built on
  exllamav3's converter and stays under its MIT licence.
- **Bertholomus AI** for
  [bertholomus/Ling-3.1-flash-1019-EXL3-3.03bpw](https://huggingface.co/bertholomus/Ling-3.1-flash-1019-EXL3-3.03bpw)
  (MIT) and its recipe [ling-3.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/ling-3.1-tensorfold-tp2-2xgb10):
  the 3.03 bpw pack is the comparison point in LingSpark's quality tables, run on the same engine and items.
- **Mia's AI Lab** ([MiaAI-Lab](https://github.com/MiaAI-Lab)) for the two-Spark recipe layout and the GLM-5.3
  TensorFold image
  ([MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
  Apache-2.0) that LingSpark's build and converter images use as their CUDA base.
- **b12x** ([local-inference-lab/b12x](https://github.com/local-inference-lab/b12x), Apache-2.0), Luke Alonso and the
  b12x contributors: TensorFold's two-rank RoCE gather, which LingSpark uses between the Sparks, adapts b12x's
  RoCEnante one-shot all-gather.
- **SGLang** for the YaRN `rope_scaling` override that stretches Ling's 262,144-token window to 1,048,576.
- **Zig** ([ziglang.org](https://ziglang.org)), the toolchain TensorFold's engine is built with.

Evaluation data:

- **GSM8K** ([openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), MIT), Cobbe et al., 2021.
- **MMLU-Pro** ([TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), MIT), Wang et al., 2024.
- **WikiText-2** ([Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), CC BY-SA 3.0), Merity et
  al., 2016, for the KL and perplexity comparison.

Speed numbers come from [spark-bench](https://github.com/ajensenwaud/spark-bench).

Any errors in LingSpark are mine, not theirs. LingSpark's own code is under the BSD 3-Clause License; the patches
and tools above that modify other projects keep those projects' licences.
