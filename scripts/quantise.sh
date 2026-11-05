#!/usr/bin/env bash
# Quantise the LingSpark base pack (KDA attention at 5 bits) from the FP8 source with upstream exllamav3 (ling-exl3
# image, scripts/build-image.sh) on the node holding the source; scripts/requant_kda.sh then takes KDA to 4 bits.
# About 30 h on one Spark. Resumable: rerun the same command and the job continues from its work dir (-r). Run detached:
#   ssh -n -f <source node> 'cd <recipe> && setsid nohup scripts/quantise.sh >> logs/quantise.log 2>&1 < /dev/null &'
# The GPU lock says SHARED-OK: short correctness tests with a wide memory margin may run beside it, never speed runs.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
RECIPE="${RECIPE:-conversion/ling-2.7.yaml}"
NAME="${NAME:-Ling-3.1-flash-1019-exl3-2.7bpw}"
out="$MODELS_ROOT/packs/$NAME"
work="$WORK_ROOT/$NAME"
# EXL3 matrices outside the MTP layer: 68 MoE layers x 512 experts x 3, 63 KDA x 4 (qkv, f, g, o), 9 MLA x 4
# (q_a, q_b, kv_a, dense; kv_b stays unquantised), 68 shared experts x 3, 4 dense MLPs x 3, the head.
# turboderp's 2.60 pack of the same architecture holds exactly this many (104,953).
EXPECT_MAIN=$(( 68 * 512 * 3 + 63 * 4 + 9 * 4 + 68 * 3 + 4 * 3 + 1 ))
docker image inspect "$IMAGE:latest" >/dev/null 2>&1 || die "no $IMAGE image here: run scripts/build-image.sh"
[[ -f "$SRC_DIR/model.safetensors.index.json" ]] || die "no source checkpoint at $SRC_DIR"
if [[ -f "$out/model.safetensors.index.json" && -f "$out/quantization_config.json" ]]; then log "already done ($out)"; exit 0; fi
( set -o noclobber; echo "LingSpark conversion $NAME $(date) SHARED-OK" > "$MODELS_ROOT/gpu.lock" ) \
  || die "gpu.lock held: $(cat "$MODELS_ROOT/gpu.lock")"
trap 'rm -f "$MODELS_ROOT/gpu.lock"' EXIT
mkdir -p "$work" "$out" "$CACHE_DIR"
resume=(); [[ -f "$work/args.json" ]] && resume=(-r)
log "converting ${resume[*]:+(resuming) }$RECIPE -> $out"
docker run --rm --name ling-quant --gpus all --ipc=host --shm-size 16g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$SRC_DIR:/src:ro" -v "$work:/work" -v "$out:/out" -v "$CACHE_DIR:/root/.cache" -v "$PWD/conversion:/conversion:ro" \
  --entrypoint env "$IMAGE:latest" -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE python3 /tools/convert.py \
  -i /src -w /work -o /out -rcp "/$RECIPE" -hb 6 -mb 3 -cr 250 -cc 2048 -cb mul1 "${resume[@]}"
[[ -f "$out/model.safetensors.index.json" ]] || die "the converter exited without writing a pack"
mats=$(python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['weight_map']; t=[k for k in m if k.endswith('.trellis')]; print(sum('.layers.72.' not in k for k in t), sum('.layers.72.' in k for k in t))" \
  "$out/model.safetensors.index.json")
read -r main mtp <<< "$mats"
[[ "$main" == "$EXPECT_MAIN" ]] || die "$main EXL3 matrices outside the MTP layer, expected $EXPECT_MAIN"
# MTP layer 72: 512 experts x 3, shared expert 3, MLA 4, eh_proj (turboderp's pack: 106,497 in all).
[[ "$mtp" == 1544 ]] || die "MTP layer has $mtp EXL3 matrices, expected 1,544"
python3 tools/exl3/verify_exl3_pack.py "$out" "$(( main + mtp ))"
for f in chat_template.jinja tokenizer.json tokenizer_config.json special_tokens_map.json; do
  [[ -f "$out/$f" ]] || cp "$SRC_DIR/$f" "$out/"
done
log "done: $main + $mtp (MTP) EXL3 matrices, $(du -sh "$out" | cut -f1)"
