#!/usr/bin/env bash
# KDA attention 5 -> 4 bits without a full conversion: requantize only the 63 KDA layers'
# qkv/f/g/o projections of the base pack from scripts/quantise.sh (exllamav3 v1.6.6 + the Python-only overlay
# tools/requant/overlay/requant_dense.py, bind-mounted), then assemble the new pack from the base by byte copies
# (tools/requant/dense_pack.py), check every kept tensor byte-identical, verify the matrix count. Resumable: rerun the
# same command (the job resumes from its checkpoint; finished steps are skipped). About 70 min on one Spark.
#   scripts/requant_kda.sh full|trial [extra requant_dense args, e.g. --max_layers 2]
# Run detached on the node with the base pack and the FP8 source:
#   ssh -n -f <source node> 'cd <recipe> && setsid nohup scripts/requant_kda.sh full >> logs/requant-kda4.log 2>&1 < /dev/null &'
# Writes $WORK/summary.txt (DONE/FAILED line last). Takes the GPU lock (noclobber) unless LOCK_HELD=1.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
MODE="${1:-full}"; shift || true
BITS="${BITS:-4}"
case "$MODE" in
  full)  BASE="$MODELS_ROOT/packs/Ling-3.1-flash-1019-exl3-2.7bpw"; SRC="$SRC_DIR"
         OUT="${OUT:-$MODELS_ROOT/packs/Ling-3.1-flash-1019-exl3-kda4}"; WORK="${WORK:-$MODELS_ROOT/requant/work-kda4}"
         EXPECT=106497 ;;
  # trial/src: symlinks into $SRC_DIR (mounted at its own path too) + the truncated config/index
  trial) BASE="$MODELS_ROOT/trial/out"; SRC="$MODELS_ROOT/trial/src"
         OUT="${OUT:-$MODELS_ROOT/requant/trial-kda4}"; WORK="${WORK:-$MODELS_ROOT/requant/work-trial}"
         EXPECT=6201 ;;
  *) die "usage: $0 full|trial [requant_dense args]" ;;
esac
# The converter's calibration rows (250 x 2048, seeded corpus) as saved in a converter checkpoint, when one is kept
# (any conversion of this source at the same rows/cols): if present, the requant checks its own rows equal to it.
IDS_REF="${IDS_REF:-$MODELS_ROOT/trial/work/ckpt/original_input_ids.safetensors}"
SUMMARY="$WORK/summary.txt"
INCLUDE="*.attention.qkv_proj,*.attention.f_proj,*.attention.g_proj,*.attention.o_proj"
SITE=/usr/local/lib/python3.12/dist-packages/exllamav3
mkdir -p "$WORK" "$CACHE_DIR"
say() { log "$*"; echo "[$(date '+%F %T')] $*" >> "$SUMMARY"; }
fail() { say "FAILED: $*"; exit 1; }
[[ -f "$BASE/config.json" ]] || fail "no base pack at $BASE"
ids=(); if [[ -f "$IDS_REF" ]]; then ids=(-v "$IDS_REF:/ids_ref.safetensors:ro"); else log "no $IDS_REF: calibration rows not cross-checked"; fi

if [[ "${LOCK_HELD:-0}" != 1 ]]; then
  ( set -o noclobber; echo "LingSpark KDA4 requant $MODE $(date)" > "$MODELS_ROOT/gpu.lock" ) \
    || fail "gpu.lock held: $(cat "$MODELS_ROOT/gpu.lock")"
  trap 'rm -f "$MODELS_ROOT/gpu.lock"' EXIT
fi

# 1. Requantize (GPU)
if [[ ! -f "$WORK/requant-done.json" ]]; then
  resume=(); [[ -f "$WORK/args.json" ]] && resume=(-r)
  say "requant $MODE: $BASE (+ $SRC) -> $WORK ${resume[*]} $*"
  docker run --rm --name "ling-requant-$MODE" --gpus all --ipc=host --shm-size 16g \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v "$BASE:/base:ro" -v "$SRC:/src:ro" -v "$WORK:/work" -v "$CACHE_DIR:/root/.cache" \
    "${ids[@]}" -v "$SRC_DIR:$SRC_DIR:ro" \
    -v "$PWD/tools/requant/overlay/requant_dense.py:$SITE/conversion/requant_dense.py:ro" \
    -w / --entrypoint env "$IMAGE:latest" -u TORCH_ALLOW_TF32_CUBLAS_OVERRIDE \
    python3 -m exllamav3.conversion.requant_dense -b /base -s /src -w /work --bits "$BITS" \
      --include "$INCLUDE" --layer_types linear_attention ${ids[0]:+--ids_ref /ids_ref.safetensors} -cpi 1800 "${resume[@]}" "$@" \
    || fail "requant_dense exited $?"
  [[ -f "$WORK/requant-done.json" ]] || { say "requant stopped before the end (resumable: rerun)"; exit 0; }
  say "requant done: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["qtensors"], "blocks in", d["seconds"], "s, ids sha256", d["ids_sha256"])' "$WORK/requant-done.json")"
fi

# 2. Assemble from the base by byte copies, 3. check kept tensors byte-identical, 4. verify the matrix count
if [[ ! -f "$OUT/quantization_config.json" ]]; then
  say "assemble -> $OUT"
  python3 tools/requant/dense_pack.py assemble --base "$BASE" --qtensors "$WORK/qtensors" --out "$OUT" \
    --expect-bits "$BITS" --note "KDA attention (qkv/f/g/o, 63 layers) requantized 5 -> $BITS bits; every other tensor byte-identical to the base" \
    >> "$WORK/assemble.log" 2>&1 || fail "assemble (see $WORK/assemble.log)"
  tail -1 "$WORK/assemble.log" >> "$SUMMARY"
fi
python3 tools/requant/dense_pack.py check --base "$BASE" --pack "$OUT" --qtensors "$WORK/qtensors" \
  > "$WORK/check.log" 2>&1 || fail "byte-identity check (see $WORK/check.log)"
tail -3 "$WORK/check.log" >> "$SUMMARY"
python3 tools/exl3/verify_exl3_pack.py "$OUT" "$EXPECT" > "$WORK/verify.log" 2>&1 || fail "verify (see $WORK/verify.log)"
tail -2 "$WORK/verify.log" >> "$SUMMARY"
say "DONE: $OUT $(du -sb "$OUT" | cut -f1) bytes"
