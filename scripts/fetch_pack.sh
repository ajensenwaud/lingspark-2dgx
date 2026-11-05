#!/usr/bin/env bash
# Download the published pack instead of making it (about 30 h of conversion): from Hugging Face to PACK_DIR on the head,
# pinned to PACK_REV, run in the engine image (scripts/prepare.sh builds it; nothing installed on the host) as you, so
# no root-owned files; verified; then copied to the worker over the CX7 link and verified there. ~197 GB.
#   scripts/fetch_pack.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
need_hosts WORKER
[[ -n "$PACK_REPO" ]] || die "no PACK_REPO in scripts/config.sh"
[[ -n "$PACK_REV" ]] || die "no PACK_REV pinned in scripts/config.sh: the pack's revision is pinned once its upload is final"
docker image inspect "$ZIG_RUN_IMAGE" >/dev/null 2>&1 || die "no $ZIG_RUN_IMAGE image: run scripts/prepare.sh first"
mkdir -p "$PACK_DIR"
log "$PACK_REPO @ ${PACK_REV:0:12} -> $PACK_DIR (~197 GB)"
docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/out/.hf-home -v "$PACK_DIR:/out" \
  --entrypoint python3 "$ZIG_RUN_IMAGE" -c \
  "from huggingface_hub import snapshot_download; snapshot_download('$PACK_REPO', revision='$PACK_REV', local_dir='/out', max_workers=8)"
rm -rf "$PACK_DIR/.hf-home" "$PACK_DIR/.cache"
python3 tools/exl3/verify_exl3_pack.py "$PACK_DIR" "$PACK_MATRICES" || die "the download does not verify"
log "copying to $WORKER over the CX7 link"
ssh "$WORKER" "mkdir -p '$PACK_DIR'" </dev/null
rsync -a --delete "$PACK_DIR/" "$WORKER:$PACK_DIR/"
ssh "$WORKER" python3 - "$PACK_DIR" "$PACK_MATRICES" < tools/exl3/verify_exl3_pack.py || die "the worker's copy does not verify"
log "pack ready on both nodes: ./start.sh"
