#!/usr/bin/env bash
# overlay.sh PACK_DIR OVERLAY_DIR YARN_CONFIG(0|1): the directory the serve reads as /model. Every pack file is a link
# to /pack/<name> (the pack's mount inside the container); config.json is a copy, with SGLang's YaRN override for
# 1,048,576 positions when YARN_CONFIG=1. The pack itself is never modified. start.sh runs it on both nodes.
set -euo pipefail
PACK=${1:?pack dir}; OUT=${2:?overlay dir}; YARN_CONFIG=${3:?0 or 1}
[[ -f "$PACK/config.json" ]] || { echo "overlay.sh: no pack at $PACK" >&2; exit 1; }
mkdir -p "$OUT"
find "$OUT" -maxdepth 1 \( -type l -o -name config.json \) -delete
for f in "$PACK"/*; do
  n=$(basename "$f")
  [[ "$n" == config.json ]] || ln -s "/pack/$n" "$OUT/$n"
done
python3 - "$PACK/config.json" "$OUT/config.json" "$YARN_CONFIG" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
if sys.argv[3] == "1":
    c["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144,
                         "rope_theta": 6000000, "partial_rotary_factor": 0.5}
    c["max_position_embeddings"] = 1048576
json.dump(c, open(sys.argv[2], "w"), indent=2)
EOF
