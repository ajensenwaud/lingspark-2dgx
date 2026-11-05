# LingSpark (Ling-3.1-flash-1019 on two DGX Sparks via TensorFold's Zig engine): shared settings. Override any of
# them in scripts/local.sh (gitignored; start from scripts/local.sh.example).
# shellcheck disable=SC2034

# --- nodes -------------------------------------------------------------------------------------------------------
# No defaults: set these in scripts/local.sh. Scripts that reach the other node call need_hosts (below).
HEAD="${HEAD:-}"                              # rank 0 (serves the API), e.g. spark-a
WORKER="${WORKER:-}"                          # rank 1 as ssh reaches it over the CX7 link, e.g. user@192.0.2.2
MASTER="${MASTER:-}"                          # rank 0's CX7 address (rendezvous + NCCL), e.g. 192.0.2.1
SRC_NODE="${SRC_NODE:-}"                      # holds the source checkpoint and runs the conversion (disk)

# --- checkpoints -------------------------------------------------------------------------------------------------
# Source for our EXL3 pack: inclusionAI's block-FP8 release (e4m3, 128x128 scales), MIT, ungated. 54 shards, 573.8 GB.
SRC_REPO="inclusionAI/Ling-3.1-flash-1019"
SRC_REV="dddfee4bb43519a92761fbc5cefe3f0d3ff4aded"
SRC_SHARDS=54

MODELS_ROOT="${MODELS_ROOT:-$HOME/ling}"
SRC_DIR="${SRC_DIR:-$MODELS_ROOT/src/Ling-3.1-flash-1019}"
WORK_ROOT="${WORK_ROOT:-$MODELS_ROOT/quant-work}"
CACHE_DIR="${CACHE_DIR:-$MODELS_ROOT/cache}"

# --- converter image (docker/Dockerfile, scripts/build-image.sh) -------------------------------------------------
# A TensorFold GB10 runtime plus upstream exllamav3 at EXL3_COMMIT (v1.6.6: Ling-3.1 native).
IMAGE="${IMAGE:-ling-exl3}"
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529}"
EXL3_COMMIT="bd1f519172406a9560bd5c90e61000691bb69a07"   # exllamav3 v1.6.6

# --- Zig engine (scripts/prepare.sh) -------------------------------------------------------------------------------
# TensorFold's Zig engine at the v1.2.2 tag plus patches-zig/ (the ling CUDA family: TP=1 and TP=2, EXL3 through
# main's shared operators, YaRN for 1M). ZIG_TREE is the git tree those patches produce on TF_COMMIT (commit hashes
# depend on who applies them; trees do not): update it whenever patches-zig/ changes.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold}"
TF_TAG="v1.2.2"
TF_COMMIT="00e7dddbdb2dd0c481f3d7e8d850fa3e1f2e80b7"   # v1.2.2^{commit}; .zig-version 0.17.0
ZIG_TREE="561e0e75531d44957ebb6e96f0fa3a62fd3839ad"   # git tree of TF_COMMIT with patches-zig/ applied (prepare.sh checks)
ZIG_DIR="${ZIG_DIR:-$MODELS_ROOT/engine/tensorfold-zig}"   # the patched tree and its build (zig-out/), same path on both nodes
# The build image: Zig 0.17.0 on BASE_IMAGE (docker/Dockerfile.zig). prepare.sh builds it when it is missing.
ZIG_BUILD_IMAGE="${ZIG_BUILD_IMAGE:-ling-zig-build:0.17.0}"
ZIG_BUILD_MEM="${ZIG_BUILD_MEM:-12g}"          # the build container's memory cap
ZIG_BUILD_JOBS="${ZIG_BUILD_JOBS:-8}"
ZIG_RUN_IMAGE="${ZIG_RUN_IMAGE:-$ZIG_BUILD_IMAGE}"   # the CUDA runtime the binaries run in (present on both nodes)

# --- the pack (scripts/fetch_pack.sh) ------------------------------------------------------------------------------
# Our own EXL3 pack: 2.71 bpw (routed experts 2-3 bits, KDA attention 4), sized so 1M context fits two Sparks.
PACK_REPO="${PACK_REPO:-ajensenwaud/Ling-3.1-flash-1019-exl3-2.71bpw}"   # Hugging Face
PACK_REV="${PACK_REV:-ddb290bc9999f720849daaaed2a5bcaa7a033609}"   # the published pack's revision
PACK_MATRICES=106497                          # EXL3 matrices in the pack, MTP layer included (fetch_pack.sh checks)
PACK_NAME="${PACK_NAME:-Ling-3.1-flash-1019-exl3-kda4}"
PACK_DIR="${PACK_DIR:-$MODELS_ROOT/packs/$PACK_NAME}"   # the same path on both nodes

# --- serving (./start.sh, ./stop.sh) ---------------------------------------------------------------------------------
# tensorfold-native serve on both nodes: rank 0 (HEAD) serves the API, rank 1 (WORKER) follows its steps.
SERVED_NAME="${SERVED_NAME:-Ling-3.1-flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"                          # API port
TP_PORT="${TP_PORT:-29651}"                   # rank rendezvous on MASTER (NCCL_COMM_ID); the next port carries the TCP line
# Context per stream. The native window is 262,144; past it YaRN (factor 4) stretches the rope to 1,048,576. start.sh
# then serves a config overlay (the pack's files linked, config.json with SGLang's YaRN rope_scaling), never edits the
# pack. YARN=auto turns YaRN on only past the native window; on/off force it (TF_LING_YARN).
CONTEXT="${CONTEXT:-1048576}"
NATIVE_CONTEXT=262144
YARN="${YARN:-auto}"
PARALLEL="${PARALLEL:-1}"                     # stream slots
MAX_TOKENS="${MAX_TOKENS:-32768}"
# Prompt cache budget (GiB a rank; 0 turns it off). Empty: 3 past the native window, 6 at or below it (start.sh). Sized
# so the node with less free memory keeps >= 3 GiB with a full cache. Measured at 1M: 8.05 GiB MemAvailable after the
# warm-up; a 119K prompt took 2.71 = 0.77 for its kept state (10 MLA layers x 59.5K rows a rank x 1,152 B + 0.13 KDA)
# + 1.94 of first-request scratch. Budget = 8.05 - 1.94 - 3 = 3.11 -> 3 at 1M (a state up to ~530K tokens fits it).
# 256K (cache not split): 10.96 after the warm-up - 1.94 - 3 = 6.02 -> 6 (two full 256K states, 2.94 each).
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-}"
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-5}" # default 12.2 leaves no room at 1M; 5 = 3 GiB floor + ~2.4 a 1st request adds unseen
# Request defaults when a request omits them: inclusionAI's recommended sampling, plus main's --loop-guard (closes a
# reasoning block that repeats itself verbatim; on MMLU-Pro it rescued 2 of 28 looping replies, while 40 normal MMLU-Pro
# and 100 GSM8K replies stayed byte-identical).
SAMPLING_ARGS="${SAMPLING_ARGS:---temperature 1.0 --top-p 0.95 --top-k 20 --loop-guard}"
GUARD_MIB="${GUARD_MIB:-500}"                 # mem guard: both ranks removed when either node's MemAvailable falls below
ZIG_CONTAINER="${ZIG_CONTAINER:-ling-zig}"    # rank 1: $ZIG_CONTAINER-follow on the worker
PROMPT_TILES="${PROMPT_TILES:-}"              # empty: the engine's default (EXL3 prompt projections on the tensor-core tiles); 0 turns them off
# Kept prompt states on disk too: every state the prompt cache keeps is written behind to $SPILL_DIR on the head (both
# ranks' shares in rank 0's store), and evicted or restarted chats read it back instead of prefilling. SPILL_GIB caps
# the disk it takes (0 turns it off); SPILL_DAILY_GIB caps what it writes a day.
SPILL_GIB="${SPILL_GIB:-64}"
SPILL_DIR="${SPILL_DIR:-$MODELS_ROOT/spill}"
SPILL_DAILY_GIB="${SPILL_DAILY_GIB:-1024}"
ZIG_ENV="${ZIG_ENV:-}"                        # extra "-e NAME=value ..." for both ranks
# NCCL over the CX7 link: set in scripts/local.sh (interface, RoCE HCA, GID index).
NCCL_IFNAME="${NCCL_IFNAME:-}"
NCCL_HCA="${NCCL_HCA:-}"
NCCL_GID="${NCCL_GID:-3}"

# --- helpers -----------------------------------------------------------------------------------------------------
log() { printf '[%s %s] %s\n' "$(date +%H:%M:%S)" "$(basename "$0")" "$*" >&2; }
die() { log "error: $*"; exit 1; }
need_hosts() { local v; for v in "$@"; do [[ -n "${!v}" ]] || die "$v is empty: set it in scripts/local.sh (copy scripts/local.sh.example)"; done; }

if [[ -f "$(dirname "${BASH_SOURCE[0]}")/local.sh" ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/local.sh"
fi
