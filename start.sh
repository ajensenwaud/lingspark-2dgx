#!/usr/bin/env bash
# Start Ling-3.1-flash on TensorFold's Zig engine across both Sparks: rank 1 on the worker, then rank 0 here, which
# serves the OpenAI-compatible API on :$PORT. The pack ($PACK_DIR) and the build ($ZIG_DIR/zig-out, from
# scripts/prepare.sh) sit at the same paths on both nodes. Waits for /v1/models, then sends one warm-up chat.
#   ./start.sh                     1,048,576 tokens of context (YaRN on), 1 stream
#   CONTEXT=262144 ./start.sh      the native window (YaRN off)
#   ./start.sh restart             ./stop.sh first (saves both ranks' logs), then start
# Other settings (scripts/config.sh): PARALLEL, YARN, MAX_TOKENS, PROMPT_CACHE_GIB, MEMORY_RESERVE_GIB, SAMPLING_ARGS,
# GUARD_MIB, ZIG_ENV (extra docker run arguments for both ranks, e.g. "-e NAME=value").
set -euo pipefail
cd "$(dirname "$0")"
source scripts/config.sh
need_hosts HEAD WORKER MASTER NCCL_IFNAME NCCL_HCA

[[ "${1:-}" == restart ]] && ./stop.sh
BIN="$ZIG_DIR/zig-out"
NAT=/zig/native/bin/tensorfold-native
[[ -x "$BIN/native/bin/tensorfold-native" ]] || die "no Zig build in $BIN: scripts/prepare.sh"
[[ -f "$PACK_DIR/config.json" ]] || die "no pack at $PACK_DIR (scripts/fetch_pack.sh)"
ssh "$WORKER" "test -f '$PACK_DIR/config.json' && test -x '$BIN/native/bin/tensorfold-native'" </dev/null \
  || die "the worker ($WORKER) lacks the pack or the build at $PACK_DIR / $BIN"
docker ps -a --format '{{.Names}}' | grep -qx "$ZIG_CONTAINER" && die "$ZIG_CONTAINER exists: ./stop.sh first (or ./start.sh restart)"
ss -ltn "( sport = :$PORT )" | grep -q LISTEN && die "port $PORT is taken"

# YaRN: the config overlay carries the 1M rope_scaling when the context passes the native window (or YARN=on)
yarn_config=0
[[ "$CONTEXT" -gt "$NATIVE_CONTEXT" || "$YARN" == on ]] && yarn_config=1
OVERLAY="$MODELS_ROOT/serve/$PACK_NAME"
scripts/overlay.sh "$PACK_DIR" "$OVERLAY" "$yarn_config"
ssh "$WORKER" "bash -s -- '$PACK_DIR' '$OVERLAY' $yarn_config" < scripts/overlay.sh || die "could not write the overlay on the worker"
log "context $CONTEXT, YARN=$YARN (config rope_scaling: $([[ $yarn_config == 1 ]] && echo yarn x4 || echo none)), $PARALLEL stream(s)"

[[ -n "$PROMPT_CACHE_GIB" ]] || PROMPT_CACHE_GIB=$(( CONTEXT > NATIVE_CONTEXT ? 3 : 6 ))   # config.sh: the arithmetic
pc_arg="--prompt-cache-gib $PROMPT_CACHE_GIB"
spill_arg="" spill_mount=""
if [[ "$SPILL_GIB" != 0 ]]; then   # config.sh: kept states on disk under $SPILL_DIR (rank 0's store)
  mkdir -p "$SPILL_DIR"; ssh "$WORKER" "mkdir -p '$SPILL_DIR'" </dev/null
  spill_arg="--learn-conversations --learn-dir /spill --learn-gib $SPILL_GIB --spill-gib $SPILL_GIB --learn-daily-gib $SPILL_DAILY_GIB"
  spill_mount="-v $SPILL_DIR:/spill"
fi
SERVE_ARGS="serve /model --host $HOST --port $PORT --name $SERVED_NAME --context $CONTEXT --parallel $PARALLEL --max-tokens $MAX_TOKENS $SAMPLING_ARGS $pc_arg $spill_arg"
zig_args() {   # RANK: docker run arguments for one rank
  echo "--gpus all --network host --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1" \
       "--user $(id -u):$(id -g) -e HOME=/tmp" \
       "-e TF_LING_RANK=$1 -e TF_LING_WORLD=2 -e TF_LING_YARN=$YARN -e NCCL_COMM_ID=$MASTER:$TP_PORT" \
       "-e TENSORFOLD_MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB ${PROMPT_TILES:+-e TF_LING_PROMPT_TILES=$PROMPT_TILES}" \
       "-e NCCL_SOCKET_IFNAME=$NCCL_IFNAME -e NCCL_IB_HCA=$NCCL_HCA -e NCCL_IB_GID_INDEX=$NCCL_GID $ZIG_ENV" \
       "-v $PACK_DIR:/pack:ro -v $OVERLAY:/model:ro -v $BIN:/zig:ro $spill_mount --entrypoint $NAT"
}

# Another GPU job on a node (~/ling/gpu.lock) blocks the start, unless its lock says SHARED-OK (a conversion that
# allows short tests beside it) or it is one of this project's own test locks ("G6 recipe ..."). MIN_AVAIL_GIB
# (default 0): refuse when a node has less MemAvailable than that.
lock_ok='l=$(cat ~/ling/gpu.lock 2>/dev/null); [ -z "$l" ] || [ "${l#*SHARED-OK}" != "$l" ] || [ "${l#G6 recipe}" != "$l" ] || { echo "GPU lock: $l" >&2; exit 3; }
m=$(awk "/MemAvailable/ {print int(\$2/1048576)}" /proc/meminfo); [ "$m" -ge '"${MIN_AVAIL_GIB:-0}"' ] || { echo "MemAvailable $m GiB" >&2; exit 3; }'
bash -c "$lock_ok" || die "$HEAD is busy (GPU lock or memory)"

rm -f "$MODELS_ROOT/mem_guard.tripped"
ssh "$WORKER" "rm -f ~/ling/mem_guard.tripped; mkdir -p ~/ling/bin && cat > ~/ling/bin/mem_guard.sh && chmod +x ~/ling/bin/mem_guard.sh" \
  < scripts/mem_guard.sh || die "could not copy the memory guard to the worker"
mkdir -p logs
log "rank 1 on $WORKER"
# shellcheck disable=SC2029
ssh "$WORKER" "$lock_ok
  docker rm -f $ZIG_CONTAINER-follow >/dev/null 2>&1
  docker run -d --name $ZIG_CONTAINER-follow $(zig_args 1) $ZIG_RUN_IMAGE $SERVE_ARGS >/dev/null || exit 4
  GUARD_MIB=$GUARD_MIB setsid nohup ~/ling/bin/mem_guard.sh $ZIG_CONTAINER-follow >/dev/null 2>&1 </dev/null &" </dev/null \
  || die "rank 1 did not start on $WORKER"
log "rank 0 here, :$PORT"
# shellcheck disable=SC2046,SC2086
docker run -d --name "$ZIG_CONTAINER" $(zig_args 0) "$ZIG_RUN_IMAGE" $SERVE_ARGS >/dev/null
GUARD_MIB=$GUARD_MIB GUARD_DIR=$MODELS_ROOT setsid nohup scripts/mem_guard.sh "$ZIG_CONTAINER" "$WORKER" "$ZIG_CONTAINER-follow" \
  >/dev/null 2>&1 </dev/null &
{ echo "# $(date -Is) CONTEXT=$CONTEXT YARN=$YARN ./start.sh"; echo "pack $PACK_DIR (overlay $OVERLAY, yarn config $yarn_config)"
  echo "build $BIN (tensorfold-native $(grep native/bin "$BIN/SHA256SUMS" 2>/dev/null | cut -c1-16))"
  echo "rank0 docker run $(zig_args 0) $ZIG_RUN_IMAGE $SERVE_ARGS"; } > logs/serve-config.txt

log "loading: docker logs -f $ZIG_CONTAINER"
for i in $(seq 1 240); do
  if curl -fsS -m 3 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    log "up after $((i * 5)) s"
    log "engine: $(docker logs "$ZIG_CONTAINER" 2>&1 | grep -m1 'CUDA sm_' | sed 's/^\[tensorfold\] //' | cut -c1-300)"
    log "rope: $(docker logs "$ZIG_CONTAINER" 2>&1 | grep -m1 'YaRN factor' || echo 'plain (YaRN off)')"
    t0=$(date +%s.%N)
    out=$(curl -sS -m 900 -w '\n%{http_code}' "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\": \"$SERVED_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}],
           \"max_tokens\": 1024, \"temperature\": 1.0, \"top_p\": 0.95, \"top_k\": 20, \"seed\": 1,
           \"chat_template_kwargs\": {\"enable_thinking\": false}}") || die "warm-up request failed: docker logs $ZIG_CONTAINER"
    code=$(tail -1 <<< "$out")
    [[ "$code" == 200 ]] || die "warm-up chat returned HTTP $code: $(head -c 300 <<< "$out")"
    log "warm-up: HTTP $code, $(python3 -c 'import json,sys; j=json.loads(sys.argv[1]); print(j["usage"], j["choices"][0]["finish_reason"])' "$(sed '$d' <<< "$out")" 2>/dev/null) in $(python3 -c "print(round($(date +%s.%N) - $t0, 1))") s"
    log "serving $SERVED_NAME (Zig engine, TP=2) at http://$(hostname):$PORT/v1"
    exit 0
  fi
  [[ -f "$MODELS_ROOT/mem_guard.tripped" ]] && die "the memory guard removed the serve: $(tail -3 "$MODELS_ROOT/mem_guard-$ZIG_CONTAINER.log" 2>/dev/null)"
  docker inspect -f '{{.State.Running}}' "$ZIG_CONTAINER" 2>/dev/null | grep -q true || {
    docker logs --tail 30 "$ZIG_CONTAINER"; ./stop.sh; die "rank 0 exited"; }
  if (( i % 6 == 0 )); then
    ssh "$WORKER" "docker inspect -f '{{.State.Running}}' $ZIG_CONTAINER-follow 2>/dev/null" </dev/null | grep -q true || {
      ssh "$WORKER" "docker logs --tail 30 $ZIG_CONTAINER-follow" </dev/null; ./stop.sh; die "rank 1 exited"; }
  fi
  sleep 5
done
die "not up after 20 min: docker logs $ZIG_CONTAINER"
