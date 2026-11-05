#!/usr/bin/env bash
# Build the converter image ling-exl3 on this node: pinned TensorFold base + upstream exllamav3 @ EXL3_COMMIT.
# Tags ling-exl3:<first 10 of EXL3_COMMIT> and ling-exl3:latest.
#   scripts/build-image.sh                              # full speed (MAX_JOBS=12)
#   MAX_JOBS=2 BUILD_MEMORY=9g scripts/build-image.sh   # beside a running serve: legacy builder with a memory cap
# Copy to the other node:  docker save ling-exl3:<tag> | ssh <node> docker load   (then tag latest there)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
MAX_JOBS="${MAX_JOBS:-12}"
tag="${EXL3_COMMIT:0:10}"
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || die "base image $BASE_IMAGE not present locally (never pull it)"
args=(--build-arg "BASE_IMAGE=$BASE_IMAGE" --build-arg "EXL3_COMMIT=$EXL3_COMMIT" --build-arg "MAX_JOBS=$MAX_JOBS"
      --build-arg "RECIPE_COMMIT=$(git rev-parse --short=10 HEAD 2>/dev/null || echo unknown)"
      -f docker/Dockerfile -t "$IMAGE:$tag")
log "building $IMAGE:$tag (exllamav3 ${EXL3_COMMIT:0:10}, MAX_JOBS=$MAX_JOBS${BUILD_MEMORY:+, memory cap $BUILD_MEMORY})"
if [[ -n "${BUILD_MEMORY:-}" ]]; then
  DOCKER_BUILDKIT=0 docker build --memory "$BUILD_MEMORY" --memory-swap "$BUILD_MEMORY" "${args[@]}" .
else
  docker build "${args[@]}" .
fi
got=$(docker run --rm --entrypoint cat "$IMAGE:$tag" /opt/exllamav3.commit)
[[ "$got" == "$EXL3_COMMIT" ]] || die "image has exllamav3 $got, expected $EXL3_COMMIT"
docker tag "$IMAGE:$tag" "$IMAGE:latest"
log "done: $IMAGE:$tag (= latest), exllamav3 $got"
