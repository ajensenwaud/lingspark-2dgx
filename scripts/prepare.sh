#!/usr/bin/env bash
# Build the Zig engine: TensorFold at TF_COMMIT (v1.2.2) plus patches-zig/, compiled in a build container (nothing is
# installed on the hosts), then copied to the worker over the CX7 link. Run on the head (rank 0); CPU only (the build
# container is capped at ZIG_BUILD_MEM and ZIG_BUILD_JOBS).
#   scripts/prepare.sh               clone/patch, check the tree, build, SHA256SUMS, copy zig-out/ to $WORKER
#   NO_WORKER=1 scripts/prepare.sh   the same without the copy
set -euo pipefail
cd "$(dirname "$0")/.."
RECIPE=$(pwd)
source scripts/config.sh
[[ -n "${NO_WORKER:-}" ]] || need_hosts WORKER

# 1. the source: TensorFold at the pinned commit, the patches on a fresh branch, the tree hash checked
if [[ ! -d "$ZIG_DIR/.git" ]]; then
  log "cloning TensorFold into $ZIG_DIR"
  mkdir -p "$(dirname "$ZIG_DIR")"
  git clone -q "$TF_REPO" "$ZIG_DIR"
fi
cd "$ZIG_DIR"
[[ -d .git/rebase-apply ]] && git am --abort
if ! git cat-file -e "$TF_COMMIT^{commit}" 2>/dev/null; then
  git fetch -q origin --tags || die "could not fetch $TF_REPO"
  git cat-file -e "$TF_COMMIT^{commit}" 2>/dev/null || die "$TF_REPO has no commit $TF_COMMIT ($TF_TAG)"
fi
git checkout -q -f -B ling-zig "$TF_COMMIT"
log "applying $(ls "$RECIPE"/patches-zig/*.patch | wc -l) patches to $TF_TAG (${TF_COMMIT:0:12})"
GIT_COMMITTER_NAME=recipe GIT_COMMITTER_EMAIL=recipe@localhost git am -q --3way "$RECIPE"/patches-zig/*.patch \
  || { git am --abort; die "patches-zig/ did not apply"; }
tree=$(git rev-parse HEAD^{tree})
[[ "$tree" == "$ZIG_TREE" ]] || die "patched tree is $tree, config.sh pins $ZIG_TREE"
log "tree $tree"

# 2. the build image (docker/Dockerfile.zig: Zig 0.17.0 on the pinned TensorFold CUDA base, pulled by digest once),
#    then compile in it: the server with the ling family for GB10 (sm_121)
if ! docker image inspect "$ZIG_BUILD_IMAGE" >/dev/null 2>&1; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || { log "pulling the base image $BASE_IMAGE"; docker pull -q "$BASE_IMAGE" >/dev/null; }
  log "building $ZIG_BUILD_IMAGE on the base image"
  docker build -q --pull=false -t "$ZIG_BUILD_IMAGE" --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -f "$RECIPE/docker/Dockerfile.zig" "$RECIPE/docker" >/dev/null || die "could not build $ZIG_BUILD_IMAGE"
fi
log "compiling ($ZIG_BUILD_JOBS jobs, $ZIG_BUILD_MEM; ~10 min from a clean tree)"
rm -rf "$ZIG_DIR/zig-out"
docker run --rm --name ling-zig-build --user "$(id -u):$(id -g)" -e HOME=/tmp -e ZIG_GLOBAL_CACHE_DIR=/work/.zig-global \
  --memory="$ZIG_BUILD_MEM" --memory-swap="$ZIG_BUILD_MEM" --cpus="$ZIG_BUILD_JOBS" \
  -v "$ZIG_DIR:/work" -w /work "$ZIG_BUILD_IMAGE" bash -c "
    set -e
    NVCC=\$(which nvcc)
    zig build install native -j$ZIG_BUILD_JOBS -Dnvcc=\$NVCC -Dtp-nvcc=\$NVCC -Dsm=121 --prefix /work/zig-out --summary failures"
for f in bin/tensorfold bin/tf-cuda-test native/bin/tensorfold-native; do
  [[ -x "$ZIG_DIR/zig-out/$f" ]] || die "build incomplete: no $f"
done
(cd "$ZIG_DIR/zig-out" && sha256sum bin/* native/bin/* > SHA256SUMS)

# 3. the same build on the worker, checked against the head's sums
if [[ -z "${NO_WORKER:-}" ]]; then
  log "copying the build to $WORKER:$ZIG_DIR/zig-out"
  if ! ssh "$WORKER" "docker image inspect '$ZIG_RUN_IMAGE' >/dev/null 2>&1" </dev/null; then
    log "copying the image $ZIG_RUN_IMAGE to $WORKER (one time)"
    docker save "$ZIG_RUN_IMAGE" | ssh "$WORKER" docker load >/dev/null || die "could not copy $ZIG_RUN_IMAGE to $WORKER"
  fi
  ssh "$WORKER" "mkdir -p $ZIG_DIR" </dev/null
  rsync -a --delete "$ZIG_DIR/zig-out/" "$WORKER:$ZIG_DIR/zig-out/"
  ssh "$WORKER" "cd $ZIG_DIR/zig-out && sha256sum --quiet -c SHA256SUMS" </dev/null || die "the worker's copy fails SHA256SUMS"
fi
log "built $(git rev-parse --short HEAD) (tree ${tree:0:12}) in $ZIG_DIR/zig-out; tensorfold-native sha256 $(grep native/bin zig-out/SHA256SUMS | cut -c1-16)"
