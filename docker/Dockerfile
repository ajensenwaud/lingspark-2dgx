# LingSpark converter image (ling-exl3): the pinned TensorFold GB10 CUDA runtime (Mia's AI Lab's GLM-5.3 TensorFold
# image, by digest) plus upstream exllamav3 v1.6.6, which has Ling-3.1's architecture (bailing_hybrid and its
# MTP layer) natively. exllamav3 is the quantiser for our EXL3 pack and the reference forward the engine is checked
# against.
#
# Build with scripts/build-image.sh. Never pull the base: it is present locally on both nodes by digest.
ARG BASE_IMAGE=ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529
FROM ${BASE_IMAGE}

# exllamav3 at the pinned tag, CUDA extension compiled for GB10 (sm_121). --no-deps: the base's PyTorch is an NVIDIA
# build that pip must never replace.
ARG EXL3_REPO=https://github.com/turboderp-org/exllamav3
ARG EXL3_COMMIT=bd1f519172406a9560bd5c90e61000691bb69a07
ARG MAX_JOBS=12
RUN git clone -q "${EXL3_REPO}" /opt/exllamav3 && git -C /opt/exllamav3 checkout -q "${EXL3_COMMIT}" \
 && cd /opt/exllamav3 && TORCH_CUDA_ARCH_LIST="12.1" MAX_JOBS=${MAX_JOBS} \
    pip install --no-deps --no-build-isolation --no-cache-dir . \
 && pip install --no-deps --no-cache-dir marisa_trie llguidance \
 && git rev-parse HEAD > /opt/exllamav3.commit \
 && cd / && python3 -c "import exllamav3, exllamav3_ext; from importlib.metadata import version; \
from exllamav3.architecture.architectures import get_architectures as g; \
assert 'BailingMoeV3ForCausalLM' in g(), 'no Ling architecture'; print('exllamav3', version('exllamav3'), 'Ling registered')"

# Converter wrapper (clears sys.argv after parsing; see tools/exl3/convert.py)
COPY tools/exl3 /tools
ARG RECIPE_COMMIT=unknown
LABEL org.opencontainers.image.title="ling-exl3" org.opencontainers.image.revision="${RECIPE_COMMIT}" \
      io.exllamav3.revision="${EXL3_COMMIT}"
WORKDIR /workspace
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3"]
