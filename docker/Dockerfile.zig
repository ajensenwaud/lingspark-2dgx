# LingSpark build image for TensorFold's Zig engine: the pinned TensorFold GB10 CUDA base (BASE_IMAGE in
# scripts/config.sh) plus Zig 0.17.0, the version TensorFold pins in .zig-version. The engine is compiled and run in
# it; nothing is installed on the hosts. scripts/prepare.sh builds it when it is missing.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG ZIG_VERSION=0.17.0
# sha256 from https://ziglang.org/download/index.json, "0.17.0" -> "aarch64-linux"
ARG ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" \
 && echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 \
 && rm /tmp/zig.tar.xz && /opt/zig/zig version
ENV PATH=/opt/zig:${PATH} \
    ZIG_GLOBAL_CACHE_DIR=/work/.zig-cache-global
WORKDIR /work
ENTRYPOINT []
