# CPU-only llama.cpp build that can load LiquidAI's LFM3 / LFM3-VL GGUFs.
#
# No published llama.cpp release carries LFM3 support yet (PR #31252 merged into master on
# 2026-10-27, after the b11935/b11936 tags were cut), so `src/models/lfm3.cpp` 404s at every
# release tag and the official images fail with "unknown model arch 'lfm3'". See UPSTREAM_REF
# for what is pinned and how to move to an official release once one exists.
#
# Two CPU variants are built from the same source, selected with --build-arg CPU_VARIANT:
#   avx2    SSE4.2+AVX+AVX2+FMA+F16C+BMI2 baseline. Runs on every x86 worker in the cluster.
#   avx512  adds AVX512F/VBMI/VNNI/BF16. Faster prefill, SIGILLs on a CPU without AVX-512, so
#           only deploy it on a node you have checked: lscpu | grep -o avx512f
# Deliberately NOT -march=native (GGML_NATIVE=OFF): the runner's CPU is not the cluster's CPU,
# and ggml puts these feature flags on ARCH_FLAGS for the whole target rather than per-kernel
# with runtime dispatch (ggml/src/ggml-cpu/CMakeLists.txt), so a native build would produce an
# image that only boots on the runner.
ARG VERSION=dev

FROM debian:bookworm-slim AS build
ARG UPSTREAM=https://github.com/ggml-org/llama.cpp.git
ARG UPSTREAM_REF
ARG CPU_VARIANT=avx2

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates ccache libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Fetch one commit by SHA so the image is reproducible from UPSTREAM_REF alone. This form also
# accepts a pull-request ref, e.g. refs/pull/31252/head.
WORKDIR /usr/local/src
RUN git init repo \
    && git -C repo remote add origin "$UPSTREAM" \
    && git -C repo fetch --depth 1 origin "$UPSTREAM_REF" \
    && git -C repo checkout --detach FETCH_HEAD \
    && git -C repo rev-parse HEAD > /llama_cpp_commit \
    && test -f repo/src/models/lfm3.cpp \
       || { echo "FATAL: $UPSTREAM_REF has no src/models/lfm3.cpp - LFM3 text would not load"; exit 1; } \
    && test -f repo/tools/mtmd/models/lfm3vl.cpp \
       || { echo "FATAL: $UPSTREAM_REF has no tools/mtmd/models/lfm3vl.cpp - the LFM3-VL vision projector would not load"; exit 1; }

# LFM3 is a hybrid: recurrent (shortconv) layers plus a mix of full and sliding-window
# attention (src/models/lfm3.cpp reads ATTENTION_SLIDING_WINDOW_PATTERN and refuses a model
# without SWA layers). Two consequences for a CPU build:
#   * GGML_OPENMP stays ON - the conv/recurrent ops fan out over the hidden dimension, and the
#     OpenMP path is what uses the extra cores.
#   * no GPU backend at all (CUDA/Vulkan/SYCL/Metal are all default-OFF here): a ggml build
#     that can see a device it does not have is how a CPU pod ends up failing at load time.
RUN set -eux; \
    case "$CPU_VARIANT" in \
      avx2) \
        FEAT="-DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON" ;; \
      avx512) \
        FEAT="-DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
              -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON" ;; \
      *) echo "FATAL: CPU_VARIANT must be avx2 or avx512, got '$CPU_VARIANT'"; exit 1 ;; \
    esac; \
    cmake -S repo -B build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_NATIVE=OFF \
      -DGGML_OPENMP=ON \
      -DGGML_LLAMAFILE=ON \
      -DLLAMA_OPENSSL=ON \
      -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_BUILD_TOOLS=ON \
      -DLLAMA_BUILD_UI=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      $FEAT \
      -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" ; \
    cmake --build build --config Release --parallel "$(nproc)"

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 libssl3 ca-certificates curl python3 \
    && rm -rf /var/lib/apt/lists/*
# llama-server needs mtmd (the vision projector library) linked in - it is built with the
# tools and statically linked, so the runtime stage carries no extra libraries.
COPY --from=build /usr/local/src/build/bin/llama-server    /usr/local/bin/llama-server
COPY --from=build /usr/local/src/build/bin/llama-cli       /usr/local/bin/llama-cli
COPY --from=build /usr/local/src/build/bin/llama-bench     /usr/local/bin/llama-bench
COPY --from=build /usr/local/src/build/bin/llama-mtmd-cli  /usr/local/bin/llama-mtmd-cli
COPY --from=build /llama_cpp_commit                        /opt/llama_cpp_commit

# Proof the two things this image exists for are really in the binary: the arch table knows
# 'lfm3' and the mtmd registry knows the LFM3-VL vision tower. Without the second one a
# --mmproj request fails at request time rather than at startup, which is much worse.
RUN set -eux; \
    llama-server --version | head -5; \
    grep -qa 'lfm3'   /usr/local/bin/llama-server; \
    grep -qa 'lfm3vl' /usr/local/bin/llama-server; \
    echo "built from llama.cpp $(cat /opt/llama_cpp_commit)"

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/llama-server"]
