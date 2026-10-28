# llama-cpp-vl

CPU-only `llama.cpp` image that can load **LiquidAI LFM3 / LFM3-VL** GGUFs — which no published
`llama.cpp` release can do yet.

Published at `ghcr.io/birdrock00/llama-cpp-vl`.

## Why this repo exists

`LiquidAI/LFM3-VL-2B-GGUF` is architecture `lfm3` (text) plus an `lfm3vl` vision projector
(`mmproj-*.gguf`). Support landed in [ggml-org/llama.cpp#31252], which was squash-merged into
`master` at `19cce691999` on 2026-10-27 — *after* the `b11935`/`b11936` release tags were cut:

```
$ curl -o /dev/null -w '%{http_code}\n' \
    https://raw.githubusercontent.com/ggml-org/llama.cpp/b11936/src/models/lfm3.cpp
404
```

So every official image (Docker Hub / GHCR `ghcr.io/ggml-org/llama.cpp:server-bXXXX`) fails on
the GGUF with `unknown model arch 'lfm3'`. This repo builds the pinned commit that carries the
support, and will keep doing so until an official release includes it — the daily scheduled run
probes the latest release tag and says so in its summary when the answer changes.

**Retiring this repo is a one-line change**: set `REF=` in [`UPSTREAM_REF`](UPSTREAM_REF) to the
first release tag that serves `src/models/lfm3.cpp`, or drop the deployment onto
`ghcr.io/ggml-org/llama.cpp:server-b<tag>` entirely (same `llama-server` CLI).

## What is in the image

`llama-server` (with `libmtmd` statically linked, so `--mmproj` works), plus `llama-cli`,
`llama-bench` and `llama-mtmd-cli` for benchmarking inside the pod. `ENTRYPOINT` is
`llama-server`, so the Deployment only supplies flags.

## Two CPU variants — this is the part that bites

`ggml` puts `-mavx2` / `-mavx512*` on the *whole* target (`ggml/src/ggml-cpu/CMakeLists.txt`),
not per-kernel behind runtime dispatch, and `GGML_NATIVE=OFF` leaves every feature flag **off**
by default. Consequences:

| Tag | Built with | Runs on |
|---|---|---|
| `avx2-latest`, `latest` | SSE4.2, AVX, AVX2, FMA, F16C, BMI2 | every x86-64 worker in this cluster |
| `avx512-latest` | + AVX512F/VBMI/VNNI/BF16 | only nodes where `lscpu \| grep avx512f` succeeds |
| `avx2-<sha>` / `avx512-<sha>` | same, pinned to the `llama.cpp` commit | same |

A `-march=native` build (i.e. leaving `GGML_NATIVE` on) is the trap: it would compile for the
*runner's* CPU and SIGILL on the cluster, and a plain `-DGGML_NATIVE=OFF` build with no feature
flags would quietly run the scalar kernels and look like the model is twice as slow as it is.

## Building

```bash
docker build \
  --build-arg UPSTREAM_REF="$(grep '^REF=' UPSTREAM_REF | cut -d= -f2)" \
  --build-arg CPU_VARIANT=avx2 \
  -t llama-cpp-vl:local .
```

CI (`.github/workflows/build-image.yml`) builds both variants, then a `smoke` job loads the
`QAD-Q4_0` quant plus the `Q8_0` projector in the published image, encodes a generated PNG and
requires a non-empty completion — the only step that catches a server built without the vision
tower, which otherwise fails at request time rather than at startup. It can also be run against
a PR ref: dispatch with `upstream_ref = refs/pull/31252/head`.

## Running an LFM3-VL model on CPU

```bash
llama-server \
  --model        /models/LFM3-VL-2B-Q4_K_M.gguf \
  --mmproj       /models/mmproj-LFM3-VL-2B-Q8_0.gguf \
  --alias        lfm3-vl \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 8192 --threads 8 \
  --temp 0.1 --top-p 1.0 --top-k 50 --min-p 0.0 --repeat-penalty 1.1 \
  --no-reasoning-preserve --jinja
```

* `--mmproj` is **required** for image input; without it the server starts fine and every image
  request 400s.
* `--jinja` makes the server apply the model's own chat template — LFM3's template is what
  implements thinking blocks and tool calls, so without it those tokens appear as literal text.
* Sampling flags are the values from the [model card](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF)
  (`--temp 0.1 --top-p 1.0 --top-k 50 --min-p 0.0 --repeat-penalty 1.1 --no-reasoning-preserve`).
* `--no-mmap` is worth testing when weights live on a network mount; on local disk, leave mmap on.
* Video input is **not** in the merged PR ("Video input support will be added later"), so a
  frame-by-frame caller has to send stills.

Deployed in this fleet as `llama-cpp-vision` (see the `rpi` repo:
`custom-deployments/llama-cpp-vision/`, playbook `148-install-llama-cpp-vision.yaml`).

[ggml-org/llama.cpp#31252]: https://github.com/ggml-org/llama.cpp/pull/31252
