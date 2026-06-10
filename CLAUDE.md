# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a fork of [ollama](https://github.com/ollama/ollama) that adds a **second, Go-native inference engine built on Apple's MLX** alongside the upstream llama.cpp/GGML engine, plus image generation, an Anthropic-compatible API, and a built-in coding agent. Most fork-specific code lives under `x/`, `ml/`, `anthropic/`, and `middleware/`.

`AGENTS.md` holds the canonical short build instructions. This file expands on architecture and the non-obvious pieces.

## Build & run

The CMake project at the repo root is an **orchestration superbuild**, not the thing that compiles inference kernels directly. A full build produces three artifacts:

1. The Go binary `./ollama` (built via `go build` from the root — see `cmake/local.cmake`).
2. `llama-server` + GGML backends, configured from `llama/server/` and pulled via FetchContent from llama.cpp pinned at `LLAMA_CPP_VERSION`.
3. MLX backends, pulled via FetchContent from MLX / MLX-C pinned at `MLX_VERSION` / `MLX_C_VERSION`.

```sh
# Full build from a fresh checkout or after changing native/CGO code
cmake -B build .
cmake --build build --parallel 8
./ollama serve

# Fast Go-only iteration against an already-built native payload
go build .          # or: go run . serve
```

Native payloads install under `build/lib/ollama`; the Go binary locates them there (and in several install layouts — see `docs/development.md` "Library detection").

- **CGO is in play.** Stale CGO data structures cause mysterious crashes after native changes — `go clean -cache` forces a full rebuild.
- **Backend selection** (non-macOS-arm64, which defaults to Metal MLX): `-DOLLAMA_LLAMA_BACKENDS="cuda_v13;vulkan"` and/or `-DOLLAMA_MLX_BACKENDS=cuda_v13`. Narrow GPU arches with `-DCMAKE_CUDA_ARCHITECTURES=native`, `-DCMAKE_HIP_ARCHITECTURES=gfx1100`, etc.
- **MLX on macOS arm64** requires the Xcode Metal toolchain: `xcodebuild -downloadComponent MetalToolchain`. macOS 26 SDK selects `metal_v4`, older selects `metal_v3`.
- **Local MLX/llama.cpp source override** (for debugging the native side): `OLLAMA_MLX_SOURCE`, `OLLAMA_MLX_C_SOURCE`, `OLLAMA_LLAMA_CPP_SOURCE` env vars, or the matching `FETCHCONTENT_SOURCE_DIR_*` CMake vars.

## Test & lint

```sh
go test ./...                                   # all Go tests
go test -count=1 -benchtime=1x ./...            # how CI runs them (no cache, benches once)
go test ./x/mlxrunner/ -run TestName            # a single test
go test -count=1 -tags updater_live ./app/...   # app tests with the live-updater build tag

golangci-lint run                               # config in .golangci.yaml (v2; gofumpt + gofmt enforced)
go generate ./...                               # regenerates CGO wrappers, typescript structs, etc.
go mod tidy                                      # CI fails on `go mod tidy --diff`
```

Regenerating the MLX Go wrappers (the CGO bindings over MLX-C) is a CMake target, not plain `go generate`: it needs the fetched MLX-C headers first.
```sh
cmake -S . -B build/mlx-generate -DOLLAMA_MLX_BACKENDS=<backend>
cmake --build build/mlx-generate --target ollama-mlx-generate-wrappers
```

## Architecture: two inference engines behind one scheduler

The scheduler `server/sched.go` is where a request is routed to one of the engines. The fork branches on **model format**, exposed as `Model.IsMLX()` (`server/images.go`), which is true when `Config.ModelFormat == "safetensors"`:

- **GGUF models** → `llm.NewLlamaServer` (`llm/server.go`, `llm/llama_server.go`) spawns and supervises the upstream `llama-server` binary. Memory placement, context, batch, and mmap heuristics all live in `sched.go` and only apply to this path.
- **safetensors / MLX models** → `mlxrunner.NewClient` (`x/mlxrunner`), the Go-native MLX engine.
- **image-capable MLX models** → `imagegen.NewServer` (`x/imagegen`).

All three implement the same `llm.LlamaServer` interface so the rest of the scheduler treats them uniformly.

### Runner subprocesses

The server runs inference engines as **separate processes of the same binary**. `cmd/cmd.go` registers a hidden `runner` command; `runner/runner.go` dispatches `--mlx-engine` → `mlxrunner.Execute` and `--imagegen-engine` → `imagegen.Execute`. Each runner is an HTTP server the parent talks to over a local socket. `cmd/runner/main.go` is a standalone entry to the same code.

### The MLX engine (`x/`)

- `x/mlxrunner/` — the runner: HTTP `server.go`, request loop `runner.go`, KV/trie/prompt cache (`cache.go`, `cache_trie.go`), **MTP (multi-token-prediction) speculative decoding** (`mtp.go`, `pipeline.go`), `batch/`, `sample/`, and the CGO MLX bindings in `mlx/`.
- `x/models/` — pure-Go model definitions (gemma3, gemma4, glm4_moe_lite, laguna, llama, qwen3, qwen3_5, qwen3_5_moe). Each registers itself; they are wired in via blank imports in `x/mlxrunner/imports.go`. **Adding a model means adding a package here and a blank import there.**
- `ml/` and `ml/nn/` — shared Go ML primitives the models build on: the `Backend`/`Device` abstraction (`ml/backend.go`, `ml/device.go`) and layers (`ml/nn/`: attention, rope, normalization, linear, embedding, convolution, pooling).
- `x/create/` — builds/quantizes safetensors models into the local store (the MLX equivalent of GGUF conversion). `x/transfer/` — registry pull/push with sparse-file support. `x/safetensors/`, `x/tokenizer/` — formats. `x/server/show.go` — `show` for MLX models.

### MLX threading invariant (`x/internal/mlxthread`)

MLX/Metal calls must execute on **one fixed, locked OS thread**. `mlxthread.Thread` marshals every MLX operation onto a single goroutine pinned with `runtime.LockOSThread`. Anything touching MLX (model load, forward pass, cache ops) must go through `Thread.Do` — do not call MLX from arbitrary goroutines, or you get crashes/corruption. This constraint shapes the runner's request loop.

### Anthropic-compatible API & coding agent

- `anthropic/` + `middleware/anthropic.go` translate the Anthropic Messages API to ollama's chat API. The `/v1/messages` route (`server/routes.go`) is wired through `middleware.AnthropicMessagesMiddleware()` → `ChatHandler`, so Anthropic-protocol clients (Claude Code, etc.) can target a local model. OpenAI compatibility lives in `openai/` + `middleware/`.
- `x/cmd/run.go`, `x/agent/`, `x/tools/` — a built-in terminal coding agent: an agent loop with tool-use (`bash`, `webfetch`, `websearch` in `x/tools/`) gated by an interactive approval system (`x/agent/approval*.go`, per-OS).

## Conventions

- This fork keeps the upstream module path `github.com/ollama/ollama` and the upstream package layout (`api/`, `server/`, `llm/`, `cmd/`, `convert/`, `template/`, etc.). New work is deliberately siloed under `x/` to minimize merge conflicts with upstream — prefer adding there over editing core packages when feasible.
- Go 1.26. Formatting is **gofumpt** (stricter than gofmt) — both run in CI. `errcheck` and `usestdlibvars` are intentionally disabled in `.golangci.yaml`; `SA1019` (deprecation) is suppressed.
- Commit messages in this repo follow lowercase `area: imperative summary` (e.g. `mlxrunner: record committed MTP drafts before streaming them`).
