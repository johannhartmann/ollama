#!/usr/bin/env bash
#
# Verify Ollama's native ColBERT path (/api/multivectors -> llama-server
# /colbert) against a direct llama.cpp run of the same model.
#
# Ollama returns projected, L2-normalized vectors for the ColBERT token plan.
# This script reproduces that pipeline independently: it takes the token plan
# Ollama reports (include_tokens; query mode retains every plan position),
# feeds the exact token ids to a raw llama-server /embeddings call (pooling
# none, full attention), applies the .colbert_proj sidecar projection and L2
# normalization in python, and compares the matrices. A pass means Ollama's
# plan, forward pass, projection and normalization are all faithful.
#
# Usage:
#   COLBERT_GGUF=/models/model.llama.gguf \
#   LLAMA_SERVER=/path/to/llama-server \
#   scripts/validate-multivectors-llamacpp-parity.sh
#
# Optional:
#   COLBERT_PROJ  projection sidecar (default: ${COLBERT_GGUF}.colbert_proj)
#   OLLAMA_HOST   ollama server (default http://127.0.0.1:11434)
#   OLLAMA_BIN    ollama binary for `create` (default: ./ollama, else `ollama`)
#   TEST_MODEL    temporary model name (default: multivector-parity-test)
#   LLAMA_PORT    port for the temporary llama-server (default: 8899)
#   PROMPT        text to encode (default: "Why is the sky blue?")
#   TOLERANCE     max abs diff (default: 1e-5)
#
# Requires: curl, jq, python3.

set -euo pipefail

: "${COLBERT_GGUF:?set COLBERT_GGUF to the path of an exported ColBERT/ModernColBERT GGUF}"
: "${LLAMA_SERVER:?set LLAMA_SERVER to the path of a llama-server binary}"
COLBERT_PROJ="${COLBERT_PROJ:-${COLBERT_GGUF}.colbert_proj}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
TEST_MODEL="${TEST_MODEL:-multivector-parity-test}"
LLAMA_PORT="${LLAMA_PORT:-8899}"
PROMPT="${PROMPT:-Why is the sky blue?}"
TOLERANCE="${TOLERANCE:-1e-5}"

[ -f "$COLBERT_GGUF" ] || { echo "error: COLBERT_GGUF '$COLBERT_GGUF' not found" >&2; exit 1; }
[ -f "$COLBERT_PROJ" ] || { echo "error: projection sidecar '$COLBERT_PROJ' not found" >&2; exit 1; }
[ -x "$LLAMA_SERVER" ] || { echo "error: LLAMA_SERVER '$LLAMA_SERVER' not found or not executable" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "error: curl is required" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "error: jq is required" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "error: python3 is required" >&2; exit 1; }

if [ -z "${OLLAMA_BIN:-}" ]; then
  if [ -x "./ollama" ]; then OLLAMA_BIN="./ollama"; else OLLAMA_BIN="ollama"; fi
fi

workdir="$(mktemp -d)"
llama_pid=""

cleanup() {
  [ -n "$llama_pid" ] && kill "$llama_pid" >/dev/null 2>&1 || true
  [ "${KEEP_TEST_MODEL:-0}" = "1" ] || "$OLLAMA_BIN" rm "$TEST_MODEL" >/dev/null 2>&1 || true
  rm -rf "$workdir"
}
trap cleanup EXIT

echo "==> starting llama-server on port $LLAMA_PORT"
"$LLAMA_SERVER" -m "$COLBERT_GGUF" --embedding --pooling none --ctx-size 512 \
  --host 127.0.0.1 --port "$LLAMA_PORT" >"$workdir/llama.log" 2>&1 &
llama_pid=$!

echo "    waiting for llama-server to become ready"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$llama_pid" >/dev/null 2>&1; then
    echo "error: llama-server exited early; see $workdir/llama.log" >&2
    cat "$workdir/llama.log" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "==> creating temporary Ollama model '$TEST_MODEL'"
# FROM auto-attaches the ${COLBERT_GGUF}.colbert_proj sidecar as the
# projection layer.
cat > "$workdir/Modelfile" <<EOF
FROM ${COLBERT_GGUF}
PARAMETER num_ctx 512
EOF
"$OLLAMA_BIN" create "$TEST_MODEL" -f "$workdir/Modelfile"

echo "==> encoding via Ollama /api/multivectors (query mode)"
jq -n --arg m "$TEST_MODEL" --arg p "$PROMPT" \
  '{model:$m, input:$p, input_type:"query", include_tokens:true}' \
  | curl -fsS -X POST "$OLLAMA_HOST/api/multivectors" -H 'Content-Type: application/json' -d @- \
  > "$workdir/ollama.json"

echo "==> running the same token plan through raw llama.cpp /embeddings"
jq '{content: .data[0].tokens}' "$workdir/ollama.json" \
  | curl -fsS -X POST "http://127.0.0.1:$LLAMA_PORT/embeddings" -H 'Content-Type: application/json' -d @- \
  > "$workdir/llamacpp.json"

echo "==> applying the projection sidecar and comparing"
TOLERANCE="$TOLERANCE" COLBERT_PROJ="$COLBERT_PROJ" \
  python3 - "$workdir/llamacpp.json" "$workdir/ollama.json" <<'PY'
import json, math, os, struct, sys

tol = float(os.environ["TOLERANCE"])

# load the projection sidecar: OLPROJ01, out/in/has_bias u32 LE, f32 weight
# row-major [out][in], optional f32 bias[out]
raw = open(os.environ["COLBERT_PROJ"], "rb").read()
if raw[:8] != b"OLPROJ01":
    print("FAIL: projection sidecar has an unknown format", file=sys.stderr)
    sys.exit(1)
out_f, in_f, has_bias = struct.unpack("<III", raw[8:20])
weight = struct.unpack(f"<{out_f * in_f}f", raw[20:20 + out_f * in_f * 4])
bias = [0.0] * out_f
if has_bias:
    off = 20 + out_f * in_f * 4
    bias = list(struct.unpack(f"<{out_f}f", raw[off:off + out_f * 4]))

llama = json.load(open(sys.argv[1]))
ollama = json.load(open(sys.argv[2]))

entry = next((e for e in llama if e.get("index", 0) == 0), llama[0])
rows = entry["embedding"]
if rows and not isinstance(rows[0], list):
    print("FAIL: llama.cpp returned a pooled vector; start it with --pooling none", file=sys.stderr)
    sys.exit(1)

orows = ollama["data"][0]["vectors"]
if len(rows) != len(orows):
    print(f"FAIL: row count mismatch (llama.cpp {len(rows)} vs ollama {len(orows)})", file=sys.stderr)
    sys.exit(1)

max_diff = 0.0
for raw_row, got in zip(rows, orows):
    if len(raw_row) != in_f or len(got) != out_f:
        print(f"FAIL: dim mismatch (raw {len(raw_row)} vs proj in {in_f}; "
              f"ollama {len(got)} vs proj out {out_f})", file=sys.stderr)
        sys.exit(1)
    proj = [bias[j] + sum(weight[j * in_f + k] * raw_row[k] for k in range(in_f))
            for j in range(out_f)]
    norm = math.sqrt(sum(v * v for v in proj))
    if norm <= 0 or not math.isfinite(norm):
        print("FAIL: zero or non-finite projected vector", file=sys.stderr)
        sys.exit(1)
    for j in range(out_f):
        max_diff = max(max_diff, abs(proj[j] / norm - got[j]))

print(f"rows={len(rows)} raw_dim={in_f} out_dim={out_f} max_abs_diff={max_diff:.6g}")
if max_diff > tol:
    print(f"FAIL: max abs diff {max_diff:.6g} exceeds tolerance {tol:.6g}", file=sys.stderr)
    sys.exit(1)

print("PASS: Ollama's ColBERT pipeline matches raw llama.cpp + sidecar projection")
PY
