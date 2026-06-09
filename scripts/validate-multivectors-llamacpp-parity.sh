#!/usr/bin/env bash
#
# Compare Ollama's /api/multivectors output to a direct llama.cpp server
# /embeddings call for the same ColBERT GGUF, to confirm Ollama preserves the
# token rows verbatim (no normalization, no cropping, no row dropping).
#
# Usage:
#   COLBERT_GGUF=/models/model.gguf \
#   LLAMA_SERVER=/path/to/llama-server \
#   scripts/validate-multivectors-llamacpp-parity.sh
#
# Optional:
#   OLLAMA_HOST   ollama server (default http://127.0.0.1:11434)
#   OLLAMA_BIN    ollama binary for `create` (default: ./ollama, else `ollama`)
#   TEST_MODEL    temporary model name (default: multivector-parity-test)
#   LLAMA_PORT    port for the temporary llama-server (default: 8899)
#   PROMPT        text to embed (default: "Why is the sky blue?")
#   TOLERANCE     max abs diff for sampled values (default: 1e-3)
#
# Requires: curl, jq, python3.

set -euo pipefail

: "${COLBERT_GGUF:?set COLBERT_GGUF to the path of a ColBERT/ModernColBERT GGUF}"
: "${LLAMA_SERVER:?set LLAMA_SERVER to the path of a llama-server binary}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
TEST_MODEL="${TEST_MODEL:-multivector-parity-test}"
LLAMA_PORT="${LLAMA_PORT:-8899}"
PROMPT="${PROMPT:-Why is the sky blue?}"
TOLERANCE="${TOLERANCE:-1e-3}"

[ -f "$COLBERT_GGUF" ] || { echo "error: COLBERT_GGUF '$COLBERT_GGUF' not found" >&2; exit 1; }
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
# --pooling none is required so the server emits one row per token. Some models
# also need non-causal attention; add it manually if your model requires it.
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
cat > "$workdir/Modelfile" <<EOF
FROM ${COLBERT_GGUF}
PARAMETER num_ctx 512
EOF
"$OLLAMA_BIN" create "$TEST_MODEL" -f "$workdir/Modelfile"

echo "==> requesting embeddings from both servers"
jq -n --arg p "$PROMPT" '{content:$p}' \
  | curl -fsS -X POST "http://127.0.0.1:$LLAMA_PORT/embeddings" -H 'Content-Type: application/json' -d @- \
  > "$workdir/llamacpp.json"

jq -n --arg m "$TEST_MODEL" --arg p "$PROMPT" '{model:$m, input:$p}' \
  | curl -fsS -X POST "$OLLAMA_HOST/api/multivectors" -H 'Content-Type: application/json' -d @- \
  > "$workdir/ollama.json"

echo "==> comparing"
TOLERANCE="$TOLERANCE" python3 - "$workdir/llamacpp.json" "$workdir/ollama.json" <<'PY'
import json, os, sys

tol = float(os.environ.get("TOLERANCE", "1e-3"))
llama = json.load(open(sys.argv[1]))
ollama = json.load(open(sys.argv[2]))

# llama.cpp non-OAI /embeddings: [{"index":0,"embedding":[[...],...]}]
entry = next((e for e in llama if e.get("index", 0) == 0), llama[0])
lrows = entry["embedding"]
if lrows and not isinstance(lrows[0], list):
    print("FAIL: llama.cpp returned a pooled vector; start it with --pooling none", file=sys.stderr)
    sys.exit(1)

orows = ollama["data"][0]["vectors"]

rows_l, rows_o = len(lrows), len(orows)
dim_l = len(lrows[0]) if lrows else 0
dim_o = len(orows[0]) if orows else 0

max_diff = 0.0
if rows_l == rows_o and dim_l == dim_o:
    for a, b in zip(lrows, orows):
        for x, y in zip(a, b):
            max_diff = max(max_diff, abs(x - y))

print(f"rows_llamacpp={rows_l} rows_ollama={rows_o} "
      f"dim_llamacpp={dim_l} dim_ollama={dim_o} max_abs_diff_sample={max_diff:.6g}")

if rows_l != rows_o:
    print(f"FAIL: row count mismatch ({rows_l} vs {rows_o}) -- rows were dropped", file=sys.stderr)
    sys.exit(1)
if dim_l != dim_o:
    print(f"FAIL: dimension mismatch ({dim_l} vs {dim_o})", file=sys.stderr)
    sys.exit(1)
if max_diff > tol:
    print(f"FAIL: max abs diff {max_diff:.6g} exceeds tolerance {tol:.6g}", file=sys.stderr)
    sys.exit(1)

print("PASS: Ollama multivector output matches llama.cpp within tolerance")
PY
