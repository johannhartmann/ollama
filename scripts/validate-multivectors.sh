#!/usr/bin/env bash
#
# Developer validation for the /api/multivectors endpoint against a real
# ColBERT / ModernColBERT GGUF model. The model is NOT committed to the repo;
# point COLBERT_GGUF at a local file.
#
# Usage:
#   COLBERT_GGUF=/models/SauerkrautLM-Multi-ModernColBERT.gguf scripts/validate-multivectors.sh
#
# Optional:
#   OLLAMA_HOST     ollama server to talk to (default http://127.0.0.1:11434)
#   OLLAMA_BIN      ollama binary to use for `create` (default: ./ollama, else `ollama`)
#   TEST_MODEL      temporary model name (default: multivector-validate-test)
#   KEEP_TEST_MODEL set to 1 to keep the temporary model on exit
#
# Requires: curl, jq.

set -euo pipefail

: "${COLBERT_GGUF:?set COLBERT_GGUF to the path of a ColBERT/ModernColBERT GGUF}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
TEST_MODEL="${TEST_MODEL:-multivector-validate-test}"

if [ ! -f "$COLBERT_GGUF" ]; then
  echo "error: COLBERT_GGUF '$COLBERT_GGUF' not found" >&2
  exit 1
fi

command -v curl >/dev/null 2>&1 || { echo "error: curl is required" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "error: jq is required" >&2; exit 1; }

if [ -z "${OLLAMA_BIN:-}" ]; then
  if [ -x "./ollama" ]; then
    OLLAMA_BIN="./ollama"
  else
    OLLAMA_BIN="ollama"
  fi
fi

api() {
  # api <path> <json-body>
  curl -fsS -X POST "$OLLAMA_HOST$1" -H 'Content-Type: application/json' -d "$2"
}

cleanup() {
  if [ "${KEEP_TEST_MODEL:-0}" != "1" ]; then
    "$OLLAMA_BIN" rm "$TEST_MODEL" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "==> creating temporary model '$TEST_MODEL' from $COLBERT_GGUF"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"; cleanup' EXIT
cat > "$workdir/Modelfile" <<EOF
FROM ${COLBERT_GGUF}
PARAMETER num_ctx 512
EOF
"$OLLAMA_BIN" create "$TEST_MODEL" -f "$workdir/Modelfile"

echo "==> /api/show capabilities"
show="$(api /api/show "$(jq -n --arg m "$TEST_MODEL" '{model:$m}')")"
echo "$show" | jq '{capabilities, pooling: .model_info | to_entries | map(select(.key|endswith(".pooling_type"))) | .[0].value}'
echo "$show" | jq -e '.capabilities | index("multivector")' >/dev/null \
  || { echo "FAIL: model does not advertise the multivector capability" >&2; exit 1; }

echo "==> /api/multivectors (single input)"
single="$(api /api/multivectors "$(jq -n --arg m "$TEST_MODEL" '{model:$m, input:"Why is the sky blue?"}')")"
echo "$single" | jq '{embedding_type, pooling, dimension, rows: (.data[0].shape[0]), dim: (.data[0].shape[1])}'
echo "$single" | jq -e '.embedding_type == "multi_vector"' >/dev/null || { echo "FAIL: embedding_type" >&2; exit 1; }
echo "$single" | jq -e '.pooling == "none"' >/dev/null || { echo "FAIL: pooling" >&2; exit 1; }
echo "$single" | jq -e '.data[0].shape[0] > 0 and .data[0].shape[1] > 0' >/dev/null || { echo "FAIL: empty shape" >&2; exit 1; }

echo "==> /api/multivectors (two inputs)"
two="$(api /api/multivectors "$(jq -n --arg m "$TEST_MODEL" '{model:$m, input:["Berlin is the capital of Germany.","Paris is the capital of France."]}')")"
echo "$two" | jq -e '(.data | length) == 2' >/dev/null || { echo "FAIL: expected two data items" >&2; exit 1; }

echo "==> /api/multivectors (include_tokens)"
tokens="$(api /api/multivectors "$(jq -n --arg m "$TEST_MODEL" '{model:$m, input:"Why is the sky blue?", include_tokens:true}')")"
echo "$tokens" | jq -e '(.data[0].tokens | length) > 0' >/dev/null || { echo "FAIL: include_tokens returned no tokens" >&2; exit 1; }

echo "==> /api/multivectors (encoding_format=base64)"
b64="$(api /api/multivectors "$(jq -n --arg m "$TEST_MODEL" '{model:$m, input:"Why is the sky blue?", encoding_format:"base64"}')")"
echo "$b64" | jq -e '.data[0].data and (.data[0].data | length) > 0' >/dev/null || { echo "FAIL: base64 data missing" >&2; exit 1; }
echo "$b64" | jq -e '.data[0].vectors == null' >/dev/null || { echo "FAIL: base64 response should not include vectors" >&2; exit 1; }

echo "==> /api/embed should reject this model"
embed_code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$OLLAMA_HOST/api/embed" \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg m "$TEST_MODEL" '{model:$m, input:"Why is the sky blue?"}')")"
if [ "$embed_code" != "400" ]; then
  echo "FAIL: /api/embed returned $embed_code, expected 400" >&2
  exit 1
fi
embed_body="$(curl -s -X POST "$OLLAMA_HOST/api/embed" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg m "$TEST_MODEL" '{model:$m, input:"Why is the sky blue?"}')")"
echo "$embed_body" | grep -q "/api/multivectors" \
  || { echo "FAIL: /api/embed error should redirect to /api/multivectors" >&2; exit 1; }

echo "PASS: all multivector checks succeeded"
