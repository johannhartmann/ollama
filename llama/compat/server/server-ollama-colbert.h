// Ollama ColBERT (late-interaction) encoding support for llama-server.
//
// Implements the model-specific half of the POST /colbert endpoint: the
// runtime profile (parsed from the `pg_colbert.profile_json` GGUF metadata
// key), the projection sidecar (path supplied via the
// OLLAMA_COLBERT_PROJECTION environment variable), the ColBERT token plan
// (prefixing, truncation, query [MASK] expansion, document skiplist
// filtering), and the per-token post-processing (dense projection +
// L2 normalization).
//
// The HTTP half — task submission and response plumbing — lives in
// server_routes::handle_colbert_impl, added to tools/server/server-context.cpp
// by llama/compat/llama-cpp-server-colbert.patch, because it needs the
// translation-unit-private server_context_impl / server_res_generator types.
//
// The token plan mirrors pg_colbert's llama engine
// (PgColbertBuildEncodePlan / PgColbertShouldRetainToken): queries retain
// every position including [MASK] expansion vectors; documents drop padding,
// skiplist and single-character punctuation positions. Query encoding shares
// the same documented approximation as pg_colbert: content tokens attend to
// the [MASK] expansion tokens (llama.cpp computes full non-causal attention),
// so strict PyLate profiles with attend_to_expansion_tokens=false are
// approximated, not bit-exact. Document/content-token parity is exact.

#pragma once

#include "llama.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace ollama_colbert {

struct encode_plan {
    std::vector<llama_token> tokens;      // full planned sequence fed to the model
    std::vector<bool>        output_mask; // rows to keep after the forward pass
    int                      output_count = 0;
    bool                     is_query     = false;
};

// Idempotent, thread-safe. Parses the ColBERT profile from the model
// metadata and loads the projection sidecar named by the
// OLLAMA_COLBERT_PROJECTION environment variable. Returns false with a
// description when the model does not carry a usable ColBERT profile or the
// sidecar is missing/invalid.
bool ensure_init(const llama_model * model, std::string & err);

// Parses {"input": string | [string, ...], "input_type": "query"|"document"}.
// input_type defaults to "document".
bool parse_request(const nlohmann::ordered_json & body,
                   std::vector<std::string> & inputs,
                   bool & is_query,
                   std::string & err);

// Builds the ColBERT token plan for one input. Requires ensure_init().
bool build_plan(const llama_vocab * vocab,
                const std::string & input,
                bool is_query,
                encode_plan & plan,
                std::string & err);

// Projects + L2-normalizes the raw per-token embeddings for one plan and
// encodes the response item: {"tokens": [...], "embedding": [[...], ...]}.
// rows must hold one raw embedding (model n_embd floats) per planned token.
bool encode_response_item(const encode_plan & plan,
                          const std::vector<std::vector<float>> & rows,
                          nlohmann::ordered_json & item,
                          std::string & err);

// Final vector dimensionality (projection output, or the model embedding
// width when the profile declares an identity projection).
int output_dim();

} // namespace ollama_colbert
