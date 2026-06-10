#include "server-ollama-colbert.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>

using ordered_json = nlohmann::ordered_json;

namespace ollama_colbert {

namespace {

constexpr char PROJ_SIDECAR_MAGIC[8] = {'O', 'L', 'P', 'R', 'O', 'J', '0', '1'};
constexpr const char * PROFILE_KEY   = "pg_colbert.profile_json";
constexpr const char * PROJECTION_ENV = "OLLAMA_COLBERT_PROJECTION";

struct projection {
    int32_t            out_features = 0;
    int32_t            in_features  = 0;
    std::vector<float> weight; // row-major [out][in]
    std::vector<float> bias;   // empty when the sidecar has no bias
};

struct profile {
    std::string query_prefix;
    std::string document_prefix;
    int32_t     query_max_length    = 0;  // required
    int32_t     document_max_length = 0;  // required
    int32_t     query_pad_to        = -1; // default: query_max_length
    int32_t     query_pad_token_id  = -1; // default: mask_token_id, then vocab mask
    int32_t     mask_token_id       = -1;
    int32_t     pad_token_id        = -1;
    std::vector<int32_t> skiplist;
    int32_t     declared_output_dim = 0;
    std::string projection_kind     = "dense";
};

struct state {
    bool        initialized = false;
    bool        ok          = false;
    std::string err;
    profile     prof;
    bool        has_projection = false;
    projection  proj;
    int32_t     n_embd  = 0; // model embedding width (projection input)
    int32_t     out_dim = 0; // final vector width
};

std::mutex g_mutex;
state      g_state;

const ordered_json * find_member(const ordered_json & obj, const char * key) {
    if (!obj.is_object()) {
        return nullptr;
    }
    auto it = obj.find(key);
    if (it == obj.end() || it->is_null()) {
        return nullptr;
    }
    return &*it;
}

int32_t int_field(const ordered_json & obj, const char * key, int32_t fallback) {
    const ordered_json * v = find_member(obj, key);
    if (v != nullptr && v->is_number_integer()) {
        return v->get<int32_t>();
    }
    return fallback;
}

std::string string_field(const ordered_json & obj, const char * key, const std::string & fallback) {
    const ordered_json * v = find_member(obj, key);
    if (v != nullptr && v->is_string()) {
        return v->get<std::string>();
    }
    return fallback;
}

bool read_profile_json(const llama_model * model, std::string & raw, std::string & err) {
    int32_t needed = llama_model_meta_val_str(model, PROFILE_KEY, nullptr, 0);
    if (needed < 0) {
        err = "model metadata does not contain " + std::string(PROFILE_KEY) +
              "; this GGUF was not exported as a ColBERT model";
        return false;
    }
    std::vector<char> buf(needed + 1, '\0');
    llama_model_meta_val_str(model, PROFILE_KEY, buf.data(), buf.size());
    raw.assign(buf.data());
    return true;
}

bool parse_profile(const std::string & raw, profile & p, std::string & err) {
    ordered_json doc = ordered_json::parse(raw, nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) {
        err = std::string(PROFILE_KEY) + " is not valid JSON";
        return false;
    }

    p.declared_output_dim = int_field(doc, "output_dim", 0);

    if (const ordered_json * tok = find_member(doc, "tokenizer")) {
        if (const ordered_json * st = find_member(*tok, "special_tokens")) {
            p.mask_token_id = int_field(*st, "mask_token_id", -1);
            p.pad_token_id  = int_field(*st, "pad_token_id", -1);
        }
    }

    const ordered_json * query = find_member(doc, "query");
    if (query == nullptr) {
        err = std::string(PROFILE_KEY) + " is missing the \"query\" section";
        return false;
    }
    p.query_prefix       = string_field(*query, "prefix", "");
    p.query_max_length   = int_field(*query, "max_length", 0);
    p.query_pad_to       = int_field(*query, "pad_to", -1);
    p.query_pad_token_id = int_field(*query, "pad_token_id", -1);
    if (p.query_max_length <= 0) {
        err = std::string(PROFILE_KEY) + " query.max_length must be a positive integer";
        return false;
    }

    const ordered_json * document = find_member(doc, "document");
    if (document == nullptr) {
        err = std::string(PROFILE_KEY) + " is missing the \"document\" section";
        return false;
    }
    p.document_prefix     = string_field(*document, "prefix", "");
    p.document_max_length = int_field(*document, "max_length", 0);
    if (p.document_max_length <= 0) {
        err = std::string(PROFILE_KEY) + " document.max_length must be a positive integer";
        return false;
    }
    if (const ordered_json * skip = find_member(*document, "skiplist_token_ids")) {
        if (skip->is_array()) {
            for (const auto & v : *skip) {
                if (v.is_number_integer()) {
                    p.skiplist.push_back(v.get<int32_t>());
                }
            }
        }
    }

    if (const ordered_json * projection = find_member(doc, "projection")) {
        p.projection_kind = string_field(*projection, "kind", "dense");
    }
    return true;
}

bool load_sidecar(const char * path, projection & proj, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = std::string("cannot open ColBERT projection sidecar: ") + path;
        return false;
    }

    char magic[8];
    uint32_t header[3]; // out_features, in_features, has_bias
    if (!f.read(magic, sizeof(magic)) ||
        !f.read(reinterpret_cast<char *>(header), sizeof(header))) {
        err = std::string("ColBERT projection sidecar is truncated: ") + path;
        return false;
    }
    if (std::memcmp(magic, PROJ_SIDECAR_MAGIC, sizeof(magic)) != 0) {
        err = std::string("ColBERT projection sidecar has an unknown format: ") + path;
        return false;
    }

    const uint32_t out_features = header[0];
    const uint32_t in_features  = header[1];
    const uint32_t has_bias     = header[2];
    if (out_features == 0 || in_features == 0 || out_features > 1 << 16 || in_features > 1 << 16) {
        err = std::string("ColBERT projection sidecar declares implausible dimensions: ") + path;
        return false;
    }

    proj.out_features = (int32_t) out_features;
    proj.in_features  = (int32_t) in_features;
    proj.weight.resize((size_t) out_features * in_features);
    if (!f.read(reinterpret_cast<char *>(proj.weight.data()),
                (std::streamsize) (proj.weight.size() * sizeof(float)))) {
        err = std::string("ColBERT projection sidecar weight data is truncated: ") + path;
        return false;
    }
    if (has_bias != 0) {
        proj.bias.resize(out_features);
        if (!f.read(reinterpret_cast<char *>(proj.bias.data()),
                    (std::streamsize) (proj.bias.size() * sizeof(float)))) {
            err = std::string("ColBERT projection sidecar bias data is truncated: ") + path;
            return false;
        }
    }
    // exactly at EOF?
    f.peek();
    if (!f.eof()) {
        err = std::string("ColBERT projection sidecar has trailing data: ") + path;
        return false;
    }
    return true;
}

bool init_locked(const llama_model * model, std::string & err) {
    state & s = g_state;
    s.n_embd = llama_model_n_embd_out(model);

    std::string raw;
    if (!read_profile_json(model, raw, err)) {
        return false;
    }
    if (!parse_profile(raw, s.prof, err)) {
        return false;
    }

    const char * sidecar = std::getenv(PROJECTION_ENV);
    if (sidecar != nullptr && sidecar[0] != '\0') {
        if (!load_sidecar(sidecar, s.proj, err)) {
            return false;
        }
        if (s.proj.in_features != s.n_embd) {
            err = "ColBERT projection expects input dim " + std::to_string(s.proj.in_features) +
                  " but the model produces " + std::to_string(s.n_embd);
            return false;
        }
        s.has_projection = true;
        s.out_dim = s.proj.out_features;
    } else if (s.prof.projection_kind == "identity") {
        s.has_projection = false;
        s.out_dim = s.n_embd;
    } else {
        err = "model declares a \"" + s.prof.projection_kind +
              "\" ColBERT projection but no sidecar was provided (set " +
              PROJECTION_ENV + " to the .colbert_proj file)";
        return false;
    }

    if (s.prof.declared_output_dim > 0 && s.prof.declared_output_dim != s.out_dim) {
        err = "ColBERT profile declares output_dim " + std::to_string(s.prof.declared_output_dim) +
              " but the serving pipeline produces " + std::to_string(s.out_dim);
        return false;
    }
    return true;
}

// llama.cpp's BERT WPM tokenizer prefixes word-start pieces with U+2581;
// strip it before the single-character punctuation check (mirrors
// pg_colbert's PgColbertIsPunctuationToken).
bool is_punctuation_token(const llama_vocab * vocab, llama_token token) {
    const char * piece = llama_vocab_get_text(vocab, token);
    if (piece == nullptr || piece[0] == '\0') {
        return false;
    }
    if ((unsigned char) piece[0] == 0xe2 &&
        (unsigned char) piece[1] == 0x96 &&
        (unsigned char) piece[2] == 0x81) {
        piece += 3;
    }
    return piece[0] != '\0' && piece[1] == '\0' &&
           std::strchr("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", piece[0]) != nullptr;
}

// Document retention (mirrors pg_colbert's PgColbertShouldRetainToken):
// drop padding, profile skiplist entries and single-character punctuation.
// Queries retain every position, including the [MASK] expansion vectors.
bool should_retain_document_token(const llama_vocab * vocab, const profile & p, llama_token token) {
    if (token == LLAMA_TOKEN_NULL) {
        return false;
    }
    const llama_token vocab_pad = llama_vocab_pad(vocab);
    if (vocab_pad != LLAMA_TOKEN_NULL && token == vocab_pad) {
        return false;
    }
    if (p.pad_token_id >= 0 && token == (llama_token) p.pad_token_id) {
        return false;
    }
    for (int32_t id : p.skiplist) {
        if ((llama_token) id == token) {
            return false;
        }
    }
    return !is_punctuation_token(vocab, token);
}

} // namespace

bool ensure_init(const llama_model * model, std::string & err) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_state.initialized) {
        g_state.ok = init_locked(model, g_state.err);
        g_state.initialized = true;
    }
    if (g_state.ok) {
        err.clear();
    } else {
        err = g_state.err;
    }
    return g_state.ok;
}

bool parse_request(const ordered_json & body,
                   std::vector<std::string> & inputs,
                   bool & is_query,
                   std::string & err) {
    const ordered_json * input = find_member(body, "input");
    if (input == nullptr) {
        err = "\"input\" must be provided";
        return false;
    }
    if (input->is_string()) {
        inputs.push_back(input->get<std::string>());
    } else if (input->is_array()) {
        for (const auto & v : *input) {
            if (!v.is_string()) {
                err = "\"input\" must be a string or an array of strings";
                return false;
            }
            inputs.push_back(v.get<std::string>());
        }
    } else {
        err = "\"input\" must be a string or an array of strings";
        return false;
    }

    const std::string input_type = string_field(body, "input_type", "document");
    if (input_type == "query") {
        is_query = true;
    } else if (input_type == "document" || input_type.empty()) {
        is_query = false;
    } else {
        err = "\"input_type\" must be \"query\" or \"document\"";
        return false;
    }
    return true;
}

bool build_plan(const llama_vocab * vocab,
                const std::string & input,
                bool is_query,
                encode_plan & plan,
                std::string & err) {
    const profile & p = g_state.prof;

    plan = encode_plan{};
    plan.is_query = is_query;

    const std::string full = (is_query ? p.query_prefix : p.document_prefix) + input;

    int32_t required = llama_tokenize(vocab, full.c_str(), (int32_t) full.size(),
                                      nullptr, 0, /*add_special*/ true, /*parse_special*/ true);
    if (required == INT32_MIN) {
        err = "tokenization failed";
        return false;
    }
    if (required < 0) {
        required = -required;
    }
    if (required <= 0) {
        err = "input produced no tokens";
        return false;
    }

    const int32_t max_plan = is_query ? p.query_max_length : p.document_max_length;
    int32_t pad_to = -1;
    if (is_query) {
        pad_to = p.query_pad_to > 0 ? p.query_pad_to : p.query_max_length;
    }

    int32_t capacity = required;
    if (pad_to > capacity) {
        capacity = pad_to;
    }
    plan.tokens.resize(capacity);
    int32_t n = llama_tokenize(vocab, full.c_str(), (int32_t) full.size(),
                               plan.tokens.data(), capacity, true, true);
    if (n <= 0) {
        err = "tokenization failed";
        return false;
    }
    plan.tokens.resize(n);
    if (max_plan > 0 && (int32_t) plan.tokens.size() > max_plan) {
        plan.tokens.resize(max_plan);
    }

    if (is_query && (int32_t) plan.tokens.size() < pad_to) {
        llama_token mask = p.query_pad_token_id >= 0 ? (llama_token) p.query_pad_token_id
                         : p.mask_token_id >= 0      ? (llama_token) p.mask_token_id
                                                     : llama_vocab_mask(vocab);
        if (mask == LLAMA_TOKEN_NULL) {
            err = "vocabulary does not define a mask token required for query expansion";
            return false;
        }
        while ((int32_t) plan.tokens.size() < pad_to) {
            plan.tokens.push_back(mask);
        }
    }

    plan.output_mask.resize(plan.tokens.size());
    for (size_t i = 0; i < plan.tokens.size(); i++) {
        const bool retain = is_query || should_retain_document_token(vocab, p, plan.tokens[i]);
        plan.output_mask[i] = retain;
        if (retain) {
            plan.output_count++;
        }
    }
    if (plan.output_count == 0) {
        err = "input retained no token vectors";
        return false;
    }
    return true;
}

bool encode_response_item(const encode_plan & plan,
                          const std::vector<std::vector<float>> & rows,
                          ordered_json & item,
                          std::string & err) {
    const state & s = g_state;

    if (rows.size() != plan.tokens.size()) {
        err = "model returned " + std::to_string(rows.size()) +
              " token embeddings for " + std::to_string(plan.tokens.size()) + " planned tokens";
        return false;
    }

    ordered_json tokens    = ordered_json::array();
    ordered_json embedding = ordered_json::array();

    std::vector<float> projected(s.out_dim);
    for (size_t i = 0; i < rows.size(); i++) {
        if (!plan.output_mask[i]) {
            continue;
        }
        const std::vector<float> & row = rows[i];
        if ((int32_t) row.size() != s.n_embd) {
            err = "model returned a token embedding of dim " + std::to_string(row.size()) +
                  ", expected " + std::to_string(s.n_embd);
            return false;
        }

        if (s.has_projection) {
            for (int32_t j = 0; j < s.proj.out_features; j++) {
                double value = s.proj.bias.empty() ? 0.0 : (double) s.proj.bias[j];
                const float * w = s.proj.weight.data() + (size_t) j * s.proj.in_features;
                for (int32_t k = 0; k < s.proj.in_features; k++) {
                    value += (double) row[k] * (double) w[k];
                }
                projected[j] = (float) value;
            }
        } else {
            std::copy(row.begin(), row.end(), projected.begin());
        }

        double norm = 0.0;
        for (int32_t j = 0; j < s.out_dim; j++) {
            norm += (double) projected[j] * (double) projected[j];
        }
        if (norm <= 0.0 || !std::isfinite(norm)) {
            err = "model returned a zero or non-finite token embedding";
            return false;
        }
        norm = std::sqrt(norm);

        ordered_json vec = ordered_json::array();
        for (int32_t j = 0; j < s.out_dim; j++) {
            vec.push_back((float) (projected[j] / norm));
        }
        embedding.push_back(std::move(vec));
        tokens.push_back((int32_t) plan.tokens[i]);
    }

    item = ordered_json{
        {"tokens",    std::move(tokens)},
        {"embedding", std::move(embedding)},
    };
    return true;
}

int output_dim() {
    return g_state.out_dim;
}

} // namespace ollama_colbert
