package server

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
)

// multivectorRedirectError is returned by the dense embedding endpoints when a
// model emits per-token (pooling=none) embeddings rather than a single pooled
// vector.
const multivectorRedirectError = "model produces multivector embeddings; use /api/multivectors"

// parseEmbedLikeInput normalizes the polymorphic "input" field shared by the
// embedding-style endpoints into a slice of strings. It accepts a single string
// or a list of strings; any other shape (including a non-string list element) is
// an error. A nil input yields an empty, non-nil slice.
func parseEmbedLikeInput(input any) ([]string, error) {
	switch i := input.(type) {
	case nil:
		return []string{}, nil
	case string:
		if i == "" {
			return []string{}, nil
		}
		return []string{i}, nil
	case []any:
		out := make([]string, 0, len(i))
		for _, v := range i {
			s, ok := v.(string)
			if !ok {
				return nil, errors.New("invalid input type")
			}
			out = append(out, s)
		}
		return out, nil
	default:
		return nil, errors.New("invalid input type")
	}
}

// validateMultiVectorRequest checks the request fields that can be validated
// without loading the model. It does not reject empty input: an empty input is
// answered with an empty data list by the handler.
func validateMultiVectorRequest(req api.MultiVectorRequest) error {
	switch req.InputType {
	case "", "query", "document":
	default:
		return fmt.Errorf("invalid input_type %q: must be one of \"\", \"query\", or \"document\"", req.InputType)
	}

	switch req.EncodingFormat {
	case "", "float", "base64":
	default:
		return fmt.Errorf("invalid encoding_format %q: must be one of \"\", \"float\", or \"base64\"", req.EncodingFormat)
	}

	return nil
}

// validateMultiVectorMatrix verifies that a token-row matrix is rectangular and
// reports its dimensions. An empty matrix is valid and reports (0, 0).
func validateMultiVectorMatrix(vectors [][]float32) (rows, dim int, err error) {
	if len(vectors) == 0 {
		return 0, 0, nil
	}

	dim = len(vectors[0])
	for i, row := range vectors {
		if len(row) != dim {
			return 0, 0, fmt.Errorf("ragged multivector matrix: row %d has %d values, expected %d", i, len(row), dim)
		}
	}

	return len(vectors), dim, nil
}

// normalizePoolingType maps a GGUF pooling_type value to its canonical name.
// The value is usually a numeric enum (llama.cpp: 0=none, 1=mean, 2=cls,
// 3=last, 4=rank) but a string form (either the name or its decimal) is
// tolerated. The bool is false for unknown values.
func normalizePoolingType(v any) (string, bool) {
	name := func(i int64) (string, bool) {
		switch i {
		case 0:
			return "none", true
		case 1:
			return "mean", true
		case 2:
			return "cls", true
		case 3:
			return "last", true
		case 4:
			return "rank", true
		default:
			return "", false
		}
	}

	switch t := v.(type) {
	case uint32:
		return name(int64(t))
	case uint64:
		return name(int64(t))
	case uint:
		return name(int64(t))
	case int:
		return name(int64(t))
	case int32:
		return name(int64(t))
	case int64:
		return name(t)
	case float32:
		return name(int64(t))
	case float64:
		return name(int64(t))
	case string:
		s := strings.ToLower(strings.TrimSpace(t))
		switch s {
		case "none", "mean", "cls", "last", "rank":
			return s, true
		}
		if i, err := strconv.ParseInt(s, 10, 64); err == nil {
			return name(i)
		}
		return "", false
	default:
		return "", false
	}
}

// ggufPoolingType reports the model's pooling type from its GGUF metadata. The
// bool is false when the model declares no pooling_type (i.e. it is not an
// embedding model) or the value is unrecognized. Presence is checked directly
// because a missing key and pooling_type=none both read as the zero value.
func ggufPoolingType(kv ggml.KV) (string, bool) {
	raw, ok := kv[fmt.Sprintf("%s.pooling_type", kv.Architecture())]
	if !ok {
		return "", false
	}
	return normalizePoolingType(raw)
}

// isPoolingNone reports whether the model emits per-token (multivector)
// embeddings rather than a pooled vector.
func isPoolingNone(kv ggml.KV) bool {
	p, ok := ggufPoolingType(kv)
	return ok && p == "none"
}

// isPoolingRank reports whether the model is a reranker. Reserved for future
// reranking guardrails.
func isPoolingRank(kv ggml.KV) bool {
	p, ok := ggufPoolingType(kv)
	return ok && p == "rank"
}
