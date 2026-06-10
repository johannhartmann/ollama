package server

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
)

// multivectorRedirectError is returned by the dense embedding endpoints when a
// model emits per-token (pooling=none) embeddings rather than a single pooled
// vector.
const multivectorRedirectError = "model produces multivector embeddings; use /api/multivectors"

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

