package server

import (
	"errors"
	"fmt"

	"github.com/ollama/ollama/api"
)

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
