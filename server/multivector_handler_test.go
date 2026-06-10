package server

import (
	"context"
	"encoding/json"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
)

// multivectorKV returns the GGUF metadata for a pooling=none (multivector) model.
func multivectorKV() ggml.KV {
	return ggml.KV{"llama.pooling_type": uint32(0)}
}

func TestMultiVectorHandlerEmptyInput(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "2048")
	gin.SetMode(gin.TestMode)

	mock := mockRunner{}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "colbert", multivectorKV(), "", nil)

	w := createRequest(t, s.MultiVectorHandler, api.MultiVectorRequest{Model: "colbert", Input: ""})
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp api.MultiVectorResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Data) != 0 {
		t.Fatalf("expected empty data, got %d items", len(resp.Data))
	}
}

func TestMultiVectorHandlerInvalidInput(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "2048")
	gin.SetMode(gin.TestMode)

	mock := mockRunner{}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "colbert", multivectorKV(), "", nil)

	w := createRequest(t, s.MultiVectorHandler, api.MultiVectorRequest{Model: "colbert", Input: []any{"ok", 5}})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
	}
}

func TestMultiVectorHandlerRejectsPooledModel(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "2048")
	gin.SetMode(gin.TestMode)

	mock := mockRunner{}
	s := newServerWithMockRunner(t, &mock)
	// pooling_type=1 (mean) is a dense embedding model, not multivector; it
	// fails the multivector capability check during scheduling.
	createMinimalGGUFModel(t, s, "dense", ggml.KV{"llama.pooling_type": uint32(1)}, "", nil)

	w := createRequest(t, s.MultiVectorHandler, api.MultiVectorRequest{Model: "dense", Input: "hello"})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
	}
}

func TestMultiVectorHandlerTwoInputs(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "2048")
	gin.SetMode(gin.TestMode)

	// Return matrices with different row counts per input. The values are not
	// unit-normalized, so an unchanged response proves no normalization occurred.
	mock := mockRunner{
		MultiVectorFn: func(_ context.Context, input string, _ llm.MultiVectorOptions) (llm.MultiVectorResult, int, error) {
			switch input {
			case "alpha":
				return llm.MultiVectorResult{Vectors: [][]float32{{3, 4}}, Dimension: 2}, 1, nil
			default:
				return llm.MultiVectorResult{Vectors: [][]float32{{5, 6}, {7, 8}}, Dimension: 2}, 2, nil
			}
		},
	}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "colbert", multivectorKV(), "", nil)

	w := createRequest(t, s.MultiVectorHandler, api.MultiVectorRequest{
		Model: "colbert",
		Input: []any{"alpha", "beta gamma"},
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp api.MultiVectorResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Dimension != 2 {
		t.Fatalf("dimension = %d, want 2", resp.Dimension)
	}
	if len(resp.Data) != 2 {
		t.Fatalf("expected 2 data items, got %d", len(resp.Data))
	}
	if !reflect.DeepEqual(resp.Data[0].Shape, []int{1, 2}) {
		t.Fatalf("data[0].shape = %v, want [1 2]", resp.Data[0].Shape)
	}
	if !reflect.DeepEqual(resp.Data[1].Shape, []int{2, 2}) {
		t.Fatalf("data[1].shape = %v, want [2 2]", resp.Data[1].Shape)
	}
	// Raw rows are preserved verbatim: no normalization, no cropping.
	if !reflect.DeepEqual(resp.Data[0].Vectors, [][]float32{{3, 4}}) {
		t.Fatalf("data[0].vectors = %v, want [[3 4]] (no normalization expected)", resp.Data[0].Vectors)
	}
	if resp.PromptEvalCount != 3 {
		t.Fatalf("prompt_eval_count = %d, want 3", resp.PromptEvalCount)
	}
}

func TestMultiVectorHandlerTruncation(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "2048")
	gin.SetMode(gin.TestMode)

	var gotInput string
	mock := mockRunner{
		MultiVectorFn: func(_ context.Context, input string, _ llm.MultiVectorOptions) (llm.MultiVectorResult, int, error) {
			gotInput = input
			return llm.MultiVectorResult{Vectors: [][]float32{{1, 2}}, Dimension: 2}, 1, nil
		},
	}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "colbert", multivectorKV(), "", nil)

	// the mock tokenizer emits one token per whitespace-separated word
	longInput := strings.TrimSpace(strings.Repeat("word ", 5000))

	t.Run("truncate=false rejects over-length input", func(t *testing.T) {
		truncate := false
		w := createRequest(t, s.MultiVectorHandler, api.MultiVectorRequest{
			Model:    "colbert",
			Input:    longInput,
			Truncate: &truncate,
		})
		if w.Code != http.StatusBadRequest {
			t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), "context length") {
			t.Fatalf("expected a context length error, got %s", w.Body.String())
		}
	})

	t.Run("default truncates and reports it", func(t *testing.T) {
		w := createRequest(t, s.MultiVectorHandler, api.MultiVectorRequest{
			Model: "colbert",
			Input: longInput,
		})
		if w.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
		}

		var resp api.MultiVectorResponse
		if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		if len(resp.Data) != 1 || !resp.Data[0].Truncated {
			t.Fatalf("expected one truncated item, got %+v", resp.Data)
		}
		// the runner must receive the detokenized, shortened input
		if got := len(strings.Fields(gotInput)); got >= 5000 || got == 0 {
			t.Fatalf("runner received %d tokens, expected a truncated, non-empty input", got)
		}
	})

	t.Run("short input is not truncated", func(t *testing.T) {
		w := createRequest(t, s.MultiVectorHandler, api.MultiVectorRequest{
			Model: "colbert",
			Input: "short input",
		})
		if w.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
		}
		var resp api.MultiVectorResponse
		if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
			t.Fatal(err)
		}
		if len(resp.Data) != 1 || resp.Data[0].Truncated {
			t.Fatalf("expected one untruncated item, got %+v", resp.Data)
		}
		if gotInput != "short input" {
			t.Fatalf("runner received %q, want the input verbatim", gotInput)
		}
	})
}

func TestEmbedHandlerRejectsMultiVectorModel(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "2048")
	gin.SetMode(gin.TestMode)

	mock := mockRunner{}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "colbert", multivectorKV(), "", nil)

	w := createRequest(t, s.EmbedHandler, api.EmbedRequest{Model: "colbert", Input: "hello"})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 from /api/embed for pooling=none model, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "/api/multivectors") {
		t.Fatalf("expected redirect to /api/multivectors, got %s", w.Body.String())
	}
}
