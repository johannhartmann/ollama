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
	if resp.EmbeddingType != "multi_vector" || resp.Pooling != "none" {
		t.Fatalf("unexpected envelope: %+v", resp)
	}
	if len(resp.Data) != 0 {
		t.Fatalf("expected empty data, got %d items", len(resp.Data))
	}
}

func TestMultiVectorHandlerValidationErrors(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "2048")
	gin.SetMode(gin.TestMode)

	mock := mockRunner{}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "colbert", multivectorKV(), "", nil)

	cases := []struct {
		name string
		req  api.MultiVectorRequest
	}{
		{name: "invalid input type", req: api.MultiVectorRequest{Model: "colbert", Input: []any{"ok", 5}}},
		{name: "bad input_type", req: api.MultiVectorRequest{Model: "colbert", Input: "hi", InputType: "passage"}},
		{name: "include_token_text", req: api.MultiVectorRequest{Model: "colbert", Input: "hi", IncludeTokenText: true}},
		{name: "base64 not yet", req: api.MultiVectorRequest{Model: "colbert", Input: "hi", EncodingFormat: "base64"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := createRequest(t, s.MultiVectorHandler, tc.req)
			if w.Code != http.StatusBadRequest {
				t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
			}
		})
	}
}

func TestMultiVectorHandlerRejectsPooledModel(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "2048")
	gin.SetMode(gin.TestMode)

	mock := mockRunner{}
	s := newServerWithMockRunner(t, &mock)
	// pooling_type=1 (mean) is a dense embedding model, not multivector.
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
				return llm.MultiVectorResult{Vectors: [][]float32{{3, 4}}, Dimension: 2}, 0, nil
			default:
				return llm.MultiVectorResult{Vectors: [][]float32{{5, 6}, {7, 8}}, Dimension: 2}, 0, nil
			}
		},
	}
	s := newServerWithMockRunner(t, &mock)
	createMinimalGGUFModel(t, s, "colbert", multivectorKV(), "", nil)

	w := createRequest(t, s.MultiVectorHandler, api.MultiVectorRequest{
		Model:         "colbert",
		Input:         []any{"alpha", "beta gamma"},
		IncludeTokens: true,
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
	// No normalization: raw values are preserved verbatim.
	if !reflect.DeepEqual(resp.Data[0].Vectors, [][]float32{{3, 4}}) {
		t.Fatalf("data[0].vectors = %v, want [[3 4]] (no normalization expected)", resp.Data[0].Vectors)
	}
	// include_tokens reflects mock tokenization (one token per whitespace field).
	if len(resp.Data[0].Tokens) != 1 || len(resp.Data[1].Tokens) != 2 {
		t.Fatalf("tokens = %v / %v, want 1 and 2 tokens", resp.Data[0].Tokens, resp.Data[1].Tokens)
	}
	if resp.Data[0].Data != "" || resp.Data[1].Data != "" {
		t.Fatal("expected no base64 data field for float encoding")
	}
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
