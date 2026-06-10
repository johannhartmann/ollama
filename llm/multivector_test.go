package llm

import (
	"net/http"
	"reflect"
	"strings"
	"testing"
)

func TestParseMultiVectorEmbeddings(t *testing.T) {
	t.Run("two token rows", func(t *testing.T) {
		body := `[{"index":0,"embedding":[[1.0,2.0],[3.0,4.0]]}]`
		result, count, err := parseMultiVectorEmbeddings([]byte(body))
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(result.Vectors, [][]float32{{1, 2}, {3, 4}}) {
			t.Fatalf("vectors = %v", result.Vectors)
		}
		if result.Dimension != 2 {
			t.Fatalf("dimension = %d, want 2", result.Dimension)
		}
		if count != 2 {
			t.Fatalf("count = %d, want 2", count)
		}
	})

	cases := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name:    "empty result list",
			body:    `[]`,
			wantErr: "contained 0 results",
		},
		{
			name:    "multiple result items",
			body:    `[{"index":0,"embedding":[[1]]},{"index":1,"embedding":[[2]]}]`,
			wantErr: "contained 2 results",
		},
		{
			name:    "no rows",
			body:    `[{"index":0,"embedding":[]}]`,
			wantErr: "did not return token embeddings",
		},
		{
			name:    "ragged rows",
			body:    `[{"index":0,"embedding":[[1.0,2.0],[3.0]]}]`,
			wantErr: "ragged token embeddings",
		},
		{
			name:    "dense vector instead of matrix",
			body:    `[{"index":0,"embedding":[1.0,2.0]}]`,
			wantErr: "unmarshal embeddings response",
		},
		{
			name:    "not json",
			body:    `not json`,
			wantErr: "unmarshal embeddings response",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := parseMultiVectorEmbeddings([]byte(tc.body))
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want substring %q", err, tc.wantErr)
			}
		})
	}
}

func TestNormalizeEmbeddingErrorLimits(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		wantStatus int
		wantMsg    string
	}{
		{
			name:       "batch size limit",
			body:       `{"error":{"message":"input is too large to process. increase the physical batch size"}}`,
			wantStatus: http.StatusBadRequest,
			wantMsg:    "the input length exceeds the batch size; increase num_batch",
		},
		{
			name:       "context limit",
			body:       `{"error":{"message":"the request exceeds the available context size"}}`,
			wantStatus: http.StatusBadRequest,
			wantMsg:    "the input length exceeds the context length",
		},
		{
			name:       "other error passes through",
			body:       `{"error":{"message":"model is sleeping"}}`,
			wantStatus: http.StatusInternalServerError,
			wantMsg:    "model is sleeping",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			status, msg := normalizeEmbeddingError(http.StatusInternalServerError, []byte(tc.body))
			if status != tc.wantStatus || msg != tc.wantMsg {
				t.Fatalf("got (%d, %q), want (%d, %q)", status, msg, tc.wantStatus, tc.wantMsg)
			}
		})
	}
}
