package server

import (
	"testing"

	"github.com/ollama/ollama/api"
)

func TestParseEmbedLikeInput(t *testing.T) {
	cases := []struct {
		name    string
		input   any
		want    []string
		wantErr bool
	}{
		{name: "nil", input: nil, want: []string{}},
		{name: "empty string", input: "", want: []string{}},
		{name: "single string", input: "hello", want: []string{"hello"}},
		{name: "string slice", input: []any{"a", "b"}, want: []string{"a", "b"}},
		{name: "empty slice", input: []any{}, want: []string{}},
		{name: "non-string element", input: []any{"a", 1}, wantErr: true},
		{name: "number", input: 42, wantErr: true},
		{name: "map", input: map[string]any{"x": 1}, wantErr: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseEmbedLikeInput(tc.input)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %v", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(got) != len(tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("got %v, want %v", got, tc.want)
				}
			}
		})
	}
}

func TestValidateMultiVectorRequest(t *testing.T) {
	cases := []struct {
		name    string
		req     api.MultiVectorRequest
		wantErr bool
	}{
		{name: "defaults", req: api.MultiVectorRequest{}},
		{name: "query", req: api.MultiVectorRequest{InputType: "query"}},
		{name: "document", req: api.MultiVectorRequest{InputType: "document"}},
		{name: "bad input_type", req: api.MultiVectorRequest{InputType: "passage"}, wantErr: true},
		{name: "float", req: api.MultiVectorRequest{EncodingFormat: "float"}},
		{name: "base64", req: api.MultiVectorRequest{EncodingFormat: "base64"}},
		{name: "bad encoding", req: api.MultiVectorRequest{EncodingFormat: "protobuf"}, wantErr: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateMultiVectorRequest(tc.req)
			if tc.wantErr != (err != nil) {
				t.Fatalf("wantErr=%v, got err=%v", tc.wantErr, err)
			}
		})
	}
}

func TestValidateMultiVectorMatrix(t *testing.T) {
	cases := []struct {
		name     string
		vectors  [][]float32
		wantRows int
		wantDim  int
		wantErr  bool
	}{
		{name: "empty", vectors: nil, wantRows: 0, wantDim: 0},
		{name: "single row", vectors: [][]float32{{1, 2, 3}}, wantRows: 1, wantDim: 3},
		{name: "rectangular", vectors: [][]float32{{1, 2}, {3, 4}, {5, 6}}, wantRows: 3, wantDim: 2},
		{name: "ragged", vectors: [][]float32{{1, 2}, {3}}, wantErr: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rows, dim, err := validateMultiVectorMatrix(tc.vectors)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if rows != tc.wantRows || dim != tc.wantDim {
				t.Fatalf("got (rows=%d, dim=%d), want (rows=%d, dim=%d)", rows, dim, tc.wantRows, tc.wantDim)
			}
		})
	}
}
