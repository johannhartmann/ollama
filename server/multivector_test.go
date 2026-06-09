package server

import (
	"encoding/base64"
	"encoding/binary"
	"math"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
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

func TestNormalizePoolingType(t *testing.T) {
	cases := []struct {
		name   string
		value  any
		want   string
		wantOk bool
	}{
		{name: "uint32 none", value: uint32(0), want: "none", wantOk: true},
		{name: "uint32 mean", value: uint32(1), want: "mean", wantOk: true},
		{name: "int cls", value: 2, want: "cls", wantOk: true},
		{name: "int64 last", value: int64(3), want: "last", wantOk: true},
		{name: "float64 rank", value: float64(4), want: "rank", wantOk: true},
		{name: "string name", value: "none", want: "none", wantOk: true},
		{name: "string mixed case", value: "Mean", want: "mean", wantOk: true},
		{name: "string numeric", value: "2", want: "cls", wantOk: true},
		{name: "out of range", value: uint32(9), wantOk: false},
		{name: "unknown string", value: "softmax", wantOk: false},
		{name: "wrong type", value: []int{0}, wantOk: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := normalizePoolingType(tc.value)
			if ok != tc.wantOk || (ok && got != tc.want) {
				t.Fatalf("got (%q, %v), want (%q, %v)", got, ok, tc.want, tc.wantOk)
			}
		})
	}
}

func TestEncodeFloat32MatrixBase64(t *testing.T) {
	t.Run("2x2 byte length and round trip", func(t *testing.T) {
		matrix := [][]float32{{1.5, -2.25}, {0, 3.5}}
		encoded, err := encodeFloat32MatrixBase64(matrix)
		if err != nil {
			t.Fatal(err)
		}

		raw, err := base64.StdEncoding.DecodeString(encoded)
		if err != nil {
			t.Fatal(err)
		}
		if len(raw) != 2*2*4 {
			t.Fatalf("decoded byte length = %d, want 16", len(raw))
		}

		var got []float32
		for i := 0; i < len(raw); i += 4 {
			got = append(got, math.Float32frombits(binary.LittleEndian.Uint32(raw[i:i+4])))
		}
		want := []float32{1.5, -2.25, 0, 3.5}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("decoded[%d] = %v, want %v", i, got[i], want[i])
			}
		}
	})

	t.Run("empty matrix", func(t *testing.T) {
		encoded, err := encodeFloat32MatrixBase64(nil)
		if err != nil {
			t.Fatal(err)
		}
		if encoded != "" {
			t.Fatalf("expected empty string, got %q", encoded)
		}
	})

	t.Run("ragged matrix", func(t *testing.T) {
		if _, err := encodeFloat32MatrixBase64([][]float32{{1, 2}, {3}}); err == nil {
			t.Fatal("expected error for ragged matrix")
		}
	})
}

func TestNewMultiVectorDataEncoding(t *testing.T) {
	matrix := [][]float32{{1, 2, 3}}

	float, err := newMultiVectorData(0, matrix, nil, false, "float")
	if err != nil {
		t.Fatal(err)
	}
	if float.Data != "" || len(float.Vectors) != 1 {
		t.Fatalf("float encoding should return vectors and no data: %+v", float)
	}

	b64, err := newMultiVectorData(0, matrix, nil, false, "base64")
	if err != nil {
		t.Fatal(err)
	}
	if b64.Data == "" || b64.Vectors != nil {
		t.Fatalf("base64 encoding should return data and no vectors: %+v", b64)
	}
	if want := []int{1, 3}; b64.Shape[0] != want[0] || b64.Shape[1] != want[1] {
		t.Fatalf("base64 shape = %v, want %v", b64.Shape, want)
	}
}

func TestPoolingTypeFromKV(t *testing.T) {
	cases := []struct {
		name     string
		kv       ggml.KV
		wantType string
		wantOk   bool
		none     bool
		rank     bool
	}{
		{
			name:   "no pooling key",
			kv:     ggml.KV{"general.architecture": "llama"},
			wantOk: false,
		},
		{
			name:     "pooling none",
			kv:       ggml.KV{"general.architecture": "bert", "bert.pooling_type": uint32(0)},
			wantType: "none", wantOk: true, none: true,
		},
		{
			name:     "pooling mean",
			kv:       ggml.KV{"general.architecture": "bert", "bert.pooling_type": uint32(1)},
			wantType: "mean", wantOk: true,
		},
		{
			name:     "pooling rank",
			kv:       ggml.KV{"general.architecture": "bert", "bert.pooling_type": uint32(4)},
			wantType: "rank", wantOk: true, rank: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := ggufPoolingType(tc.kv)
			if ok != tc.wantOk || (ok && got != tc.wantType) {
				t.Fatalf("ggufPoolingType = (%q, %v), want (%q, %v)", got, ok, tc.wantType, tc.wantOk)
			}
			if isPoolingNone(tc.kv) != tc.none {
				t.Fatalf("isPoolingNone = %v, want %v", isPoolingNone(tc.kv), tc.none)
			}
			if isPoolingRank(tc.kv) != tc.rank {
				t.Fatalf("isPoolingRank = %v, want %v", isPoolingRank(tc.kv), tc.rank)
			}
		})
	}
}
