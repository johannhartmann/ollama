package llm

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseMultiVectorEmbeddings(t *testing.T) {
	cases := []struct {
		name    string
		body    string
		want    [][]float32
		wantErr string
	}{
		{
			name: "valid two rows three dims",
			body: `[{"index":0,"embedding":[[1,2,3],[4,5,6]]}]`,
			want: [][]float32{{1, 2, 3}, {4, 5, 6}},
		},
		{
			name: "single row",
			body: `[{"index":0,"embedding":[[0.5,-0.5]]}]`,
			want: [][]float32{{0.5, -0.5}},
		},
		{
			name: "out of order index zero not first",
			body: `[{"index":1,"embedding":[[9,9]]},{"index":0,"embedding":[[1,2]]}]`,
			want: [][]float32{{1, 2}},
		},
		{
			name:    "empty matrix",
			body:    `[{"index":0,"embedding":[]}]`,
			wantErr: "no token rows",
		},
		{
			name:    "ragged rows",
			body:    `[{"index":0,"embedding":[[1,2,3],[4,5]]}]`,
			wantErr: "ragged",
		},
		{
			name:    "dense response",
			body:    `[{"index":0,"embedding":[1,2,3]}]`,
			wantErr: "pooling is not none",
		},
		{
			name:    "missing embedding",
			body:    `[{"index":0}]`,
			wantErr: "missing embedding",
		},
		{
			name:    "empty array",
			body:    `[]`,
			wantErr: "empty multivector response",
		},
		{
			name:    "missing index zero",
			body:    `[{"index":1,"embedding":[[1,2]]},{"index":2,"embedding":[[3,4]]}]`,
			wantErr: "missing index 0",
		},
		{
			name:    "invalid json",
			body:    `not json`,
			wantErr: "unmarshal multivector response",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseMultiVectorEmbeddings([]byte(tc.body))
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil (result %v)", tc.wantErr, got)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("error %q does not contain %q", err.Error(), tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("got %v, want %v", got, tc.want)
			}
		})
	}
}
