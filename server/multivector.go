package server

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
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

// colbertProfile decodes an optional ColBERT profile embedded in the model's
// GGUF metadata under "pg_colbert.profile_json". The raw string is already
// surfaced verbatim through /api/show's model_info; this helper decodes it for
// callers that want structured access. The profile itself is applied by
// llama-server's /colbert endpoint (prefixes, query expansion, skiplist,
// projection); on the Go side it is informational. It returns nil when the
// key is absent, empty, or not valid JSON — a malformed profile must never
// block model use, so the error is only logged.
func colbertProfile(kv ggml.KV) map[string]any {
	raw, ok := kv["pg_colbert.profile_json"]
	if !ok {
		return nil
	}

	s, ok := raw.(string)
	if !ok || s == "" {
		return nil
	}

	var profile map[string]any
	if err := json.Unmarshal([]byte(s), &profile); err != nil {
		slog.Debug("ignoring malformed pg_colbert.profile_json", "error", err)
		return nil
	}
	return profile
}

// multiVectorSimilarity describes how the returned encodings are meant to be
// compared. Ollama serves the encodings; late interaction (MaxSim) is performed
// downstream.
var multiVectorSimilarity = api.MultiVectorSimilarity{
	Comparator:    "max_sim",
	Metric:        "dot",
	Normalization: "model_or_raw",
}

// encodeFloat32MatrixBase64 encodes a token-row matrix as base64 of its
// row-major, little-endian float32 bytes. A ragged matrix is an error; an empty
// matrix encodes to the empty string.
func encodeFloat32MatrixBase64(vectors [][]float32) (string, error) {
	if _, _, err := validateMultiVectorMatrix(vectors); err != nil {
		return "", err
	}

	var total int
	for _, row := range vectors {
		total += len(row) * 4
	}

	buf := make([]byte, 0, total)
	var scratch [4]byte
	for _, row := range vectors {
		for _, v := range row {
			binary.LittleEndian.PutUint32(scratch[:], math.Float32bits(v))
			buf = append(buf, scratch[:]...)
		}
	}

	return base64.StdEncoding.EncodeToString(buf), nil
}

// newMultiVectorData assembles the per-input response item, validating that the
// token matrix is rectangular. The "" and "float" encodings return the rows in
// Vectors; "base64" returns them packed into Data instead. Shape is always set.
func newMultiVectorData(index int, vectors [][]float32, tokens []int, truncated bool, encodingFormat string) (api.MultiVectorData, error) {
	rows, dim, err := validateMultiVectorMatrix(vectors)
	if err != nil {
		return api.MultiVectorData{}, err
	}

	item := api.MultiVectorData{
		Index:     index,
		Shape:     []int{rows, dim},
		Tokens:    tokens,
		Truncated: truncated,
	}

	if encodingFormat == "base64" {
		encoded, err := encodeFloat32MatrixBase64(vectors)
		if err != nil {
			return api.MultiVectorData{}, err
		}
		item.Data = encoded
	} else {
		item.Vectors = vectors
	}

	return item, nil
}

// MultiVectorHandler serves POST /api/multivectors. It mirrors the safe parts of
// EmbedHandler but preserves every token row: it never normalizes, never crops
// dimensions, and never drops rows. It is local-only and requires a pooling=none
// model.
func (s *Server) MultiVectorHandler(c *gin.Context) {
	checkpointStart := time.Now()

	var req api.MultiVectorRequest
	switch err := c.ShouldBindJSON(&req); {
	case errors.Is(err, io.EOF):
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	case err != nil:
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := validateMultiVectorRequest(req); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if req.IncludeTokenText {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "include_token_text is not implemented for multivectors"})
		return
	}

	modelRef, err := parseAndValidateModelRef(req.Model)
	if err != nil {
		writeModelRefParseError(c, err, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model))
		return
	}
	if modelRef.Source == modelSourceCloud {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "multivector embeddings are only available for local models"})
		return
	}

	input, err := parseEmbedLikeInput(req.Input)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	name, err := getExistingName(modelRef.Name)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": fmt.Sprintf("model '%s' not found", req.Model)})
		return
	}

	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{}, req.Options, req.KeepAlive, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	if len(input) == 0 {
		c.JSON(http.StatusOK, api.MultiVectorResponse{
			Model:         req.Model,
			EmbeddingType: "multi_vector",
			Pooling:       "none",
			Similarity:    multiVectorSimilarity,
			Data:          []api.MultiVectorData{},
		})
		return
	}

	kvData, _, err := getModelData(m.ModelPath, false)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if !isPoolingNone(kvData) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "model does not produce multivector embeddings"})
		return
	}

	ctx := c.Request.Context()

	adjustTokenLimit := func(tokens []int, limit int) int {
		if bos := kvData.Uint("tokenizer.ggml.bos_token_id"); len(tokens) > 0 && tokens[0] != int(bos) && kvData.Bool("add_bos_token", true) {
			limit--
		}
		if eos := kvData.Uint("tokenizer.ggml.eos_token_id"); len(tokens) > 0 && tokens[len(tokens)-1] != int(eos) && kvData.Bool("add_eos_token", true) {
			limit--
		}
		return limit
	}

	// prepare tokenizes the input, applies the model context limit, and (unless
	// truncation is disabled) trims the input to fit. It returns the text and
	// token ids that will actually be embedded.
	prepare := func(text string) (string, []int, bool, error) {
		tokens, err := r.Tokenize(ctx, text)
		if err != nil {
			return "", nil, false, err
		}

		ctxLen := int(kvData.ContextLength())
		if opts.NumCtx > 0 {
			ctxLen = min(opts.NumCtx, ctxLen)
		}
		ctxLen = adjustTokenLimit(tokens, ctxLen)
		if ctxLen <= 0 {
			return "", nil, false, fmt.Errorf("input after truncation exceeds maximum context length")
		}

		if len(tokens) <= ctxLen {
			return text, tokens, false, nil
		}

		if req.Truncate != nil && !*req.Truncate {
			return "", nil, false, api.StatusError{
				StatusCode:   http.StatusBadRequest,
				ErrorMessage: "the input length exceeds the context length",
			}
		}

		truncatedTokens := tokens[:ctxLen]
		truncatedText, err := r.Detokenize(ctx, truncatedTokens)
		if err != nil {
			return "", nil, false, err
		}
		return truncatedText, truncatedTokens, true, nil
	}

	var g errgroup.Group
	data := make([]api.MultiVectorData, len(input))
	var totalTokens uint64
	for i, text := range input {
		g.Go(func() error {
			finalText, _, truncated, err := prepare(text)
			if err != nil {
				return err
			}

			result, promptTokens, err := r.MultiVector(ctx, finalText, llm.MultiVectorOptions{InputType: req.InputType})
			if err != nil {
				return err
			}

			// Report the retained plan tokens: these are the ids the returned
			// vectors actually correspond to (after the ColBERT prefix, query
			// expansion and document skiplist are applied by the runner).
			var reportTokens []int
			if req.IncludeTokens {
				reportTokens = make([]int, len(result.Tokens))
				for j, t := range result.Tokens {
					reportTokens[j] = int(t)
				}
			}

			// Truncation can happen on either side: the context-length guard
			// above, or the runner cutting the plan to the profile max length.
			item, err := newMultiVectorData(i, result.Vectors, reportTokens, truncated || result.Truncated, req.EncodingFormat)
			if err != nil {
				return err
			}

			data[i] = item
			atomic.AddUint64(&totalTokens, uint64(promptTokens))
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		s.sched.expireRunnersForRuntimeOOM(m, err)
		var serr api.StatusError
		if errors.As(err, &serr) {
			c.AbortWithStatusJSON(serr.StatusCode, gin.H{"error": strings.TrimSpace(serr.ErrorMessage)})
			return
		}
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": strings.TrimSpace(err.Error())})
		return
	}

	dimension := 0
	for _, d := range data {
		if len(d.Shape) == 2 && d.Shape[0] > 0 {
			dimension = d.Shape[1]
			break
		}
	}

	c.JSON(http.StatusOK, api.MultiVectorResponse{
		Model:           req.Model,
		EmbeddingType:   "multi_vector",
		Pooling:         "none",
		Similarity:      multiVectorSimilarity,
		Dimension:       dimension,
		Data:            data,
		TotalDuration:   time.Since(checkpointStart),
		LoadDuration:    checkpointLoaded.Sub(checkpointStart),
		PromptEvalCount: int(totalTokens),
	})
}
