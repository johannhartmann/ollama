package server

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
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

// MultiVectorHandler serves POST /api/multivectors. It returns one matrix per
// input — one row per token, as the model produces them. Ollama does not
// normalize rows, crop dimensions, or apply retrieval-model semantics
// (query/document markers, query expansion, skiplist filtering, MaxSim);
// those are the caller's responsibility. Local pooling=none models only.
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

	r, m, opts, err := s.scheduleRunner(c.Request.Context(), name.String(), []model.Capability{model.CapabilityMultivector}, req.Options, req.KeepAlive, nil)
	if err != nil {
		handleScheduleError(c, req.Model, err)
		return
	}

	checkpointLoaded := time.Now()

	if len(input) == 0 {
		c.JSON(http.StatusOK, api.MultiVectorResponse{
			Model: req.Model,
			Data:  []api.MultiVectorData{},
		})
		return
	}

	kvData, _, err := getModelData(m.ModelPath, false)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	// defensive guard behind the capability check above
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
	// truncation is disabled) trims the input to fit. It returns the text that
	// will actually be embedded.
	prepare := func(text string) (string, bool, error) {
		tokens, err := r.Tokenize(ctx, text)
		if err != nil {
			return "", false, err
		}

		ctxLen := int(kvData.ContextLength())
		if opts.NumCtx > 0 {
			ctxLen = min(opts.NumCtx, ctxLen)
		}
		ctxLen = adjustTokenLimit(tokens, ctxLen)
		if ctxLen <= 0 {
			return "", false, fmt.Errorf("input after truncation exceeds maximum context length")
		}

		if len(tokens) <= ctxLen {
			return text, false, nil
		}

		if req.Truncate != nil && !*req.Truncate {
			return "", false, api.StatusError{
				StatusCode:   http.StatusBadRequest,
				ErrorMessage: "the input length exceeds the context length",
			}
		}

		truncatedText, err := r.Detokenize(ctx, tokens[:ctxLen])
		if err != nil {
			return "", false, err
		}
		return truncatedText, true, nil
	}

	var g errgroup.Group
	data := make([]api.MultiVectorData, len(input))
	var totalTokens uint64
	for i, text := range input {
		g.Go(func() error {
			finalText, truncated, err := prepare(text)
			if err != nil {
				return err
			}

			result, promptTokens, err := r.MultiVector(ctx, finalText, llm.MultiVectorOptions{})
			if err != nil {
				return err
			}

			rows, dim, err := validateMultiVectorMatrix(result.Vectors)
			if err != nil {
				return err
			}

			data[i] = api.MultiVectorData{
				Index:     i,
				Shape:     []int{rows, dim},
				Vectors:   result.Vectors,
				Truncated: truncated,
			}
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
		Dimension:       dimension,
		Data:            data,
		TotalDuration:   time.Since(checkpointStart),
		LoadDuration:    checkpointLoaded.Sub(checkpointStart),
		PromptEvalCount: int(totalTokens),
	})
}
