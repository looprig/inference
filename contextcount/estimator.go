package contextcount

import (
	"context"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/geminiapi"
	"github.com/looprig/inference/codec/openaiapi"
)

const bytesPerEstimatedToken uint64 = 4

// EstimatorRevision pins both the bundled OpenAI/Anthropic/Gemini encoder suite
// and the complete-request bytes/4 heuristic. Any count-affecting codec change
// must bump this revision so durable measurements remain attributable.
const EstimatorRevision inference.TokenizerRevision = "bundled-openai-anthropic-gemini-request-bytes-div4-v1"

// Estimator deterministically estimates input occupancy from a dialect's encoded
// complete request. Its zero value is ready for use.
type Estimator struct{}

var _ inference.ContextCounter = (*Estimator)(nil)

// NewEstimator constructs a deterministic complete-request estimator.
func NewEstimator() *Estimator { return &Estimator{} }

// CountContext encodes the request in its model's API dialect and estimates one
// token per four encoded bytes. Invoke mode is canonical because ContextCounter
// has no response mode and streaming is response mechanics, not semantic input.
func (e *Estimator) CountContext(ctx context.Context, req inference.Request) (inference.ContextCount, error) {
	if e == nil {
		return inference.ContextCount{}, &EstimatorStateError{Reason: EstimatorStateNilReceiver}
	}
	if ctx == nil {
		return inference.ContextCount{}, &EstimatorStateError{Reason: EstimatorStateNilContext}
	}
	if err := ctx.Err(); err != nil {
		return inference.ContextCount{}, interruptedCount(req, err)
	}

	model := req.Model.Key()
	if err := model.Validate(); err != nil {
		return inference.ContextCount{}, &ModelIdentityError{Model: model, Err: err}
	}

	body, err := encodeRequest(req)
	if err != nil {
		return inference.ContextCount{}, err
	}
	// The bundled encoders are synchronous and do not accept a context, so they
	// cannot be interrupted mid-encode. Observe cancellation immediately after
	// that bounded local operation and before publishing its result.
	if err := ctx.Err(); err != nil {
		return inference.ContextCount{}, interruptedCount(req, err)
	}

	return inference.ContextCount{
		Model: model,
		// len returns a nonnegative int, whose value always fits uint64 on Go's
		// supported architectures.
		InputTokens: estimatedTokensForBytes(uint64(len(body))),
		Quality:     inference.CountQualityHeuristicEstimate,
	}, nil
}

func interruptedCount(req inference.Request, cause error) *inference.ContextCountError {
	return &inference.ContextCountError{
		Model:   req.Model.Key(),
		Quality: inference.CountQualityHeuristicEstimate,
		Cause:   cause,
	}
}

// CounterCapability declares that estimation stays in process, retains no
// request data, and is provider-neutral. A nil receiver returns invalid zero
// metadata rather than claiming a capability for unusable state.
func (e *Estimator) CounterCapability() inference.CounterCapability {
	if e == nil {
		return inference.CounterCapability{}
	}
	return inference.CounterCapability{
		Transport:    inference.CounterTransportLocal,
		Retention:    inference.RetentionNone,
		TokenizerRev: EstimatorRevision,
		Quality:      inference.CountQualityHeuristicEstimate,
	}
}

func encodeRequest(req inference.Request) ([]byte, error) {
	var (
		body []byte
		err  error
	)
	switch req.Model.APIFormat {
	case inference.APIFormatOpenAI:
		body, err = openaiapi.EncodeRequest(req, false)
	case inference.APIFormatAnthropic:
		body, err = anthropicapi.EncodeRequest(req, false)
	case inference.APIFormatGemini:
		body, err = geminiapi.EncodeRequest(req)
	default:
		return nil, &UnsupportedAPIFormatError{APIFormat: req.Model.APIFormat}
	}
	if err != nil {
		return nil, &RequestEncodingError{APIFormat: req.Model.APIFormat, Err: err}
	}
	return body, nil
}

func estimatedTokensForBytes(encodedBytes uint64) content.TokenCount {
	tokens := encodedBytes / bytesPerEstimatedToken
	if encodedBytes%bytesPerEstimatedToken != 0 {
		// tokens cannot be MaxUint64 here: division by four bounds it, so the
		// increment is safe even when encodedBytes is MaxUint64.
		tokens++
	}
	return content.TokenCount(tokens)
}
