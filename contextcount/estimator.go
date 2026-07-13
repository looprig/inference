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

// EstimatorRevision pins the complete-request byte heuristic used by Estimator.
const EstimatorRevision inference.TokenizerRevision = "complete-request-bytes-div4-v1"

// Estimator deterministically estimates input occupancy from a dialect's encoded
// complete request. Its zero value is ready for use.
type Estimator struct{}

var _ inference.ContextCounter = (*Estimator)(nil)

// NewEstimator constructs a deterministic complete-request estimator.
func NewEstimator() *Estimator { return &Estimator{} }

// CountContext encodes the request in its model's API dialect and estimates one
// token per four encoded bytes. Invoke mode is canonical because ContextCounter
// has no response mode and streaming is response mechanics, not semantic input.
func (e *Estimator) CountContext(_ context.Context, req inference.Request) (inference.ContextCount, error) {
	if e == nil {
		return inference.ContextCount{}, &EstimatorStateError{Reason: EstimatorStateNilReceiver}
	}

	model := req.Model.Key()
	if err := model.Validate(); err != nil {
		return inference.ContextCount{}, &ModelIdentityError{Model: model, Err: err}
	}

	body, err := encodeRequest(req)
	if err != nil {
		return inference.ContextCount{}, err
	}

	return inference.ContextCount{
		Model: model,
		// len returns a nonnegative int, whose value always fits uint64 on Go's
		// supported architectures.
		InputTokens: estimatedTokensForBytes(uint64(len(body))),
		Quality:     inference.CountQualityHeuristicEstimate,
	}, nil
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
