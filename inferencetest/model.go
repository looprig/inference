package inferencetest

import (
	"github.com/looprig/core/content"
	"github.com/looprig/inference/model"
)

const (
	// ProviderName is the provider label Model reports.
	ProviderName model.ProviderName = "inferencetest"
	// APIFormat is the API-format label Model reports. No codec speaks it: a
	// Model is meant to be paired with a Client from this package.
	APIFormat model.APIFormat = "inferencetest"
	// ModelName is the model name Model reports.
	ModelName = "scripted"
	// ContextWindow is the context window Model advertises.
	ContextWindow content.TokenCount = 200_000
)

// Model returns a descriptor for a scripted Client. It validates, and it
// declares the capabilities an agent runtime gates on — tools, image input and
// native structured output, with and without tools — so a composition built
// around it takes the same paths it would with a capable provider model. It
// advertises a ContextWindow-token window and no endpoint.
//
// Options apply after those defaults, so model.WithThinking or
// model.WithContextLimits can adjust them.
func Model(opts ...model.ModelOption) model.Model {
	m := model.CustomModel(ProviderName, APIFormat, "", ModelName,
		model.WithImages(),
		model.WithStructuredOutputWithTools(),
		model.WithContextLimits(model.ContextLimits{WindowTokens: ContextWindow}),
	)
	for _, opt := range opts {
		opt(&m)
	}
	return m
}
