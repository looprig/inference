package model

// ThinkingDialect names WHICH request shape a model accepts when reasoning is
// asked for. It is the companion of Capabilities.Thinking: the bool says the
// model can reason at all, the dialect says how the request has to spell it.
//
// One bool cannot carry both, and the difference is not cosmetic. Measured
// against api.anthropic.com on 2026-08-13, from a single run of one encoder:
// claude-haiku-4-5 answers `{"type":"adaptive"}` with HTTP 400 "adaptive
// thinking is not supported on this model", and claude-sonnet-5 answers
// `{"type":"enabled","budget_tokens":N}` with HTTP 400 "\"thinking.type.enabled\"
// is not supported for this model. Use \"thinking.type.adaptive\" and
// \"output_config.effort\"". Both wrong answers are hard rejections, so a codec
// choosing from one boolean has no safe default.
//
// It is deliberately a per-model capability rather than a per-format constant.
// The dialect a model accepts is server behaviour that changes with the model
// generation, not with the wire format: Anthropic's own API serves both
// spellings concurrently, and Bedrock Converse fronts the same models. It is
// also deliberately dialect-NEUTRAL vocabulary, like Effort — Gemini's
// thinkingConfig is a budget too, so "budget" describes a shape rather than one
// vendor's field name.
//
// The zero value is UNDECLARED, not a default. A catalogue that does not
// describe a model's dialect has said nothing, and a codec must fail closed
// with a diagnostic naming the model rather than guess between two spellings it
// can prove one of to be a 400.
type ThinkingDialect string

const (
	// ThinkingDialectUnknown is the zero value: the model's dialect has not
	// been declared. It is not a synonym for either real dialect.
	ThinkingDialectUnknown ThinkingDialect = ""

	// ThinkingDialectAdaptive marks a model that decides its own reasoning
	// depth and takes a coarse effort level. Anthropic spells it
	// `thinking:{"type":"adaptive"}` plus `output_config.effort`.
	ThinkingDialectAdaptive ThinkingDialect = "adaptive"

	// ThinkingDialectBudget marks a model that takes an explicit reasoning
	// token budget. Anthropic spells it
	// `thinking:{"type":"enabled","budget_tokens":N}`, and rejects
	// `output_config.effort` on the same models.
	ThinkingDialectBudget ThinkingDialect = "budget"
)

// Valid reports whether d is a known dialect. The empty value is valid here in
// the same sense Effort's is — it is a legal FIELD value meaning "unset" — and
// what a consumer does with an unset dialect is that consumer's rule, not
// Validate's.
func (d ThinkingDialect) Valid() bool {
	switch d {
	case ThinkingDialectUnknown, ThinkingDialectAdaptive, ThinkingDialectBudget:
		return true
	default:
		return false
	}
}
