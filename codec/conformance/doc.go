// Package conformance is the schema conformance gate for provider wire
// fixtures. It answers one question, on every test run, before a fixture is
// allowed anywhere near a Looprig decoder: is this byte string actually a legal
// payload for the provider API it claims to come from?
//
// The reason is epistemic rather than defensive. A fixture is only evidence
// about Looprig's correctness if the fixture itself is something the provider
// could really have sent. A decoder test that feeds itself an invented payload
// proves a property of a fiction. The gate makes that failure mode impossible
// to reach silently: MustValidate fails the test, loudly and with the exact
// violating instance path, rather than letting a malformed fixture through.
//
// # Where the gate lives, and why here
//
// It lives in the inference module, beside the codecs, and it is exported.
//
// It used to live in llm, one tier up, behind an internal/. That put it out of
// reach of exactly the tests it matters most to: inference/CLAUDE.md requires
// that "every encode path must hold its encoded body against the format's
// official request schema in tests", and the encode paths are here. The gate
// could only be applied by the provider clients that compose these codecs,
// which is a tier late — a provider test that rejects a body has already lost
// the ability to say which encoder produced it. Moving it down means the codec
// that builds a body can validate that body, and llm keeps every call site it
// had.
//
// # Requests and responses
//
// Kinds come in both directions, and the request half is the more valuable of
// the two. A response schema tells us whether we understood the provider; a
// request schema tells us whether the provider will understand us, and it says
// so before the bytes leave the process rather than after a 400. Request
// schemas are also the stricter half of every specification: they carry the
// required lists and the additionalProperties:false closures that response
// schemas routinely omit, which is exactly where an encoder bug hides. Anthropic
// closes 83 of the 84 object shapes in its request body and 5 of the 49 in its
// response.
//
// Use MustValidateRequest on an encoded request body and MustValidateResponse
// on a received payload. Non-test callers use ValidateRequest and
// ValidateResponse. All four take the bytes on the wire and refuse a kind whose
// direction does not match, because an encoded request held against a response
// schema would appear to pass while proving nothing.
//
// # What backs the gate
//
// The schema/ tree holds standalone JSON Schema 2020-12 documents derived from
// the four official upstream API descriptions — OpenAI's own openapi.yaml,
// the OpenAPI document Anthropic's own SDK points at, Google's production
// discovery endpoint, and AWS's own Smithy model distribution. They are
// checked in as real files, never fetched at test time, and every reference is
// rebased into a local $defs so each document validates offline and on its own.
// schema/provenance.json records where each source came from, when, and its
// SHA-256; schema/unenforced.json records, per document, everything the
// derived schema does NOT enforce. Regenerate with cmd/schemagen; never edit
// the tree by hand.
//
// # What the gate does not catch
//
// The honest limits, all enumerated per document in schema/unenforced.json:
//
//   - Unknown properties are allowed wherever the specification allows them.
//     The gate never adds a closure of its own — providers add response fields
//     without notice, and a closure we invented would reject tomorrow's legal
//     payload. Where the specification closes an object itself the closure is
//     kept, because the provider really will reject the extra field. That is
//     overwhelmingly a request-side phenomenon (Anthropic closes 83 of 84
//     request shapes and 5 of 49 response shapes). Everywhere else a payload
//     may carry a field that does not exist.
//   - Some unions are relaxed from oneOf to anyOf. Both OpenAI documents offer
//     overlapping branches in places — an ordinary Responses input message
//     matches both EasyInputMessage and Item — where oneOf would reject a
//     payload the live API accepts. Only unions with a demonstrated overlap are
//     allowlisted for relaxation, each is listed in schema/unenforced.json, and
//     every discriminated union keeps its oneOf, so an undefined variant is
//     still caught.
//   - "format" is an annotation, not an assertion. A malformed date-time or
//     uri passes. This matches the draft 2020-12 default and is measured
//     against the official suite in testdata/format-assertion-support.json.
//   - Semantic coherence is out of scope. Index continuity across streaming
//     deltas, token counts that add up, ids that match between frames — the
//     schema sees each payload in isolation.
//   - Google's discovery format declares almost no required properties, in
//     either direction, so the gemini documents constrain types, enums and
//     nesting but cannot catch a missing field.
//   - AWS marks only modelId as required on ConverseRequest, and modelId
//     travels in the URI path rather than the body. The Bedrock request
//     document consequently requires nothing at the top level; a body with no
//     messages passes.
//   - Where OpenAPI 3.0's "nullable" survives in a 3.1 document, the gate
//     widens the affected schema to admit null. That under-constrains those
//     positions rather than falsely rejecting a legal null.
//
// The gate's strength is exactly: structure, required properties, types,
// enums, and string/number/array constraints, as the provider itself
// publishes them.
//
// # Trusting the validator
//
// Assertions are delegated to github.com/santhosh-tekuri/jsonschema/v6. There
// is no first-party Go implementation to prefer: the JSON Schema organisation
// publishes a language-agnostic conformance suite and no Go validator, so every
// Go option is third-party and hand-rolling one would defeat the point of a
// correctness gate. Instead of trusting the library, suite_test.go runs the
// organisation's official draft2020-12 suite against it from a checked-in copy
// on every run. See that file for the required/optional split.
//
// github.com/google/jsonschema-go is the one Go validator with an institutional
// owner, and it is already in this workspace as an indirect dependency of the
// MCP Go SDK, so "consolidate on it" is the obvious question to ask. It was
// measured rather than argued, against the criteria this package actually cares
// about:
//
//   - Official required draft2020-12 suite: v0.4.3 fails 3 of 1299 cases, all in
//     vocabulary.json ($vocabulary and custom metaschemas). santhosh-tekuri
//     passes all 46 files with no skips. It also omits content.json, format.json
//     and vocabulary.json from its own vendored copy of the suite, so those
//     keywords are not self-verified upstream.
//   - Our schemas: all 14 provider documents load and resolve under both.
//   - Our verdicts: a differential over the checked-in provider fixture corpus,
//     every fixture and SSE frame against every (format, kind) gate, agreed on
//     6664 of 6664 comparisons.
//
// So the swap is behaviourally free on today's workload and costs the gate's one
// headline guarantee — 46/46, zero skips — for a pre-1.0 API. None of the three
// failures touch a keyword our schemas use, so this is a judgement call and not
// a defect finding; revisit it if google/jsonschema-go reaches 1.0 with the
// vocabulary cases passing, at which point the institutional-owner and
// one-fewer-distinct-dependency arguments carry it.
package conformance
