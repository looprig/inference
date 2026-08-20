package geminiapi_test

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/inference/codec/geminiapi"
	model "github.com/looprig/inference/model"
)

// name builds a function name of exactly n legal characters.
func name(n int) string { return strings.Repeat("a", n) }

func float64Ptr(v float64) *float64 { return &v }

func imageModel() model.Model {
	return model.Model{Name: "m", Caps: model.Capabilities{AcceptsImages: true}}
}

// --- Function names --------------------------------------------------------
//
// Spec: v1beta discovery, FunctionDeclaration.name — "Required. The name of the
// function. Must be a-z, A-Z, 0-9, or contain underscores, colons, dots, and
// dashes, with a maximum length of 128."
//
// PROSE only: the derived request document types `name` as a bare string with
// no pattern and no maxLength, so the gate accepts every value below (measured,
// see TestTheGenerateContentGateHoldsNoneOfThis). Encoder-only.

func TestEncodeRequestRejectsIllegalFunctionNames(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		request inference.Request
		want    string
	}{
		{
			name: "declared function with a slash",
			request: inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
				Tools:    []inference.Tool{{Name: "server/tool", Description: "d"}},
			},
			want: "server/tool",
		},
		{
			name: "declared function with a space",
			request: inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
				Tools:    []inference.Tool{{Name: "read file", Description: "d"}},
			},
			want: "read file",
		},
		{
			name: "declared function one character over the 128-character cap",
			request: inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
				Tools:    []inference.Tool{{Name: name(129), Description: "d"}},
			},
			want: name(129),
		},
		{
			name: "declared function with an empty name",
			request: inference.Request{
				Model:    model.Model{Name: "m"},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
				Tools:    []inference.Tool{{Name: "", Description: "d"}},
			},
			want: "",
		},
		{
			name: "replayed functionCall naming an illegal function",
			request: inference.Request{
				Model: model.Model{Name: "m"},
				Messages: content.AgenticMessages{
					userMsg(textBlock("hi")),
					aiMsg(content.NewToolUseBlock("c1", "server/tool", json.RawMessage(`{}`), nil, "")),
				},
			},
			want: "server/tool",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body, err := geminiapi.EncodeRequest(tt.request)
			if err == nil {
				t.Fatalf("EncodeRequest() accepted a name FunctionDeclaration.name forbids; body = %s", body)
			}
			var invalid *geminiapi.InvalidToolNameError
			if !errors.As(err, &invalid) {
				t.Fatalf("EncodeRequest() error = %v (%T), want *geminiapi.InvalidToolNameError", err, err)
			}
			if invalid.Name != tt.want {
				t.Errorf("InvalidToolNameError.Name = %q, want %q", invalid.Name, tt.want)
			}
			if invalid.Reason == "" {
				t.Error("InvalidToolNameError.Reason is empty; a local error must name the violated constraint")
			}
		})
	}
}

// TestEncodeRequestAcceptsLegalFunctionNames is the positive control. The two
// interesting members are "." and ":", which Gemini's class admits and
// Anthropic's and Converse's do not — a rule copied from a sibling codec would
// reject exactly the namespaced MCP names this dialect is happy to take.
func TestEncodeRequestAcceptsLegalFunctionNames(t *testing.T) {
	t.Parallel()

	for _, fn := range []string{"read", "read_file", "read-file", "mcp.search", "server:tool", "a", name(128)} {
		t.Run(fn, func(t *testing.T) {
			t.Parallel()

			body, err := geminiapi.EncodeRequest(inference.Request{
				Model: model.Model{Name: "m"},
				Messages: content.AgenticMessages{
					userMsg(textBlock("hi")),
					aiMsg(content.NewToolUseBlock("c1", fn, json.RawMessage(`{}`), nil, "")),
					toolMsg("c1", textBlock("done")),
				},
				Tools:      []inference.Tool{{Name: fn, Description: "d"}},
				ToolChoice: inference.ToolNamed(fn),
			})
			if err != nil {
				t.Fatalf("EncodeRequest() rejected the legal function name %q: %v", fn, err)
			}
			conformance.MustValidateRequest(t, "gemini", kindGenerateContentRequest, body)
		})
	}
}

// TestEncodeRequestNamedToolChoiceInheritsTheRule pins why
// allowedFunctionNames is not re-checked: a forced choice must name a declared
// tool, and every declared tool is held to the class, so the choice cannot
// carry an illegal name the tools array did not carry first.
func TestEncodeRequestNamedToolChoiceInheritsTheRule(t *testing.T) {
	t.Parallel()

	_, err := geminiapi.EncodeRequest(inference.Request{
		Model:      model.Model{Name: "m"},
		Messages:   content.AgenticMessages{userMsg(textBlock("hi"))},
		Tools:      []inference.Tool{{Name: "read file", Description: "d"}},
		ToolChoice: inference.ToolNamed("read file"),
	})
	var invalid *geminiapi.InvalidToolNameError
	if !errors.As(err, &invalid) {
		t.Fatalf("EncodeRequest() error = %v (%T), want *geminiapi.InvalidToolNameError", err, err)
	}

	_, err = geminiapi.EncodeRequest(inference.Request{
		Model:      model.Model{Name: "m"},
		Messages:   content.AgenticMessages{userMsg(textBlock("hi"))},
		Tools:      []inference.Tool{{Name: "ok", Description: "d"}},
		ToolChoice: inference.ToolNamed("read file"),
	})
	var conflict *inference.StructuredOutputConflictError
	if !errors.As(err, &conflict) {
		t.Fatalf("EncodeRequest() error = %v (%T), want *inference.StructuredOutputConflictError; "+
			"an undeclared forced name is no longer refused upstream, so this codec must check it itself", err, err)
	}
}

// --- Sampling ranges -------------------------------------------------------
//
// Spec: v1beta discovery, GenerationConfig.temperature — "Values can range from
// [0.0, 2.0]". GenerationConfig.topP is "the maximum cumulative probability of
// tokens to consider when sampling", whose interval is [0, 1] by definition of
// a cumulative probability; Google declares no numeric range for it in either
// the v1beta or the Vertex document, so that bound is the weaker citation of
// the two and is recorded as such in encode.go.
//
// Neither is in the schema, so unlike OpenAI's the gate catches nothing here.

func TestEncodeRequestRejectsOutOfRangeSampling(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		sampling  model.Sampling
		wantField string
		wantValue float64
	}{
		{name: "temperature above 2", sampling: model.Sampling{Temperature: float64Ptr(2.5)}, wantField: "temperature", wantValue: 2.5},
		{name: "temperature below 0", sampling: model.Sampling{Temperature: float64Ptr(-0.5)}, wantField: "temperature", wantValue: -0.5},
		{name: "topP above 1", sampling: model.Sampling{TopP: float64Ptr(1.5)}, wantField: "topP", wantValue: 1.5},
		{name: "topP below 0", sampling: model.Sampling{TopP: float64Ptr(-1)}, wantField: "topP", wantValue: -1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body, err := geminiapi.EncodeRequest(inference.Request{
				Model:    model.Model{Name: "m", Sampling: tt.sampling},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
			})
			if err == nil {
				t.Fatalf("EncodeRequest() accepted %s = %v, which the dialect's range forbids; body = %s",
					tt.wantField, tt.wantValue, body)
			}
			var rangeErr *geminiapi.SamplingRangeError
			if !errors.As(err, &rangeErr) {
				t.Fatalf("EncodeRequest() error = %v (%T), want *geminiapi.SamplingRangeError", err, err)
			}
			if rangeErr.Field != tt.wantField || rangeErr.Value != tt.wantValue {
				t.Errorf("SamplingRangeError = {%q, %v}, want {%q, %v}",
					rangeErr.Field, rangeErr.Value, tt.wantField, tt.wantValue)
			}
		})
	}
}

// TestEncodeRequestAcceptsInRangeSampling is the positive control. temperature
// 0 is deliberately included: the Vertex flavour of this dialect states the
// range as (0.0, 2.0] — zero exclusive — and refusing it would reject the value
// callers set for determinism and that the canonical v1beta document permits.
func TestEncodeRequestAcceptsInRangeSampling(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		sampling model.Sampling
	}{
		{name: "unset", sampling: model.Sampling{}},
		{name: "temperature 0", sampling: model.Sampling{Temperature: float64Ptr(0)}},
		{name: "temperature 1", sampling: model.Sampling{Temperature: float64Ptr(1)}},
		{name: "temperature 2", sampling: model.Sampling{Temperature: float64Ptr(2)}},
		{name: "topP 0", sampling: model.Sampling{TopP: float64Ptr(0)}},
		{name: "topP 1", sampling: model.Sampling{TopP: float64Ptr(1)}},
		{name: "both at their maxima", sampling: model.Sampling{Temperature: float64Ptr(2), TopP: float64Ptr(1)}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			body, err := geminiapi.EncodeRequest(inference.Request{
				Model:    model.Model{Name: "m", Sampling: tt.sampling},
				Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
			})
			if err != nil {
				t.Fatalf("EncodeRequest() rejected in-range sampling: %v", err)
			}
			conformance.MustValidateRequest(t, "gemini", kindGenerateContentRequest, body)
		})
	}
}

func TestEncodeRequestSamplingOverrideIsValidatedToo(t *testing.T) {
	t.Parallel()

	override := model.Sampling{Temperature: float64Ptr(2.5)}
	_, err := geminiapi.EncodeRequest(inference.Request{
		Model:    model.Model{Name: "m", Sampling: model.Sampling{Temperature: float64Ptr(0.5)}},
		Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
		Override: &override,
	})
	var rangeErr *geminiapi.SamplingRangeError
	if !errors.As(err, &rangeErr) {
		t.Fatalf("EncodeRequest() error = %v (%T), want *geminiapi.SamplingRangeError", err, err)
	}
}

// --- Image media types -----------------------------------------------------
//
// Spec: v1beta discovery, Blob.mimeType — "Examples of supported types: -
// Images: image/png, image/jpeg, image/jpg, image/webp, image/heic, image/heif,
// image/gif, image/avif …". Audio and document types were already held to this
// list; images were not, and media.go said so in its own comment.

func TestEncodeRequestRejectsImageMediaTypesBlobDoesNotAccept(t *testing.T) {
	t.Parallel()

	for _, mediaType := range []content.MediaType{
		content.MediaTypeImageSVG, // a first-class member of the neutral vocabulary
		"image/tiff",
		"image/bmp",
		"image/*", // a wildcard is not a media type; Blob's list is closed for images
		"",
	} {
		t.Run(string(mediaType), func(t *testing.T) {
			t.Parallel()

			body, err := geminiapi.EncodeRequest(inference.Request{
				Model:    imageModel(),
				Messages: content.AgenticMessages{userMsg(imageDataBlock(mediaType, []byte("bytes")))},
			})
			if err == nil {
				t.Fatalf("EncodeRequest() accepted the media type %q, which Blob's Images list omits; body = %s",
					mediaType, body)
			}
			var unsupported *geminiapi.UnsupportedBlockError
			if !errors.As(err, &unsupported) {
				t.Fatalf("EncodeRequest() error = %v (%T), want *geminiapi.UnsupportedBlockError", err, err)
			}
			if !strings.Contains(unsupported.Reason, string(mediaType)) {
				t.Errorf("UnsupportedBlockError.Reason = %q, want it to name the media type %q",
					unsupported.Reason, mediaType)
			}
		})
	}
}

// TestEncodeRequestAcceptsEveryImageMediaTypeBlobLists is the positive control:
// the whole published list must still encode, including the non-IANA image/jpg
// spelling Google itself names.
func TestEncodeRequestAcceptsEveryImageMediaTypeBlobLists(t *testing.T) {
	t.Parallel()

	for _, mediaType := range []content.MediaType{
		"image/png", "image/jpeg", "image/jpg", "image/webp",
		"image/heic", "image/heif", "image/gif", "image/avif",
	} {
		t.Run(string(mediaType), func(t *testing.T) {
			t.Parallel()

			body, err := geminiapi.EncodeRequest(inference.Request{
				Model:    imageModel(),
				Messages: content.AgenticMessages{userMsg(imageDataBlock(mediaType, []byte("bytes")))},
			})
			if err != nil {
				t.Fatalf("EncodeRequest() rejected %q, which Blob's Images list names: %v", mediaType, err)
			}
			conformance.MustValidateRequest(t, "gemini", kindGenerateContentRequest, body)
		})
	}
}

// --- fileData.fileUri ------------------------------------------------------
//
// The encoder used to write ANY string into fileUri, so an arbitrary https://
// image URL produced a request that succeeded while the picture never reached
// the model — the silent drop of caller intent the module rule forbids. Gemini
// fetches only a Files API URI, a gs:// object, or a YouTube URL.

func TestEncodeRequestRejectsAFileURIGeminiWillNotFetch(t *testing.T) {
	t.Parallel()

	for _, uri := range []string{
		"https://example.com/x.jpg",
		"https://cdn.example.com/a/b/c.png",
		"http://generativelanguage.googleapis.com/v1beta/files/abc", // http, not https
		"https://generativelanguage.googleapis.com.evil.test/files/abc",
		"file:///etc/passwd",
		"gs://", // a scheme with no object
		"",      // no source at all: the block carries neither bytes nor URI
	} {
		t.Run(uri, func(t *testing.T) {
			t.Parallel()

			body, err := geminiapi.EncodeRequest(inference.Request{
				Model:    imageModel(),
				Messages: content.AgenticMessages{userMsg(imageURLBlock(uri))},
			})
			if err == nil {
				t.Fatalf("EncodeRequest() routed %q into fileUri, which Gemini does not fetch; body = %s", uri, body)
			}
			var unsupported *geminiapi.UnsupportedBlockError
			if !errors.As(err, &unsupported) {
				t.Fatalf("EncodeRequest() error = %v (%T), want *geminiapi.UnsupportedBlockError", err, err)
			}
			if unsupported.Reason == "" {
				t.Error("UnsupportedBlockError.Reason is empty; the error must name the limitation")
			}
		})
	}
}

// TestEncodeRequestAcceptsTheFileURIFormsGeminiFetches is the positive control.
func TestEncodeRequestAcceptsTheFileURIFormsGeminiFetches(t *testing.T) {
	t.Parallel()

	for _, uri := range []string{
		"https://generativelanguage.googleapis.com/v1beta/files/abc123",
		"gs://my-bucket/path/to/image.png",
		"https://www.youtube.com/watch?v=dQw4w9WgXcQ",
		"https://youtu.be/dQw4w9WgXcQ",
	} {
		t.Run(uri, func(t *testing.T) {
			t.Parallel()

			body, err := geminiapi.EncodeRequest(inference.Request{
				Model:    imageModel(),
				Messages: content.AgenticMessages{userMsg(imageURLBlock(uri))},
			})
			if err != nil {
				t.Fatalf("EncodeRequest() rejected %q, which fileUri accepts: %v", uri, err)
			}
			conformance.MustValidateRequest(t, "gemini", kindGenerateContentRequest, body)
		})
	}
}

// --- Gate strength ---------------------------------------------------------

// TestTheGenerateContentGateHoldsNoneOfThis measures what the conformance gate
// catches for this format instead of assuming it. The answer is: none of the
// constraints above. Gemini's derived request document declares required
// properties on 1 of 49 shapes, contains no oneOf at all, and types every field
// this file constrains as a plain string or number. Every rule here is carried
// by the encoder alone, and this test is the evidence for that claim.
//
// It is written to FAIL if a future schema refresh starts enforcing any of
// them, because at that point the encoder-only comments become wrong.
func TestTheGenerateContentGateHoldsNoneOfThis(t *testing.T) {
	t.Parallel()

	accepts := map[string]string{
		"temperature 3": `{"contents":[],"generationConfig":{"temperature":3}}`,
		"topP 5":        `{"contents":[],"generationConfig":{"topP":5}}`,
		"a function name with a space": `{"contents":[],` +
			`"tools":[{"functionDeclarations":[{"name":"bad name!","description":"d"}]}]}`,
		"a 200-character function name": `{"contents":[],` +
			`"tools":[{"functionDeclarations":[{"name":"` + name(200) + `","description":"d"}]}]}`,
		"an inlineData mime type the Blob description omits": `{"contents":[{"role":"user","parts":[` +
			`{"inlineData":{"mimeType":"image/tiff","data":"AAAA"}}]}]}`,
		"an inlineData mime type that is not a media type at all": `{"contents":[{"role":"user","parts":[` +
			`{"inlineData":{"mimeType":"not-a-mime","data":"AAAA"}}]}]}`,
		"an arbitrary web URL in fileUri": `{"contents":[{"role":"user","parts":[` +
			`{"fileData":{"mimeType":"image/png","fileUri":"https://example.com/x.png"}}]}]}`,
	}
	for label, body := range accepts {
		if err := conformance.Validate("gemini", kindGenerateContentRequest, []byte(body)); err != nil {
			t.Errorf("gate rejected %s: %v\n"+
				"the schema has started enforcing this; update the encoder-only comments", label, err)
		}
	}

	// What the gate DOES hold, so it is not mistaken for inert: types and
	// structure.
	rejects := map[string]string{
		"contents as an object rather than an array": `{"contents":{"role":"user"}}`,
		"a string temperature":                       `{"contents":[],"generationConfig":{"temperature":"hot"}}`,
	}
	for label, body := range rejects {
		if err := conformance.Validate("gemini", kindGenerateContentRequest, []byte(body)); err == nil {
			t.Errorf("gate accepted %s; it is not asserting at all", label)
		}
	}
}

// --- thinkingBudget vs maxOutputTokens -------------------------------------

// TestEncodeRequestKeepsTheEffortBudgetUnderASmallOutputCap pins a DELIBERATE
// non-decision: thinkingBudget is derived from Effort alone and is NOT clamped
// to, or validated against, maxOutputTokens.
//
// EffortHigh + MaxTokens 1024 therefore emits thinkingBudget 24576 against a
// 1024-token output cap, which looks wrong and is nevertheless the least
// surprising thing to send. The reasons, in order of weight:
//
//  1. NO Google document declares a constraint between the two. The v1beta
//     discovery document types thinkingBudget as a bare integer with no
//     minimum, no maximum and no cross-field rule; the thinking guides publish
//     only PER-MODEL budget ranges (2.5 Pro 128–32768, 2.5 Flash 1–24576,
//     2.5 Flash-Lite 512–24576) and say thinking tokens are BILLED as output
//     tokens. Nothing anywhere says budget must be <= maxOutputTokens, and
//     nothing says the request is rejected when it is not. Failing closed here
//     would invent a bound the provider never published and would reject
//     requests that today succeed.
//
//  2. Clamping is actively unsafe, which is the decisive point. The budget
//     floor is per-model and non-zero — 128 on 2.5 Pro, 512 on 2.5 Flash-Lite —
//     so clamping to a small cap turns a request the API accepts into one it
//     rejects: MaxTokens 100 would clamp 24576 down to 100, below Pro's
//     published minimum. Clamping also silently rewrites the caller's declared
//     effort, which is exactly the "never silently drop caller intent" failure.
//
//  3. The observed provider behaviour is truncation, not rejection: thinking
//     tokens are drawn from the same output allowance, and an over-budget
//     generation comes back with finishReason MAX_TOKENS rather than a 400.
//     A degraded-but-answered response is a worse thing to pre-empt with a
//     local hard failure than to let through.
//
// The hazard is real and is documented on thinkingFor in encode.go: a caller
// pairing a high effort with a tight output cap can get an empty or truncated
// answer whose entire allowance went to reasoning. That is a caller-visible
// tuning problem, not a malformed request, and this codec's job is the wire
// contract. If Google ever publishes a rule, this test is where it lands.
func TestEncodeRequestKeepsTheEffortBudgetUnderASmallOutputCap(t *testing.T) {
	t.Parallel()

	cap1024 := 1024
	req := inference.Request{
		Model: model.Model{
			Name:     "m",
			Caps:     model.Capabilities{Thinking: true},
			Sampling: model.Sampling{Effort: model.EffortHigh, MaxTokens: &cap1024},
		},
		Messages: content.AgenticMessages{userMsg(textBlock("hi"))},
	}
	body, err := geminiapi.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v; a budget above the output cap must not fail closed", err)
	}

	var raw struct {
		GenerationConfig struct {
			MaxOutputTokens *int `json:"maxOutputTokens"`
			ThinkingConfig  struct {
				ThinkingBudget *int `json:"thinkingBudget"`
			} `json:"thinkingConfig"`
		} `json:"generationConfig"`
	}
	if err := json.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal encoded request: %v", err)
	}
	got := raw.GenerationConfig.ThinkingConfig.ThinkingBudget
	if got == nil || *got != 24576 {
		t.Fatalf("thinkingBudget = %v, want 24576 (unclamped); clamping to maxOutputTokens would drop below "+
			"the per-model budget floor and rewrite the caller's declared effort", got)
	}
	if raw.GenerationConfig.MaxOutputTokens == nil || *raw.GenerationConfig.MaxOutputTokens != 1024 {
		t.Errorf("maxOutputTokens = %v, want 1024", raw.GenerationConfig.MaxOutputTokens)
	}

	// And the body the provider would receive is legal, which is the whole
	// argument for not rejecting it locally.
	conformance.MustValidateRequest(t, "gemini", kindGenerateContentRequest, body)
}

// TestTheGenerateContentGateAcceptsABudgetAboveTheOutputCap measures rather
// than assumes the other half: the derived request schema expresses no
// cross-field constraint, so a body pairing a 24576-token budget with a
// 1024-token cap validates. There is nothing here for the gate to catch even in
// principle, which is why the decision above had to be made from the prose
// documentation instead.
func TestTheGenerateContentGateAcceptsABudgetAboveTheOutputCap(t *testing.T) {
	t.Parallel()

	body := `{"contents":[],"generationConfig":{"maxOutputTokens":1024,` +
		`"thinkingConfig":{"thinkingBudget":24576,"includeThoughts":true}}}`
	if err := conformance.Validate("gemini", kindGenerateContentRequest, []byte(body)); err != nil {
		t.Errorf("gate rejected a budget above the output cap: %v\n"+
			"a cross-field rule has appeared in the schema; revisit thinkingFor's hazard note", err)
	}
}
