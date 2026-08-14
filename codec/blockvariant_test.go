package codec_test

import (
	"encoding/base64"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/core/content/blocktest"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/codec/bedrockconverse"
	"github.com/looprig/inference/codec/geminiapi"
	"github.com/looprig/inference/codec/openaiapi"
	"github.com/looprig/inference/codec/openairesponses"
	"github.com/looprig/inference/model"
)

// TestEveryCodecTriagesEverySealedBlockVariant is the sealed-union completeness
// guard for the encoders, and it exists here only because the fixture builder
// moved out of Harness's internal/ tree and into core, beside the union it
// mirrors. The same guard has protected the three copy paths inside Harness for
// a while; the codecs, which translate the same union onto four provider wires,
// could not import it at all.
//
// The contract asserted is the one CLAUDE.md states as "never silently drop
// caller intent", turned into something a new variant cannot escape:
//
//	for every content.Block variant core declares, every encoder either puts
//	the block on the wire, refuses the request, or drops the block by a
//	DECISION recorded in deliberateDrops below.
//
// The fourth outcome — accepted, encoded successfully, and quietly absent from
// both the body and the list — is the one that must be unreachable, because it
// is the only one nothing else would notice. A type switch in Go is not
// exhaustive, so a variant added to core reaches whatever the encoder's default
// arm does; if that arm ever becomes a bare `continue`, this test fails and the
// older per-field tests do not.
//
// blocktest.Blocks is a bijection with the union as content's own source
// declares it, so the variant list here cannot fall behind core either.
// Presence is detected through the fixtures' populated values: Populate assigns
// every string field the value "<TypeName>.<Field>", so a block that reached
// the body brings its own type name with it.
func TestEveryCodecTriagesEverySealedBlockVariant(t *testing.T) {
	t.Parallel()

	// deliberateDrops records the (codec, variant) pairs an encoder drops on
	// purpose, with the reason. Membership is the decision; a pair that is not
	// here and is not encoded and does not error is a defect. Keep the reason
	// current with the encoder's own comment — if they disagree, one of them is
	// out of date and the encoder is the one that ships.
	deliberateDrops := map[string]string{
		"openai/ThinkingBlock": "reasoning has no member in the chat-completions request body, " +
			"so an assistant thinking block has nowhere to go (encode.go, encodeConversation)",
		"openai-responses/ThinkingBlock": "a reasoning item must carry the exact provider state " +
			"OpenAI issued, including its id; a fixture with no replayable provider state is " +
			"omitted rather than emitted as a schema-invalid item (encode.go, encodeConversation)",
		"gemini/ThinkingBlock": "a thought part must carry the exact thoughtSignature Gemini itself " +
			"issued; a block whose ProviderState is absent or belongs to another dialect fails " +
			"ReplayableAs and is omitted rather than replayed with a fabricated signature " +
			"(encode.go, encodeAIParts)",
	}

	codecs := []struct {
		name   string
		encode func(inference.Request) ([]byte, error)
	}{
		{
			name:   "anthropic",
			encode: func(req inference.Request) ([]byte, error) { return anthropicapi.EncodeRequest(req, false) },
		},
		{
			name:   "openai",
			encode: func(req inference.Request) ([]byte, error) { return openaiapi.EncodeRequest(req, false) },
		},
		{name: "gemini", encode: geminiapi.EncodeRequest},
		{name: "bedrock-converse", encode: bedrockconverse.EncodeRequest},
		{
			name:   "openai-responses",
			encode: func(req inference.Request) ([]byte, error) { return openairesponses.EncodeRequest(req, false) },
		},
	}

	// A block's legal position depends on its variant — reasoning and tool
	// calls belong to an assistant turn, media to a user turn — so each variant
	// is offered in both positions and has to be triaged in each.
	positions := []struct {
		name     string
		messages func(content.Block) content.AgenticMessages
	}{
		{
			name: "user turn",
			messages: func(block content.Block) content.AgenticMessages {
				return content.AgenticMessages{
					&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{block}}},
				}
			},
		},
		{
			name: "assistant turn",
			messages: func(block content.Block) content.AgenticMessages {
				return content.AgenticMessages{
					&content.UserMessage{Message: content.Message{
						Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hello"}},
					}},
					&content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: []content.Block{block}}},
				}
			},
		},
	}

	for _, declaredBlock := range blocktest.Blocks(t) {
		variant := reflect.TypeOf(declaredBlock).Elem().Name()
		block, markers := representativeBlock(t, declaredBlock, variant+".")
		for _, position := range positions {
			messages := position.messages(block)
			for _, codec := range codecs {
				t.Run(variant+"/"+position.name+"/"+codec.name, func(t *testing.T) {
					t.Parallel()

					body, err := codec.encode(inference.Request{
						Model:    model.Model{Name: "m"},
						Messages: messages,
					})
					if err != nil {
						// An explicit refusal is a correct outcome: the caller
						// learns the block could not be represented instead of
						// receiving a body that quietly says less.
						return
					}
					if containsAny(string(body), markers) {
						return
					}
					reason, deliberate := deliberateDrops[codec.name+"/"+variant]
					if !deliberate {
						t.Fatalf("%s encoded a request with no trace of the %s it was given, and no error. "+
							"Either encode it, refuse it, or record the decision in deliberateDrops:\n%s",
							codec.name, variant, body)
					}
					t.Logf("%s drops %s on purpose: %s", codec.name, variant, reason)
				})
			}
		}
	}
}

// representativeBlock replaces blocktest's intentionally hostile, fully
// populated fixture with a provider-valid representative of the same sealed
// variant. Using the raw reflection fixture here made the guard vacuous for
// media: its deliberately invalid media type caused every encoder to refuse
// before reaching the block-dispatch path this test is meant to audit.
func representativeBlock(t *testing.T, declared content.Block, marker string) (content.Block, []string) {
	t.Helper()
	binaryMarker := base64.StdEncoding.EncodeToString([]byte(marker))
	markers := []string{marker, binaryMarker}
	switch declared.(type) {
	case *content.TextBlock:
		return &content.TextBlock{Text: marker}, markers
	case *content.ImageBlock:
		return &content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: []byte(marker)}}, markers
	case *content.AudioBlock:
		return &content.AudioBlock{MediaType: content.MediaTypeAudioWAV, Data: []byte(marker)}, markers
	case *content.DocumentBlock:
		return &content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report", Data: []byte(marker)}, markers
	case *content.ThinkingBlock:
		return &content.ThinkingBlock{Thinking: marker}, markers
	case *content.ToolUseBlock:
		return &content.ToolUseBlock{ID: "call_1", Name: "lookup", Input: json.RawMessage(`{"marker":"` + marker + `"}`)}, markers
	case *content.ToolResultBlock:
		return &content.ToolResultBlock{ToolUseID: "call_1", Content: []content.Block{&content.TextBlock{Text: marker}}, IsError: true}, markers
	case *content.RefusalBlock:
		return &content.RefusalBlock{Text: marker}, markers
	default:
		t.Fatalf("no representative fixture for sealed variant %T", declared)
		return nil, nil
	}
}

func containsAny(body string, markers []string) bool {
	for _, marker := range markers {
		if strings.Contains(body, marker) {
			return true
		}
	}
	return false
}
