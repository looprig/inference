package conformance

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// These tests prove the gate on a representative fixture per api-format. The
// full provider fixture suite lives with the codecs it exercises; what is
// proven here is that the gate accepts real provider payloads (this file) and
// rejects payloads that could not have come off the wire (negative_test.go).

// fixture reads a checked-in fixture. Nothing is fetched, generated or
// templated: the bytes in testdata are the bytes validated.
func fixture(t testing.TB, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name)) // #nosec G304 -- fixed, checked-in fixture path
	if err != nil {
		t.Fatalf("ReadFile(testdata/%s) error = %v", name, err)
	}
	return raw
}

func TestGateAcceptsRealProviderResponses(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		format  string
		kind    string
		fixture string
	}{
		{"openai chat completion", "openai", "chat_completion", "openai_chat_completion.json"},
		{"openai responses object", "openai-responses", "response", "openai_responses_response.json"},
		{"anthropic message", "anthropic", "message", "anthropic_message.json"},
		{"gemini generate content", "gemini", "generate_content_response", "gemini_generate_content_response.json"},
		{"bedrock converse", "bedrock-converse", "converse_response", "bedrock_converse_response.json"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			MustValidate(t, tc.format, tc.kind, fixture(t, tc.fixture))
		})
	}
}

// TestGateAcceptsRealProviderRequests validates an encoded request body per
// api-format. Each fixture deliberately carries image and document inputs,
// because those are the shapes with the most encoder surface: nested source
// unions, base64 payloads, media-type enums and, on the Anthropic side, closed
// objects that reject an undeclared field outright.
func TestGateAcceptsRealProviderRequests(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		format  string
		kind    string
		fixture string
	}{
		{"openai chat completion request", "openai", "chat_completion_request", "openai_chat_completion_request.json"},
		{"openai create response request", "openai-responses", "create_response_request", "openai_responses_create_response_request.json"},
		{"anthropic create message request", "anthropic", "create_message_request", "anthropic_create_message_request.json"},
		{"gemini generate content request", "gemini", "generate_content_request", "gemini_generate_content_request.json"},
		{"bedrock converse request", "bedrock-converse", "converse_request", "bedrock_converse_request.json"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			MustValidateRequest(t, tc.format, tc.kind, fixture(t, tc.fixture))
		})
	}
}

// TestDirectionMismatchIsRejected proves the request and response entry points
// are not interchangeable. Holding an encoded request against a response schema
// would frequently appear to pass — both are open objects over similar names —
// while proving nothing at all.
func TestDirectionMismatchIsRejected(t *testing.T) {
	t.Parallel()

	stub := &stubTB{TB: t}
	stub.run(func() {
		MustValidateRequest(stub, "anthropic", "message", fixture(t, "anthropic_message.json"))
	})
	if !stub.failed {
		t.Fatal("MustValidateRequest accepted a response kind")
	}
	if !strings.Contains(stub.message, "is a response kind") {
		t.Fatalf("diagnostic does not name the direction mismatch:\n%s", stub.message)
	}

	stub = &stubTB{TB: t}
	stub.run(func() {
		MustValidateResponse(stub, "anthropic", "create_message_request", fixture(t, "anthropic_create_message_request.json"))
	})
	if !stub.failed {
		t.Fatal("MustValidateResponse accepted a request kind")
	}
	if !strings.Contains(stub.message, "is a request kind") {
		t.Fatalf("diagnostic does not name the direction mismatch:\n%s", stub.message)
	}
}

// TestEveryFormatHasBothDirections keeps the request half from being quietly
// dropped: it is the half that catches our own encoder bugs.
func TestEveryFormatHasBothDirections(t *testing.T) {
	t.Parallel()

	for _, format := range Formats() {
		var requests, responses int
		for _, kind := range Kinds(format) {
			entry, _ := Lookup(format, kind)
			switch entry.Direction {
			case DirectionRequest:
				requests++
			case DirectionResponse:
				responses++
			default:
				t.Errorf("%s/%s has direction %q, want request or response", format, kind, entry.Direction)
			}
		}
		if requests == 0 {
			t.Errorf("api-format %q has no request kind", format)
		}
		if responses == 0 {
			t.Errorf("api-format %q has no response kind", format)
		}
	}
}

func TestGateAcceptsRealProviderStreams(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		format  string
		kind    string
		fixture string
		frames  int
	}{
		{"openai chat completion chunks", "openai", "chat_completion_chunk", "openai_chat_completion_stream.sse", 4},
		{"openai responses events", "openai-responses", "stream_event", "openai_responses_stream.sse", 3},
		{"anthropic message events", "anthropic", "stream_event", "anthropic_message_stream.sse", 6},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := MustValidateStream(t, tc.format, tc.kind, fixture(t, tc.fixture)); got != tc.frames {
				t.Fatalf("MustValidateStream(%s/%s) validated %d frames, want %d", tc.format, tc.kind, got, tc.frames)
			}
		})
	}
}

// TestGateAcceptsBedrockEventStreamFrames validates the Converse stream the way
// it actually arrives. Bedrock does not use server-sent events: each frame is a
// separate event-stream message whose body is one member of the output union,
// so the fixture is a list of union-shaped frames rather than an SSE body.
func TestGateAcceptsBedrockEventStreamFrames(t *testing.T) {
	t.Parallel()

	var frames []json.RawMessage
	if err := json.Unmarshal(fixture(t, "bedrock_converse_stream.json"), &frames); err != nil {
		t.Fatalf("decode bedrock stream fixture: %v", err)
	}
	if len(frames) != 7 {
		t.Fatalf("bedrock stream fixture has %d frames, want 7", len(frames))
	}
	for _, frame := range frames {
		MustValidate(t, "bedrock-converse", "converse_stream_output", frame)
	}
}

// TestGateValidatesAnUnwrappedUnionMember covers the kind/member form, which a
// decoder test needs when it holds an event body that has already been
// unwrapped from its envelope.
func TestGateValidatesAnUnwrappedUnionMember(t *testing.T) {
	t.Parallel()

	MustValidate(t, "bedrock-converse", "converse_stream_output/contentBlockDelta",
		[]byte(`{"contentBlockIndex":0,"delta":{"text":"The forecast"}}`))
	MustValidate(t, "anthropic", "stream_event/message_stop", []byte(`{"type":"message_stop"}`))
}

// TestIndexCoversEveryDeclaredFormat pins the gate's coverage, so removing a
// format from the generator is a test failure rather than a silent gap.
func TestIndexCoversEveryDeclaredFormat(t *testing.T) {
	t.Parallel()

	want := map[string][]string{
		"openai":           {"chat_completion_request", "chat_completion", "chat_completion_chunk"},
		"openai-responses": {"create_response_request", "response", "stream_event"},
		"anthropic":        {"create_message_request", "message", "stream_event"},
		"gemini":           {"generate_content_request", "generate_content_response"},
		"bedrock-converse": {"converse_request", "converse_response", "converse_stream_output"},
	}

	for _, format := range Formats() {
		if _, ok := want[format]; !ok {
			t.Errorf("index has undeclared api-format %q", format)
		}
	}
	for format, kinds := range want {
		for _, kind := range kinds {
			entry, ok := Lookup(format, kind)
			if !ok {
				t.Errorf("index is missing %s/%s", format, kind)
				continue
			}
			if entry.Document == "" || entry.Root == "" {
				t.Errorf("index entry %s/%s is incomplete: %+v", format, kind, entry)
			}
		}
	}
}

// TestUnionsAreFullyDiscriminated asserts that every union kind can name the
// member a payload claims to be. Without that the gate still rejects bad
// payloads, but the report degenerates into every branch's failure at once.
func TestUnionsAreFullyDiscriminated(t *testing.T) {
	t.Parallel()

	cases := []struct {
		format string
		kind   string
		style  string
		least  int
	}{
		{"openai-responses", "stream_event", UnionStyleProperty, 50},
		{"anthropic", "stream_event", UnionStyleProperty, 6},
		{"bedrock-converse", "converse_stream_output", UnionStyleMemberKey, 6},
	}

	for _, tc := range cases {
		entry, ok := Lookup(tc.format, tc.kind)
		if !ok || entry.Union == nil {
			t.Errorf("%s/%s has no union in the index", tc.format, tc.kind)
			continue
		}
		if entry.Union.Style != tc.style {
			t.Errorf("%s/%s union style = %q, want %q", tc.format, tc.kind, entry.Union.Style, tc.style)
		}
		if len(entry.Union.Members) < tc.least {
			t.Errorf("%s/%s has %d discriminable members, want at least %d",
				tc.format, tc.kind, len(entry.Union.Members), tc.least)
		}
		if len(entry.Union.Ambiguous) != 0 {
			t.Errorf("%s/%s has ambiguous discriminator values %v; payloads using them fall back to the whole union",
				tc.format, tc.kind, entry.Union.Ambiguous)
		}
	}
}

// TestEverySchemaCompiles loads every embedded document at its root. A document
// that only compiles when a particular kind happens to be exercised would leave
// a latent generation bug in the tree.
func TestEverySchemaCompiles(t *testing.T) {
	t.Parallel()

	registry := load()
	if registry.err != nil {
		t.Fatalf("registry error = %v", registry.err)
	}
	count := 0
	for _, format := range Formats() {
		for _, kind := range Kinds(format) {
			entry, _ := Lookup(format, kind)
			if _, err := registry.schemaFor(entry.Document, entry.Root); err != nil {
				t.Errorf("%s/%s: %v", format, kind, err)
				continue
			}
			count++
			if entry.Union == nil {
				continue
			}
			for member, pointer := range entry.Union.Members {
				if _, err := registry.schemaFor(entry.Document, pointer); err != nil {
					t.Errorf("%s/%s member %q: %v", format, kind, member, err)
				}
			}
		}
	}
	if count == 0 {
		t.Fatal("no schemas compiled")
	}
	t.Logf("compiled %d schema entry points", count)
}

// TestParseSSERejectsGarbage covers the stream splitter directly: a stream
// fixture that is not a stream must fail rather than validate zero frames.
func TestParseSSERejectsGarbage(t *testing.T) {
	t.Parallel()

	if _, err := ParseSSE([]byte("{\"type\":\"message_stop\"}\n")); err == nil {
		t.Fatal("ParseSSE(bare JSON) = nil error, want a rejection")
	}
	if _, err := ParseSSE([]byte("event: message_stop\n\n")); err == nil {
		t.Fatal("ParseSSE(named event with no data) = nil error, want a rejection")
	}

	// A comment, reconnection metadata, one real frame and the OpenAI
	// terminator: everything a provider stream actually contains.
	frames, err := ParseSSE([]byte(": keep-alive\n\nretry: 3000\n\ndata: {\"a\":1}\n\ndata: [DONE]\n\n"))
	if err != nil {
		t.Fatalf("ParseSSE(valid stream) error = %v", err)
	}
	if len(frames) != 1 {
		t.Fatalf("ParseSSE(valid stream) returned %d frames, want 1", len(frames))
	}
	if string(frames[0].Data) != `{"a":1}` {
		t.Fatalf("ParseSSE data = %q, want the frame payload", frames[0].Data)
	}
}
