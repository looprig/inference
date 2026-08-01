package bedrockconverse_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/bedrockconverse"
	"github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

func TestDecodeResponse_ContentUsageAndFinishReason(t *testing.T) {
	t.Parallel()

	body := []byte(`{
		"output":{"message":{"role":"assistant","content":[
			{"text":"answer"},
			{"reasoningContent":{"reasoningText":{"text":"think","signature":"sig"}}},
			{"toolUse":{"toolUseId":"call-1","name":"lookup","input":{"query":"go"}}}
		]}},
		"stopReason":"tool_use",
		"usage":{"inputTokens":10,"outputTokens":8,"cacheReadInputTokens":2,"cacheWriteInputTokens":1},
		"additionalModelResponseFields":{"ignored":true}
	}`)
	response, err := bedrockconverse.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if response.Message == nil || len(response.Message.Blocks) != 3 {
		t.Fatalf("Message = %#v, want three blocks", response.Message)
	}
	if got := response.Message.Blocks[0].(*content.TextBlock).Text; got != "answer" {
		t.Errorf("text = %q, want answer", got)
	}
	thinking, ok := response.Message.Blocks[1].(*content.ThinkingBlock)
	if !ok {
		t.Fatalf("block[1] = %T, want ThinkingBlock", response.Message.Blocks[1])
	}
	if thinking.Thinking != "think" || thinking.Signature != "sig" {
		t.Errorf("thinking = %#v, want text/signature", thinking)
	}
	toolUse, ok := response.Message.Blocks[2].(*content.ToolUseBlock)
	if !ok {
		t.Fatalf("block[2] = %T, want ToolUseBlock", response.Message.Blocks[2])
	}
	if toolUse.ID != "call-1" || toolUse.Name != "lookup" || string(toolUse.Input) != `{"query":"go"}` {
		t.Errorf("tool use = %#v, want decoded call", toolUse)
	}
	wantUsage := &content.Usage{InputTokens: 10, OutputTokens: 8, CacheReadTokens: 2, CacheCreationTokens: 1}
	if *response.Usage != *wantUsage || *response.Message.Usage != *wantUsage {
		t.Errorf("usage = %#v/message usage = %#v, want %#v", response.Usage, response.Message.Usage, wantUsage)
	}
	if response.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("FinishReason = %q, want tool_use", response.FinishReason)
	}
}

func TestDecodeResponse_FinishReasons(t *testing.T) {
	t.Parallel()

	cases := []struct {
		wire string
		want stream.FinishReason
	}{
		{wire: "end_turn", want: stream.FinishReasonStop},
		{wire: "stop_sequence", want: stream.FinishReasonStop},
		{wire: "max_tokens", want: stream.FinishReasonLength},
		{wire: "tool_use", want: stream.FinishReasonToolUse},
		{wire: "content_filtered", want: stream.FinishReasonContentFilter},
		{wire: "guardrail_intervened", want: stream.FinishReasonContentFilter},
		{wire: "future_reason", want: stream.FinishReasonUnknown},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.wire, func(t *testing.T) {
			t.Parallel()
			body := []byte(`{"output":{"message":{"role":"assistant","content":[]}},"stopReason":"` + tc.wire + `"}`)
			response, err := bedrockconverse.DecodeResponse(body)
			if err != nil {
				t.Fatalf("DecodeResponse() error = %v", err)
			}
			if response.FinishReason != tc.want {
				t.Errorf("FinishReason = %q, want %q", response.FinishReason, tc.want)
			}
		})
	}
}

func TestDecodeResponse_UsageOptionalAndAdditionalFieldsIgnored(t *testing.T) {
	t.Parallel()

	response, err := bedrockconverse.DecodeResponse([]byte(`{"output":{"message":{"content":[{"text":"ok"}]}},"stopReason":"end_turn","additionalModelResponseFields":{"vendor":{"value":1}}}`))
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if response.Usage != nil || response.Message.Usage != nil {
		t.Fatalf("usage = %#v/message usage = %#v, want both nil", response.Usage, response.Message.Usage)
	}
}

func TestDecodeResponse_MalformedAndMissingFields(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		body string
	}{
		{name: "malformed JSON", body: `{"output":`},
		{name: "missing output", body: `{"stopReason":"end_turn"}`},
		{name: "missing message", body: `{"output":{},"stopReason":"end_turn"}`},
		{name: "invalid tool input", body: `{"output":{"message":{"content":[{"toolUse":{"toolUseId":"id","name":"tool","input":[]}}]}},"stopReason":"tool_use"}`},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := bedrockconverse.DecodeResponse([]byte(tc.body))
			if err == nil {
				t.Fatal("DecodeResponse() error = nil, want error")
			}
			var decodeErr *bedrockconverse.DecodeError
			if !errors.As(err, &decodeErr) {
				t.Fatalf("error = %T (%v), want *DecodeError", err, err)
			}
		})
	}
}

func TestDecodeResponse_InvalidUsageCounts(t *testing.T) {
	t.Parallel()

	cases := []string{"-1", "1.5", `"1"`, "null"}
	for _, raw := range cases {
		raw := raw
		t.Run(raw, func(t *testing.T) {
			t.Parallel()
			body := []byte(`{"output":{"message":{"content":[]}},"usage":{"inputTokens":` + raw + `}}`)
			_, err := bedrockconverse.DecodeResponse(body)
			if err == nil {
				t.Fatal("DecodeResponse() error = nil, want usage normalization error")
			}
			var normalizationErr *usage.UsageNormalizationError
			if !errors.As(err, &normalizationErr) {
				t.Fatalf("error = %T (%v), want UsageNormalizationError", err, err)
			}
		})
	}
}

func TestDecodeResponse_ToolResultAndMultimodalBlocks(t *testing.T) {
	t.Parallel()

	body := []byte(`{"output":{"message":{"content":[
		{"toolResult":{"toolUseId":"call-1","status":"error","content":[{"text":"failed"}]}},
		{"image":{"format":"png","source":{"bytes":"AQI="}}},
		{"document":{"format":"txt","name":"notes.txt","source":{"text":"notes"}}}
	]}},"stopReason":"end_turn"}`)
	response, err := bedrockconverse.DecodeResponse(body)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if len(response.Message.Blocks) != 3 {
		t.Fatalf("blocks = %#v, want 3", response.Message.Blocks)
	}
	result, ok := response.Message.Blocks[0].(*content.ToolResultBlock)
	if !ok || !result.IsError || result.ToolUseID != "call-1" {
		t.Fatalf("tool result = %#v, want error result", response.Message.Blocks[0])
	}
	image, ok := response.Message.Blocks[1].(*content.ImageBlock)
	if !ok || string(image.Source.Data) != string([]byte{1, 2}) || image.MediaType != content.MediaType("image/png") {
		t.Fatalf("image = %#v, want decoded PNG bytes", response.Message.Blocks[1])
	}
	document, ok := response.Message.Blocks[2].(*content.DocumentBlock)
	if !ok || document.Text != "notes" || document.Name != "notes.txt" {
		t.Fatalf("document = %#v, want text document", response.Message.Blocks[2])
	}
}

func TestCodecDecodeResponseDelegates(t *testing.T) {
	t.Parallel()
	response, err := (bedrockconverse.Codec{}).DecodeResponse([]byte(`{"output":{"message":{"content":[]}},"stopReason":"end_turn"}`))
	if err != nil {
		t.Fatalf("Codec.DecodeResponse() error = %v", err)
	}
	if response.Message == nil {
		t.Fatal("Codec.DecodeResponse() returned nil Message")
	}
}

func TestDecodeResponse_DoesNotExposeMalformedBodyInTypedError(t *testing.T) {
	t.Parallel()
	secret := "do-not-leak-this-payload"
	_, err := bedrockconverse.DecodeResponse([]byte(`{"output":{"message":{"content":[{"toolUse":{"input":"` + secret + `"}}]}}}`))
	if err == nil {
		t.Fatal("DecodeResponse() error = nil, want error")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("error exposes provider payload: %v", err)
	}
}
