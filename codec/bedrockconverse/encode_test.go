package bedrockconverse_test

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	codec "github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/bedrockconverse"
	model "github.com/looprig/inference/model"
)

func decodeObject(t *testing.T, raw []byte) map[string]json.RawMessage {
	t.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil {
		t.Fatalf("json.Unmarshal() error = %v; body = %s", err, raw)
	}
	return object
}

func decodeArray(t *testing.T, raw json.RawMessage) []map[string]json.RawMessage {
	t.Helper()
	var values []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &values); err != nil {
		t.Fatalf("json.Unmarshal(array) error = %v; raw = %s", err, raw)
	}
	return values
}

func asString(t *testing.T, raw json.RawMessage) string {
	t.Helper()
	var value string
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatalf("json.Unmarshal(string) error = %v; raw = %s", err, raw)
	}
	return value
}

func baseModel() model.Model {
	return model.Model{
		Provider:  model.ProviderName("bedrock"),
		APIFormat: model.APIFormatBedrockConverse,
		BaseURL:   "https://bedrock-runtime.us-east-1.amazonaws.com",
		Name:      "anthropic.claude-sonnet-4-20250514-v1:0",
	}
}

func userMessage(blocks ...content.Block) *content.UserMessage {
	return &content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: blocks}}
}

func assistantMessage(blocks ...content.Block) *content.AIMessage {
	return &content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: blocks}}
}

func systemMessage(blocks ...content.Block) *content.SystemMessage {
	return &content.SystemMessage{Message: content.Message{Role: content.RoleSystem, Blocks: blocks}}
}

func toolResultMessage(id string, isError bool, blocks ...content.Block) *content.ToolResultMessage {
	return &content.ToolResultMessage{
		Message:   content.Message{Role: content.RoleTool, Blocks: blocks},
		ToolUseID: id,
		IsError:   isError,
	}
}

func TestEncodeRequest_SystemAndMessages(t *testing.T) {
	t.Parallel()

	req := inference.Request{
		Model:  baseModel(),
		System: "base instruction",
		Messages: content.AgenticMessages{
			systemMessage(&content.TextBlock{Text: "thread instruction"}),
			userMessage(&content.TextBlock{Text: "hello"}),
			assistantMessage(&content.TextBlock{Text: "hi"}),
		},
	}
	raw, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	body := decodeObject(t, raw)
	if _, ok := body["model"]; ok {
		t.Fatal("request body contains model; Bedrock takes model ID in the URL")
	}
	systems := decodeArray(t, body["system"])
	if len(systems) != 2 || asString(t, systems[0]["text"]) != "base instruction" || asString(t, systems[1]["text"]) != "thread instruction" {
		t.Fatalf("system = %#v, want two ordered text blocks", systems)
	}
	messages := decodeArray(t, body["messages"])
	if len(messages) != 2 || asString(t, messages[0]["role"]) != "user" || asString(t, messages[1]["role"]) != "assistant" {
		t.Fatalf("messages = %#v, want user then assistant", messages)
	}
	if got := asString(t, decodeArray(t, messages[0]["content"])[0]["text"]); got != "hello" {
		t.Errorf("user text = %q, want hello", got)
	}
}

func TestEncodeRequest_MergesAdjacentOrdinaryUserTurns(t *testing.T) {
	t.Parallel()

	req := inference.Request{Model: baseModel(), Messages: content.AgenticMessages{
		userMessage(&content.TextBlock{Text: "first"}),
		userMessage(&content.TextBlock{Text: "second"}),
	}}
	raw, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	messages := decodeArray(t, decodeObject(t, raw)["messages"])
	if len(messages) != 1 {
		t.Fatalf("len(messages) = %d, want 1", len(messages))
	}
	blocks := decodeArray(t, messages[0]["content"])
	if len(blocks) != 2 || asString(t, blocks[0]["text"]) != "first" || asString(t, blocks[1]["text"]) != "second" {
		t.Fatalf("content = %#v, want both user blocks in order", blocks)
	}
}

func TestEncodeRequest_MergesParallelToolResults(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.Tools = true
	req := inference.Request{Model: m, Messages: content.AgenticMessages{
		toolResultMessage("call-1", false, &content.TextBlock{Text: "one"}),
		toolResultMessage("call-2", false, &content.TextBlock{Text: "two"}),
	}}
	raw, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	messages := decodeArray(t, decodeObject(t, raw)["messages"])
	if len(messages) != 1 {
		t.Fatalf("len(messages) = %d, want 1", len(messages))
	}
	blocks := decodeArray(t, messages[0]["content"])
	if len(blocks) != 2 || blocks[0]["toolResult"] == nil || blocks[1]["toolResult"] == nil {
		t.Fatalf("content = %#v, want two toolResult blocks", blocks)
	}
}

func TestEncodeRequest_FoldsTransientRuntimeTextIntoFinalToolResult(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.Tools = true
	req := inference.Request{
		Model: m,
		Messages: content.AgenticMessages{
			toolResultMessage("call-1", false, &content.TextBlock{Text: "tool bytes"}),
			userMessage(&content.TextBlock{Text: "runtime\nbytes"}, &content.TextBlock{Text: "second runtime block"}),
		},
		TransientMessages: 1,
	}
	raw, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	messages := decodeArray(t, decodeObject(t, raw)["messages"])
	if len(messages) != 1 {
		t.Fatalf("len(messages) = %d, want 1", len(messages))
	}
	blocks := decodeArray(t, messages[0]["content"])
	if len(blocks) != 1 || blocks[0]["text"] != nil {
		t.Fatalf("content = %#v, want one pure toolResult block", blocks)
	}
	var result map[string]json.RawMessage
	if err := json.Unmarshal(blocks[0]["toolResult"], &result); err != nil {
		t.Fatalf("toolResult: %v", err)
	}
	resultContent := decodeArray(t, result["content"])
	if len(resultContent) != 3 {
		t.Fatalf("len(toolResult.content) = %d, want 3", len(resultContent))
	}
	want := []string{"tool bytes", "runtime\nbytes", "second runtime block"}
	for i, text := range want {
		if got := asString(t, resultContent[i]["text"]); got != text {
			t.Errorf("toolResult.content[%d].text = %q, want %q", i, got, text)
		}
	}
}

func TestEncodeRequest_RejectsUserToolResultCollision(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.Tools = true
	m.Caps.AcceptsImages = true
	cases := []struct {
		name      string
		messages  content.AgenticMessages
		transient int
	}{
		{
			// A COMMITTED user turn after tool results is Looprig's own fold
			// feature and must encode (see the separator tests below). Only the
			// transient runtime suffix folds into the final tool result, and that
			// fold is text-only: a non-text transient block still fails closed
			// rather than being dropped.
			name: "non-text transient user after tool result",
			messages: content.AgenticMessages{
				toolResultMessage("call-1", false, &content.TextBlock{Text: "tool"}),
				userMessage(&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: []byte("png")}}),
			},
			transient: 1,
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := bedrockconverse.EncodeRequest(inference.Request{Model: m, Messages: tc.messages, TransientMessages: tc.transient})
			var collision *bedrockconverse.ConversationCollisionError
			if !errors.As(err, &collision) {
				t.Fatalf("EncodeRequest() error = %T (%v), want ConversationCollisionError", err, err)
			}
		})
	}
}

// messageRoles reports the role of every projected Converse message, so a test can
// assert the whole alternation in one comparison.
func messageRoles(t *testing.T, messages []map[string]json.RawMessage) []string {
	t.Helper()
	roles := make([]string, len(messages))
	for i, message := range messages {
		roles[i] = asString(t, message["role"])
	}
	return roles
}

// messageTexts reports the text of every content block of one projected message,
// failing if any block is not a plain text block.
func messageTexts(t *testing.T, message map[string]json.RawMessage) []string {
	t.Helper()
	blocks := decodeArray(t, message["content"])
	texts := make([]string, 0, len(blocks))
	for i, block := range blocks {
		if block["text"] == nil {
			t.Fatalf("content[%d] = %#v, want a text block", i, block)
		}
		texts = append(texts, asString(t, block["text"]))
	}
	return texts
}

// toolResultTexts reports the text of every block inside the single toolResult of a
// projected tool-result message.
func toolResultTexts(t *testing.T, message map[string]json.RawMessage) []string {
	t.Helper()
	blocks := decodeArray(t, message["content"])
	if len(blocks) != 1 || blocks[0]["toolResult"] == nil {
		t.Fatalf("content = %#v, want exactly one toolResult block", blocks)
	}
	var result map[string]json.RawMessage
	if err := json.Unmarshal(blocks[0]["toolResult"], &result); err != nil {
		t.Fatalf("toolResult: %v", err)
	}
	inner := decodeArray(t, result["content"])
	texts := make([]string, 0, len(inner))
	for i, block := range inner {
		if block["text"] == nil {
			t.Fatalf("toolResult.content[%d] = %#v, want a text block", i, block)
		}
		texts = append(texts, asString(t, block["text"]))
	}
	return texts
}

// assertInsertedSeparator fails unless the message is an assistant turn whose only
// content is one text block naming Looprig as its author. The naming is the contract:
// this turn is not model output, and a human reading the request body must be able to
// see that Looprig put it there.
func assertInsertedSeparator(t *testing.T, message map[string]json.RawMessage) {
	t.Helper()
	if role := asString(t, message["role"]); role != "assistant" {
		t.Fatalf("separator role = %q, want assistant", role)
	}
	texts := messageTexts(t, message)
	if len(texts) != 1 {
		t.Fatalf("separator has %d content blocks, want 1", len(texts))
	}
	if !strings.Contains(texts[0], "Looprig") {
		t.Errorf("separator text = %q, want it to name Looprig as the inserter", texts[0])
	}
}

// TestEncodeRequest_SeparatesCommittedUserTurnFromToolResults pins the projection of
// Looprig's fold feature: a user message committed to history while a tool call was
// still running lands after the tool results, and Converse cannot carry it as-is.
// The projector inserts an assistant separator turn rather than failing, because the
// message is already in stored history and a failure here wedges the session forever.
func TestEncodeRequest_SeparatesCommittedUserTurnFromToolResults(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.Tools = true
	m.Caps.AcceptsImages = true

	toolLoop := func(extra ...content.Conversation) content.AgenticMessages {
		messages := content.AgenticMessages{
			userMessage(&content.TextBlock{Text: "start"}),
			assistantMessage(&content.ToolUseBlock{ID: "call-1", Name: "read", Input: []byte(`{"path":"a"}`)}),
			toolResultMessage("call-1", false, &content.TextBlock{Text: "tool bytes"}),
		}
		return append(messages, extra...)
	}

	t.Run("one folded message", func(t *testing.T) {
		t.Parallel()
		raw, err := bedrockconverse.EncodeRequest(inference.Request{Model: m, Messages: toolLoop(
			userMessage(&content.TextBlock{Text: "actually, stop"}),
		)})
		if err != nil {
			t.Fatalf("EncodeRequest() error = %v", err)
		}
		messages := decodeArray(t, decodeObject(t, raw)["messages"])
		want := []string{"user", "assistant", "user", "assistant", "user"}
		if got := messageRoles(t, messages); !reflect.DeepEqual(got, want) {
			t.Fatalf("roles = %v, want %v", got, want)
		}
		// The tool-result turn is byte-identical to the one already sent before the
		// user typed: the separator is appended after it, never merged into it, so a
		// cached committed prefix stays reusable.
		if got := toolResultTexts(t, messages[2]); !reflect.DeepEqual(got, []string{"tool bytes"}) {
			t.Errorf("toolResult.content = %v, want the tool's own bytes only", got)
		}
		assertInsertedSeparator(t, messages[3])
		if got := messageTexts(t, messages[4]); !reflect.DeepEqual(got, []string{"actually, stop"}) {
			t.Errorf("folded user turn = %v, want the user's own words as a user turn", got)
		}
	})

	t.Run("several folded messages share one separator", func(t *testing.T) {
		t.Parallel()
		raw, err := bedrockconverse.EncodeRequest(inference.Request{Model: m, Messages: toolLoop(
			userMessage(&content.TextBlock{Text: "first"}),
			userMessage(&content.TextBlock{Text: "second"}),
		)})
		if err != nil {
			t.Fatalf("EncodeRequest() error = %v", err)
		}
		messages := decodeArray(t, decodeObject(t, raw)["messages"])
		want := []string{"user", "assistant", "user", "assistant", "user"}
		if got := messageRoles(t, messages); !reflect.DeepEqual(got, want) {
			t.Fatalf("roles = %v, want %v", got, want)
		}
		assertInsertedSeparator(t, messages[3])
		if got := messageTexts(t, messages[4]); !reflect.DeepEqual(got, []string{"first", "second"}) {
			t.Errorf("folded user turn = %v, want both folded messages merged into one user turn", got)
		}
	})

	t.Run("folded message carrying an image", func(t *testing.T) {
		t.Parallel()
		raw, err := bedrockconverse.EncodeRequest(inference.Request{Model: m, Messages: toolLoop(
			userMessage(
				&content.TextBlock{Text: "look at this"},
				&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: []byte("png")}},
			),
		)})
		if err != nil {
			t.Fatalf("EncodeRequest() error = %v", err)
		}
		messages := decodeArray(t, decodeObject(t, raw)["messages"])
		if len(messages) != 5 {
			t.Fatalf("len(messages) = %d, want 5", len(messages))
		}
		blocks := decodeArray(t, messages[4]["content"])
		if len(blocks) != 2 || blocks[0]["text"] == nil || blocks[1]["image"] == nil {
			t.Fatalf("folded user content = %#v, want text + image", blocks)
		}
	})

	t.Run("transient runtime tail joins the folded user turn", func(t *testing.T) {
		t.Parallel()
		raw, err := bedrockconverse.EncodeRequest(inference.Request{
			Model: m,
			Messages: toolLoop(
				userMessage(&content.TextBlock{Text: "actually, stop"}),
				userMessage(&content.TextBlock{Text: "runtime context"}),
			),
			TransientMessages: 1,
		})
		if err != nil {
			t.Fatalf("EncodeRequest() error = %v", err)
		}
		messages := decodeArray(t, decodeObject(t, raw)["messages"])
		want := []string{"user", "assistant", "user", "assistant", "user"}
		if got := messageRoles(t, messages); !reflect.DeepEqual(got, want) {
			t.Fatalf("roles = %v, want %v", got, want)
		}
		// Exactly one separator, and the runtime suffix rides the ordinary user turn
		// rather than being folded into a tool result it no longer follows.
		assertInsertedSeparator(t, messages[3])
		if got := toolResultTexts(t, messages[2]); !reflect.DeepEqual(got, []string{"tool bytes"}) {
			t.Errorf("toolResult.content = %v, want the tool's own bytes only", got)
		}
		if got := messageTexts(t, messages[4]); !reflect.DeepEqual(got, []string{"actually, stop", "runtime context"}) {
			t.Errorf("final user turn = %v, want the folded message then the runtime suffix", got)
		}
	})

	t.Run("model reply after the folded turn keeps alternating", func(t *testing.T) {
		t.Parallel()
		raw, err := bedrockconverse.EncodeRequest(inference.Request{Model: m, Messages: toolLoop(
			userMessage(&content.TextBlock{Text: "actually, stop"}),
			assistantMessage(&content.TextBlock{Text: "stopping"}),
			userMessage(&content.TextBlock{Text: "thanks"}),
		)})
		if err != nil {
			t.Fatalf("EncodeRequest() error = %v", err)
		}
		messages := decodeArray(t, decodeObject(t, raw)["messages"])
		want := []string{"user", "assistant", "user", "assistant", "user", "assistant", "user"}
		if got := messageRoles(t, messages); !reflect.DeepEqual(got, want) {
			t.Fatalf("roles = %v, want %v", got, want)
		}
		assertInsertedSeparator(t, messages[3])
		if got := messageTexts(t, messages[5]); !reflect.DeepEqual(got, []string{"stopping"}) {
			t.Errorf("assistant reply = %v, want the model's own reply as its own turn", got)
		}
	})
}

func TestEncodeCountTokensInput_UsesSameConversationProjection(t *testing.T) {
	t.Parallel()

	req := inference.Request{Model: baseModel(), Messages: content.AgenticMessages{
		userMessage(&content.TextBlock{Text: "first"}),
		userMessage(&content.TextBlock{Text: "second"}),
	}}
	requestRaw, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	countRaw, err := bedrockconverse.EncodeCountTokensInput(req)
	if err != nil {
		t.Fatalf("EncodeCountTokensInput() error = %v", err)
	}
	requestMessages := decodeObject(t, requestRaw)["messages"]
	countMessages := decodeObject(t, countRaw)["messages"]
	if string(requestMessages) != string(countMessages) {
		t.Fatalf("count messages = %s, request messages = %s", countMessages, requestMessages)
	}
}

func TestEncodeRequest_ImagesAndDocuments(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.AcceptsImages = true
	req := inference.Request{
		Model: m,
		Messages: content.AgenticMessages{userMessage(
			&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: []byte{1, 2, 3}}},
			&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report-pdf", Data: []byte("pdf-bytes")},
			&content.DocumentBlock{MediaType: content.MediaTypeDocumentMarkdown, Name: "notes-md", Text: "# Notes"},
			&content.TextBlock{Text: "The attached documents contain the details."},
		)},
	}
	raw, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	messages := decodeArray(t, decodeObject(t, raw)["messages"])
	blocks := decodeArray(t, messages[0]["content"])
	var image map[string]json.RawMessage
	if err := json.Unmarshal(blocks[0]["image"], &image); err != nil {
		t.Fatalf("image block: %v", err)
	}
	if got := asString(t, image["format"]); got != "png" {
		t.Errorf("image.format = %q, want png", got)
	}
	var imageSource map[string]json.RawMessage
	if err := json.Unmarshal(image["source"], &imageSource); err != nil {
		t.Fatalf("image source: %v", err)
	}
	if got := asString(t, imageSource["bytes"]); got != base64.StdEncoding.EncodeToString([]byte{1, 2, 3}) {
		t.Errorf("image.source.bytes = %q, want base64 data", got)
	}

	var document map[string]json.RawMessage
	if err := json.Unmarshal(blocks[1]["document"], &document); err != nil {
		t.Fatalf("document block: %v", err)
	}
	if got := asString(t, document["format"]); got != "pdf" {
		t.Errorf("document.format = %q, want pdf", got)
	}
	if got := asString(t, document["name"]); got != "report-pdf" {
		t.Errorf("document.name = %q, want report-pdf", got)
	}
	var documentSource map[string]json.RawMessage
	if err := json.Unmarshal(document["source"], &documentSource); err != nil {
		t.Fatalf("document source: %v", err)
	}
	if got := asString(t, documentSource["bytes"]); got != base64.StdEncoding.EncodeToString([]byte("pdf-bytes")) {
		t.Errorf("document.source.bytes = %q, want base64 data", got)
	}

	var textDocument map[string]json.RawMessage
	if err := json.Unmarshal(blocks[2]["document"], &textDocument); err != nil {
		t.Fatalf("text document block: %v", err)
	}
	var textSource map[string]json.RawMessage
	if err := json.Unmarshal(textDocument["source"], &textSource); err != nil {
		t.Fatalf("text document source: %v", err)
	}
	if got := asString(t, textSource["text"]); got != "# Notes" {
		t.Errorf("document.source.text = %q, want # Notes", got)
	}
}

func TestEncodeRequest_ReasoningToolsAndToolResult(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.Thinking = true
	m.Caps.Tools = true
	req := inference.Request{
		Model: m,
		Messages: content.AgenticMessages{
			assistantMessage(
				content.NewSignedThinkingBlock("consider", "sig", signatureFormatBedrockConverse, nil, ""),
				&content.ToolUseBlock{ID: "call-1", Name: "lookup", Input: json.RawMessage(`{"query":"go"}`)},
			),
			toolResultMessage("call-1", true, &content.TextBlock{Text: "not found"}),
		},
		Tools:      []inference.Tool{{Name: "lookup", Description: "Find a thing", Schema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}`)}},
		ToolChoice: inference.ToolRequired(),
	}
	raw, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	body := decodeObject(t, raw)
	messages := decodeArray(t, body["messages"])
	assistantBlocks := decodeArray(t, messages[0]["content"])
	var reasoning map[string]json.RawMessage
	if err := json.Unmarshal(assistantBlocks[0]["reasoningContent"], &reasoning); err != nil {
		t.Fatalf("reasoningContent: %v", err)
	}
	var reasoningText map[string]json.RawMessage
	if err := json.Unmarshal(reasoning["reasoningText"], &reasoningText); err != nil {
		t.Fatalf("reasoningText: %v", err)
	}
	if got := asString(t, reasoningText["text"]); got != "consider" {
		t.Errorf("reasoning text = %q, want consider", got)
	}
	if got := asString(t, reasoningText["signature"]); got != "sig" {
		t.Errorf("reasoning signature = %q, want sig", got)
	}
	var toolUse map[string]json.RawMessage
	if err := json.Unmarshal(assistantBlocks[1]["toolUse"], &toolUse); err != nil {
		t.Fatalf("toolUse: %v", err)
	}
	if got := asString(t, toolUse["toolUseId"]); got != "call-1" {
		t.Errorf("toolUse.toolUseId = %q, want call-1", got)
	}

	resultBlocks := decodeArray(t, messages[1]["content"])
	var toolResult map[string]json.RawMessage
	if err := json.Unmarshal(resultBlocks[0]["toolResult"], &toolResult); err != nil {
		t.Fatalf("toolResult: %v", err)
	}
	if got := asString(t, toolResult["status"]); got != "error" {
		t.Errorf("toolResult.status = %q, want error", got)
	}

	var toolConfig map[string]json.RawMessage
	if err := json.Unmarshal(body["toolConfig"], &toolConfig); err != nil {
		t.Fatalf("toolConfig: %v", err)
	}
	choices := decodeArray(t, toolConfig["tools"])
	var spec map[string]json.RawMessage
	if err := json.Unmarshal(choices[0]["toolSpec"], &spec); err != nil {
		t.Fatalf("toolSpec: %v", err)
	}
	if got := asString(t, spec["name"]); got != "lookup" {
		t.Errorf("toolSpec.name = %q, want lookup", got)
	}
	var inputSchema map[string]json.RawMessage
	if err := json.Unmarshal(spec["inputSchema"], &inputSchema); err != nil {
		t.Fatalf("toolSpec.inputSchema: %v", err)
	}
	var schema map[string]json.RawMessage
	if err := json.Unmarshal(inputSchema["json"], &schema); err != nil {
		t.Fatalf("toolSpec.inputSchema.json: %v", err)
	}
	if got := asString(t, schema["type"]); got != "object" {
		t.Errorf("tool schema type = %q, want object", got)
	}
	var choice map[string]json.RawMessage
	if err := json.Unmarshal(toolConfig["toolChoice"], &choice); err != nil {
		t.Fatalf("toolChoice: %v", err)
	}
	if _, ok := choice["any"]; !ok {
		t.Errorf("toolChoice = %#v, want any", choice)
	}
}

// TestEncodeRequest_ToolChoiceNamedTool pins the named tool-choice variant.
// Converse's ToolChoice is a Smithy union, so the member key IS the
// discriminator: SpecificToolChoice arrives as {"tool":{"name":...}} with name
// @required.
func TestEncodeRequest_ToolChoiceNamedTool(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.Tools = true
	raw, err := bedrockconverse.EncodeRequest(inference.Request{
		Model:    m,
		Messages: content.AgenticMessages{userMessage(&content.TextBlock{Text: "hi"})},
		Tools: []inference.Tool{
			{Name: "lookup", Schema: json.RawMessage(`{"type":"object"}`)},
			{Name: "search", Schema: json.RawMessage(`{"type":"object"}`)},
		},
		ToolChoice: inference.ToolNamed("search"),
	})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	toolConfig := decodeObject(t, decodeObject(t, raw)["toolConfig"])
	const want = `{"tool":{"name":"search"}}`
	if got := string(toolConfig["toolChoice"]); got != want {
		t.Errorf("toolChoice = %s, want %s", got, want)
	}
}

func TestEncodeRequest_ToolResultStatusAndContentRules(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.Tools = true
	cases := []struct {
		name       string
		blocks     []content.Block
		wantStatus string
		wantError  bool
	}{
		{name: "successful status omitted", blocks: []content.Block{&content.TextBlock{Text: "ok"}}},
		{name: "document-only result accepted", blocks: []content.Block{&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "result-pdf", Data: []byte("pdf")}}},
		{name: "empty content rejected", wantError: true},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := inference.Request{Model: m, Messages: content.AgenticMessages{toolResultMessage("call-1", false, tc.blocks...)}}
			raw, err := bedrockconverse.EncodeRequest(req)
			if tc.wantError {
				var encodeErr *bedrockconverse.EncodeError
				if !errors.As(err, &encodeErr) {
					t.Fatalf("EncodeRequest() error = %T (%v), want EncodeError", err, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("EncodeRequest() error = %v", err)
			}
			body := decodeObject(t, raw)
			messages := decodeArray(t, body["messages"])
			blocks := decodeArray(t, messages[0]["content"])
			var result map[string]json.RawMessage
			if err := json.Unmarshal(blocks[0]["toolResult"], &result); err != nil {
				t.Fatalf("toolResult = %v", err)
			}
			if _, ok := result["status"]; ok {
				t.Fatalf("successful toolResult has status = %#v, want status omitted", result["status"])
			}
		})
	}
}

func TestEncodeRequest_SamplingAndStructuredOutput(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.StructuredOutput = true
	sampling := model.Sampling{MaxTokens: intPtr(321), Temperature: floatPtr(0.2), TopP: floatPtr(0.8), Stop: []string{"END"}}
	req := inference.Request{
		Model:    m,
		Override: &sampling,
		Output: &inference.OutputSchema{
			Name:        "answer",
			Description: "The answer object",
			Schema:      json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}`),
		},
	}
	raw, err := bedrockconverse.EncodeRequest(req)
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	body := decodeObject(t, raw)
	var inferenceConfig map[string]json.RawMessage
	if err := json.Unmarshal(body["inferenceConfig"], &inferenceConfig); err != nil {
		t.Fatalf("inferenceConfig: %v", err)
	}
	if got := asInt(t, inferenceConfig["maxTokens"]); got != 321 {
		t.Errorf("maxTokens = %d, want 321", got)
	}
	if got := asFloat(t, inferenceConfig["temperature"]); got != 0.2 {
		t.Errorf("temperature = %v, want 0.2", got)
	}
	if got := asFloat(t, inferenceConfig["topP"]); got != 0.8 {
		t.Errorf("topP = %v, want 0.8", got)
	}
	var stops []string
	if err := json.Unmarshal(inferenceConfig["stopSequences"], &stops); err != nil {
		t.Fatalf("stopSequences: %v", err)
	}
	if len(stops) != 1 || stops[0] != "END" {
		t.Fatalf("stopSequences = %#v, want [END]", stops)
	}

	var outputConfig map[string]json.RawMessage
	if err := json.Unmarshal(body["outputConfig"], &outputConfig); err != nil {
		t.Fatalf("outputConfig: %v", err)
	}
	var textFormat map[string]json.RawMessage
	if err := json.Unmarshal(outputConfig["textFormat"], &textFormat); err != nil {
		t.Fatalf("textFormat: %v", err)
	}
	if got := asString(t, textFormat["type"]); got != "json_schema" {
		t.Errorf("textFormat.type = %q, want json_schema", got)
	}
	var structure map[string]json.RawMessage
	if err := json.Unmarshal(textFormat["structure"], &structure); err != nil {
		t.Fatalf("structure: %v", err)
	}
	var schema map[string]json.RawMessage
	if err := json.Unmarshal(structure["jsonSchema"], &schema); err != nil {
		t.Fatalf("jsonSchema: %v", err)
	}
	if got := asString(t, schema["name"]); got != "answer" {
		t.Errorf("jsonSchema.name = %q, want answer", got)
	}
	if got := asString(t, schema["schema"]); got != string(req.Output.Schema) {
		t.Errorf("jsonSchema.schema = %q, want original schema string", got)
	}
}

func TestEncodeRequest_RejectsReasoningEffort(t *testing.T) {
	t.Parallel()
	m := baseModel()
	m.Caps.Thinking = true
	m.Sampling.Effort = model.EffortHigh
	_, err := bedrockconverse.EncodeRequest(inference.Request{Model: m})
	var effortErr *bedrockconverse.UnsupportedEffortError
	if !errors.As(err, &effortErr) {
		t.Fatalf("EncodeRequest() error = %T (%v), want *UnsupportedEffortError", err, err)
	}
}

func TestEncodeRequest_OmitsUnrequestedOptionalFields(t *testing.T) {
	t.Parallel()

	raw, err := bedrockconverse.EncodeRequest(inference.Request{Model: baseModel()})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	body := decodeObject(t, raw)
	if len(body) != 1 {
		t.Fatalf("body keys = %#v, want only messages", body)
	}
	if _, ok := body["messages"]; !ok {
		t.Fatal("messages field omitted")
	}
}

func TestEncodeCountTokensInput_UsesOnlyConverseCountableFields(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.Tools = true
	m.Caps.StructuredOutput = true
	m.Caps.StructuredOutputWithTools = true
	maxTokens := 128
	req := inference.Request{
		Model:    m,
		System:   "count this system",
		Messages: content.AgenticMessages{userMessage(&content.TextBlock{Text: "hello"})},
		Tools:    []inference.Tool{{Name: "lookup", Schema: json.RawMessage(`{"type":"object","additionalProperties":false}`)}},
		Override: &model.Sampling{MaxTokens: &maxTokens, Temperature: floatPtr(0.1)},
		Output: &inference.OutputSchema{
			Name:   "answer",
			Schema: json.RawMessage(`{"type":"object","additionalProperties":false}`),
		},
	}
	raw, err := bedrockconverse.EncodeCountTokensInput(req)
	if err != nil {
		t.Fatalf("EncodeCountTokensInput() error = %v", err)
	}
	body := decodeObject(t, raw)
	for _, field := range []string{"messages", "system", "toolConfig"} {
		if _, ok := body[field]; !ok {
			t.Errorf("count input missing %q", field)
		}
	}
	for _, field := range []string{"inferenceConfig", "outputConfig", "model", "guardrailConfig", "serviceTier"} {
		if _, ok := body[field]; ok {
			t.Errorf("count input unexpectedly includes %q", field)
		}
	}
}

func TestCodecEncodeRequest_HeadersAndMode(t *testing.T) {
	t.Parallel()

	encoded, err := (bedrockconverse.Codec{}).EncodeRequest(inference.Request{Model: baseModel()}, codec.RequestModeStream)
	if err != nil {
		t.Fatalf("Codec.EncodeRequest() error = %v", err)
	}
	if got := encoded.Header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", got)
	}
	body, err := io.ReadAll(encoded.Body)
	if err != nil {
		t.Fatalf("ReadAll() error = %v", err)
	}
	if strings.Contains(string(body), "stream") {
		t.Errorf("Converse request unexpectedly contains stream flag: %s", body)
	}
}

func TestEncodeRequest_RejectsUnsupportedAndMalformedTools(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.AcceptsImages = true
	cases := []struct {
		name string
		req  inference.Request
	}{
		{
			// Audio is NOT here: Converse's ContentBlock union declares an
			// `audio` member, so an audio block encodes. Its real limits — the
			// AudioFormat allowlist, the minimum payload length and the
			// toolResult union that has no audio member — are in
			// encode_audio_test.go.
			name: "remote image",
			req:  inference.Request{Model: m, Messages: content.AgenticMessages{userMessage(&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{URL: "https://example.test/image.png"}})}},
		},
		{
			name: "empty schema",
			req:  inference.Request{Model: m, Tools: []inference.Tool{{Name: "tool"}}},
		},
		{
			name: "malformed schema",
			req:  inference.Request{Model: m, Tools: []inference.Tool{{Name: "tool", Schema: json.RawMessage(`{"type":`)}}},
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := bedrockconverse.EncodeRequest(tc.req)
			if err == nil {
				t.Fatal("EncodeRequest() error = nil, want typed validation error")
			}
			var unsupported *bedrockconverse.UnsupportedBlockError
			var schemaErr *bedrockconverse.ToolSchemaError
			if tc.name == "audio" || tc.name == "remote image" {
				if !errors.As(err, &unsupported) {
					t.Fatalf("error = %T (%v), want UnsupportedBlockError", err, err)
				}
			} else if !errors.As(err, &schemaErr) {
				t.Fatalf("error = %T (%v), want ToolSchemaError", err, err)
			}
		})
	}
}

func TestEncodeRequest_EnforcesBedrockMessageRolesAndDocumentRules(t *testing.T) {
	t.Parallel()

	m := baseModel()
	m.Caps.AcceptsImages = true
	m.Caps.Thinking = true
	m.Caps.Tools = true
	cases := []struct {
		name string
		req  inference.Request
	}{
		{
			name: "assistant image",
			req:  inference.Request{Model: m, Messages: content.AgenticMessages{assistantMessage(&content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: []byte{1}}})}},
		},
		{
			name: "user thinking",
			req:  inference.Request{Model: m, Messages: content.AgenticMessages{userMessage(&content.ThinkingBlock{Thinking: "think"})}},
		},
		{
			name: "assistant tool result",
			req:  inference.Request{Model: m, Messages: content.AgenticMessages{assistantMessage(&content.ToolResultBlock{ToolUseID: "call", Content: []content.Block{&content.TextBlock{Text: "result"}}})}},
		},
		{
			name: "document without text",
			req:  inference.Request{Model: m, Messages: content.AgenticMessages{userMessage(&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report-pdf", Data: []byte("pdf")})}},
		},
		{
			name: "document name with period",
			req:  inference.Request{Model: m, Messages: content.AgenticMessages{userMessage(&content.DocumentBlock{MediaType: content.MediaTypeDocumentPDF, Name: "report.pdf", Data: []byte("pdf")}, &content.TextBlock{Text: "attached"})}},
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := bedrockconverse.EncodeRequest(tc.req)
			var unsupported *bedrockconverse.UnsupportedBlockError
			if !errors.As(err, &unsupported) {
				t.Fatalf("EncodeRequest() error = %T (%v), want UnsupportedBlockError", err, err)
			}
		})
	}
}

func TestEncodeRequest_RejectsInvalidSharedFeatures(t *testing.T) {
	t.Parallel()

	_, err := bedrockconverse.EncodeRequest(inference.Request{
		Model: baseModel(),
		Output: &inference.OutputSchema{
			Name:   "answer",
			Schema: json.RawMessage(`{"type":"object","additionalProperties":false}`),
		},
	})
	var unsupported *inference.StructuredOutputUnsupportedError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error = %T (%v), want StructuredOutputUnsupportedError", err, err)
	}
}

func intPtr(value int) *int           { return &value }
func floatPtr(value float64) *float64 { return &value }

func asInt(t *testing.T, raw json.RawMessage) int {
	t.Helper()
	var value int
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatalf("json.Unmarshal(int) error = %v; raw = %s", err, raw)
	}
	return value
}

func asFloat(t *testing.T, raw json.RawMessage) float64 {
	t.Helper()
	var value float64
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatalf("json.Unmarshal(float) error = %v; raw = %s", err, raw)
	}
	return value
}

// TestEncodeRequest_RefusalBlockFailsClosed pins the fail-closed encoding of a
// content.RefusalBlock. Converse reports a decline through the response's
// stopReason ("guardrail_intervened", "content_filtered"); its ContentBlock
// union is text|image|document|video|audio|toolUse|toolResult|guardContent|
// reasoningContent|cachePoint, with no refusal member, and no request field
// carries one either. Sending it as `text` would replay the model's own
// decline back to it as something it said.
func TestEncodeRequest_RefusalBlockFailsClosed(t *testing.T) {
	t.Parallel()

	_, err := bedrockconverse.EncodeRequest(inference.Request{
		Model:    baseModel(),
		Messages: content.AgenticMessages{assistantMessage(&content.RefusalBlock{Text: "I cannot help with that."})},
	})
	if err == nil {
		t.Fatal("expected an error, got nil")
	}
	var unsupported *bedrockconverse.UnsupportedBlockError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error = %v (%T), want *UnsupportedBlockError", err, err)
	}
	if !strings.Contains(unsupported.Reason, "refusal") {
		t.Errorf("Reason = %q, want it to name the refusal limitation", unsupported.Reason)
	}
}
