package inference_test

import (
	"encoding/json"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
)

func FuzzValidateOutputSchema(f *testing.F) {
	seeds := []string{
		`{"type":"object","properties":{},"required":[],"additionalProperties":false}`,
		`{"type":"object","properties":{"x":{"type":"array","items":{"type":"integer"}}},"required":["x"],"additionalProperties":false}`,
		`{"type":"object","additionalProperties":true}`,
		`{"type":["object","null"]}`,
		`{`,
		`[]`,
		"\xff",
	}
	for _, seed := range seeds {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, schema string) {
		output := inference.OutputSchema{
			Name:        "fuzz_result",
			Description: "Fuzzed schema.",
			Schema:      json.RawMessage(schema),
			Strict:      true,
		}
		_ = inference.ValidateOutputSchema(output)
	})
}

func FuzzStructuredMessageResult(f *testing.F) {
	seeds := []struct {
		shape int
		text  string
		tool  string
	}{
		{shape: 0, text: `{"ok":true}`},
		{shape: 1, tool: `{"message":"hello"}`},
		{shape: 2, text: `{`, tool: `[]`},
		{shape: 3, text: `{} {}`, tool: `null`},
		{shape: 4, text: "世界", tool: `{"unicode":"世界"}`},
	}
	for _, seed := range seeds {
		f.Add(seed.shape, seed.text, seed.tool)
	}

	f.Fuzz(func(t *testing.T, shape int, text, tool string) {
		var blocks []content.Block
		switch shape % 8 {
		case 0:
			blocks = []content.Block{&content.TextBlock{Text: text}}
		case 1:
			blocks = []content.Block{&content.ToolUseBlock{Name: inference.StructuredOutputToolName, Input: json.RawMessage(tool)}}
		case 2:
			blocks = []content.Block{&content.TextBlock{Text: text}, &content.ToolUseBlock{Name: inference.StructuredOutputToolName, Input: json.RawMessage(tool)}}
		case 3:
			blocks = []content.Block{&content.ToolUseBlock{Name: "ordinary", Input: json.RawMessage(tool)}}
		case 4:
			blocks = []content.Block{&content.ThinkingBlock{Thinking: text}, &content.TextBlock{Text: tool}}
		case 5:
			blocks = []content.Block{(*content.TextBlock)(nil)}
		case 6:
			blocks = []content.Block{&content.ImageBlock{}}
		case 7:
			blocks = []content.Block{
				&content.ToolUseBlock{Name: inference.StructuredOutputToolName, Input: json.RawMessage(tool)},
				&content.ToolUseBlock{Name: inference.StructuredOutputToolName, Input: json.RawMessage(text)},
			}
		}
		msg := &content.AIMessage{Message: content.Message{Role: content.RoleAssistant, Blocks: blocks}}
		got, err := inference.StructuredMessageResult(msg)
		if err != nil {
			return
		}
		if !json.Valid(got) || len(got) == 0 || got[0] != '{' {
			t.Fatalf("accepted non-object JSON: %q", got)
		}
	})
}
