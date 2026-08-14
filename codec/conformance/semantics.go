package conformance

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// This file holds the checks a JSON Schema cannot express, run on encoded
// REQUEST bodies alongside the schema gate.
//
// The gate's blind spot is documented in inference/CLAUDE.md: "Where a gate is
// blind, hold the constraint somewhere that is not blind." OpenAI types
// `tool_calls[].function.arguments` as `string` (ChatCompletionMessageToolCall
// in openai-openapi), so the schema is satisfied by ANY string — including a
// string that contains another quoted string. That is exactly what a
// double-encoding encoder emits, and it shipped: the decoder left the wire's
// JSON string literal in ToolUseBlock.Input and the encoder quoted it again,
// producing "\"{\\\"city\\\":\\\"Paris\\\"}\"". Two gateways rejected it by
// name; the rest accepted the corrupted arguments with a 200. Only a semantic
// check sees it.

// checkRequestSemantics applies the per-format semantic checks for an encoded
// request body. Formats with no such check return nil.
func checkRequestSemantics(format, kind string, body []byte) error {
	if format == "openai" && kind == "chat_completion_request" {
		return checkChatToolArguments(body)
	}
	return nil
}

// checkChatToolArguments holds every assistant tool call in a Chat Completions
// request body to the semantics `arguments` carries but its `string` type does
// not state: the value must be a JSON string, and what that string CONTAINS
// must be the arguments object.
//
// A payload whose contents do not parse as JSON at all is deliberately allowed
// through. OpenAI's own schema says of this member, "the model does not always
// generate valid JSON ... Validate the arguments in your code before calling
// your function", so a replayed assistant turn may legitimately carry the
// invalid text the model produced; repairing or rejecting it here would make
// the codec lie about what the model said. What is never legitimate is
// OUR OWN encoding adding a layer — a nested JSON string, or a non-object JSON
// value where the tool's parameters belong.
func checkChatToolArguments(body []byte) error {
	var wire struct {
		Messages []struct {
			ToolCalls []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		// Malformed bodies are the schema gate's business; it runs first and
		// reports them with an instance path.
		return nil
	}
	for i, message := range wire.Messages {
		for j, call := range message.ToolCalls {
			where := fmt.Sprintf("/messages/%d/tool_calls/%d/function/arguments", i, j)
			if err := checkToolArgumentsValue(call.Function.Arguments); err != nil {
				return fmt.Errorf("conformance: openai/chat_completion_request semantic check failed at %s (tool %q, id %q): %w",
					where, call.Function.Name, call.ID, err)
			}
		}
	}
	return nil
}

func checkToolArgumentsValue(raw json.RawMessage) error {
	var text string
	if err := json.Unmarshal(raw, &text); err != nil {
		return fmt.Errorf("arguments must be a JSON string carrying the arguments object, got %s", raw)
	}
	trimmed := bytes.TrimSpace([]byte(text))
	if len(trimmed) == 0 {
		return fmt.Errorf("arguments is empty; a call with no parameters sends %q", "{}")
	}
	if !json.Valid(trimmed) {
		// Model-produced text that is not JSON: preserved verbatim by design.
		return nil
	}
	switch trimmed[0] {
	case '{':
		return nil
	case '"':
		return fmt.Errorf("arguments is double-encoded: the string contains another JSON string (%s), not an object", text)
	default:
		return fmt.Errorf("arguments contains the JSON value %s, not an object", text)
	}
}
