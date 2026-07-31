package gateway_test

// This file is Task 14 of the inference-gateway plan: an end-to-end proof
// that all 16 ingress x target dialect combinations work through the real
// gateway.Handler, using ONLY the machinery in matrix_fixtures_test.go
// (real ServerCodecs, a real transport.Client, and fake target servers built
// from each dialect's own server-side codec methods).
import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"
)

// --- Step 1/2: the 4x4 portable-subset matrix -------------------------------

// TestMatrix_PortableSubset_AllDialectPairs drives the portable-subset
// fixture (system + user text + assistant tool call + tool result + final
// user turn) through all 16 (ingress, target) combinations, proving:
//
//   - neutral text survives the round trip (the target's canned response
//     text appears correctly in the ingress-native response);
//   - system instructions reach the target correctly (asserted on what the
//     fake target server actually decoded);
//   - the tool-call/tool-result thread the harness sent reaches the target
//     correctly (asserted on the decoded request, not just "no error");
//   - usage survives both directions;
//   - the stop finish reason is correctly reported in the ingress-native
//     shape.
func TestMatrix_PortableSubset_AllDialectPairs(t *testing.T) {
	for _, ingressFormat := range matrixFormats {
		for _, targetFormat := range matrixFormats {
			ingressFormat, targetFormat := ingressFormat, targetFormat
			t.Run(dialectName(ingressFormat)+"->"+dialectName(targetFormat), func(t *testing.T) {
				t.Parallel()
				ingress := matrixDialects[ingressFormat]
				targetD := matrixDialects[targetFormat]

				srv, ft := newFakeTarget(t, targetD.codec)
				ft.setResponse(portableCannedTextResponse())
				target := buildMatrixTarget(t, targetD, srv, "target-model", broadCaps()...)
				h := buildMatrixHandler(t, ingress, target)

				rr, resp := sendMatrixInvoke(t, h, ingress, portableFixtureRequest())
				if rr.Code != http.StatusOK {
					t.Fatalf("status = %d, want 200; body: %s", rr.Code, rr.Body.String())
				}

				// --- neutral text survives the round trip ---
				gotText := allText(resp.Message.Blocks)
				if !strings.Contains(gotText, "sunny, 65F") {
					t.Errorf("ingress-native response text = %q, want it to contain the target's canned text %q", gotText, "sunny, 65F")
				}

				// --- usage survives (response direction) ---
				if resp.Usage == nil {
					t.Fatalf("ingress-native response Usage = nil, want InputTokens=123 OutputTokens=45")
				}
				if resp.Usage.InputTokens != 123 || resp.Usage.OutputTokens != 45 {
					t.Errorf("ingress-native response Usage = %+v, want InputTokens=123 OutputTokens=45", resp.Usage)
				}

				// --- finish reason survives ---
				if resp.FinishReason != stream.FinishReasonStop {
					t.Errorf("ingress-native response FinishReason = %q, want %q", resp.FinishReason, stream.FinishReasonStop)
				}

				// --- what the TARGET actually decoded ---
				if got := ft.callCount(); got != 1 {
					t.Fatalf("target received %d requests, want 1", got)
				}
				decoded := ft.lastRequest(t).Request

				if decoded.System != "You are a terse weather assistant." {
					t.Errorf("target-decoded System = %q, want the harness's system instructions to reach the target", decoded.System)
				}
				if !messagesContainText(decoded.Messages, "What's the weather in nyc?") {
					t.Errorf("target-decoded messages do not contain the original user text")
				}
				if !hasToolUse(decoded.Messages, "call_1", "get_weather", "nyc") {
					t.Errorf("target-decoded messages do not contain the original tool_use (id=call_1, name=get_weather, city=nyc)")
				}
				if !hasToolResult(decoded.Messages, "call_1", "sunny") {
					t.Errorf("target-decoded messages do not contain the original tool_result (tool_use_id=call_1, text contains sunny)")
				}
			})
		}
	}
}

// --- tool call in response + tool result in a follow-up request, both directions ---

// TestMatrix_ToolCallResponse_AndFollowUpReplay proves a tool call
// surfaces correctly in the ingress-native response (with the tool_use
// finish reason), and that a follow-up request replaying that tool call
// plus a tool result reaches the target correctly -- exercising both
// directions of tool-call/tool-result translation, and the tool_use finish
// reason, on one representative cross-dialect pair (ingress=OpenAI Chat
// Completions, target=Anthropic).
func TestMatrix_ToolCallResponse_AndFollowUpReplay(t *testing.T) {
	t.Parallel()
	ingress := matrixDialects[model.APIFormatOpenAI]
	targetD := matrixDialects[model.APIFormatAnthropic]

	srv, ft := newFakeTarget(t, targetD.codec)
	ft.setResponse(cannedToolCallResponse())
	target := buildMatrixTarget(t, targetD, srv, "target-model", broadCaps()...)
	h := buildMatrixHandler(t, ingress, target)

	rr1, resp1 := sendMatrixInvoke(t, h, ingress, portableFixtureRequest())
	if rr1.Code != http.StatusOK {
		t.Fatalf("turn 1 status = %d, body: %s", rr1.Code, rr1.Body.String())
	}
	if resp1.FinishReason != stream.FinishReasonToolUse {
		t.Fatalf("turn 1 FinishReason = %q, want %q", resp1.FinishReason, stream.FinishReasonToolUse)
	}
	tu := findFirstToolUse(resp1.Message.Blocks)
	if tu == nil || tu.ID != "call_9" || tu.Name != "get_time" {
		t.Fatalf("turn 1 ingress-native response ToolUseBlock = %#v, want id=call_9 name=get_time", tu)
	}

	// Harness continues the conversation with the tool result.
	ft.setResponse(portableCannedTextResponse())
	followUp := inference.Request{
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "what time is it in utc?"}}}},
			resp1.Message,
			&content.ToolResultMessage{
				Message:   content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "12:00"}}},
				ToolUseID: "call_9",
			},
		},
	}
	rr2, _ := sendMatrixInvoke(t, h, ingress, followUp)
	if rr2.Code != http.StatusOK {
		t.Fatalf("turn 2 status = %d, body: %s", rr2.Code, rr2.Body.String())
	}

	decoded := ft.lastRequest(t).Request
	if !hasToolUse(decoded.Messages, "call_9", "get_time", "utc") {
		t.Errorf("turn 2 target-decoded messages do not contain the replayed tool_use (id=call_9, name=get_time)")
	}
	if !hasToolResult(decoded.Messages, "call_9", "12:00") {
		t.Errorf("turn 2 target-decoded messages do not contain the tool_result (tool_use_id=call_9, text contains 12:00)")
	}
}

// TestMatrix_ParallelToolCalls proves 2+ parallel tool calls in a target's
// response preserve their distinct IDs through to the ingress-native
// response, on one representative pair (ingress=Anthropic,
// target=OpenAI Chat Completions).
func TestMatrix_ParallelToolCalls(t *testing.T) {
	t.Parallel()
	ingress := matrixDialects[model.APIFormatAnthropic]
	targetD := matrixDialects[model.APIFormatOpenAI]

	srv, ft := newFakeTarget(t, targetD.codec)
	ft.setResponse(cannedParallelToolCallResponse())
	target := buildMatrixTarget(t, targetD, srv, "target-model", broadCaps()...)
	h := buildMatrixHandler(t, ingress, target)

	rr, resp := sendMatrixInvoke(t, h, ingress, portableFixtureRequest())
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, body: %s", rr.Code, rr.Body.String())
	}
	if resp.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, stream.FinishReasonToolUse)
	}

	var calls []*content.ToolUseBlock
	for _, b := range resp.Message.Blocks {
		if tu, ok := b.(*content.ToolUseBlock); ok {
			calls = append(calls, tu)
		}
	}
	if len(calls) != 2 {
		t.Fatalf("ingress-native response has %d ToolUseBlocks, want 2 (%#v)", len(calls), resp.Message.Blocks)
	}
	if calls[0].ID != "call_a" || calls[0].Name != "get_weather" {
		t.Errorf("calls[0] = %#v, want id=call_a name=get_weather", calls[0])
	}
	if calls[1].ID != "call_b" || calls[1].Name != "get_time" {
		t.Errorf("calls[1] = %#v, want id=call_b name=get_time", calls[1])
	}
}

// TestMatrix_VisibleThinkingSurvivesAsText proves a target's visible
// reasoning/thinking text (no opaque provider state) survives as visible
// text in the ingress-native response, on two representative pairs covering
// both an ingress that has no native "thinking" wire concept of its own
// (OpenAI Chat Completions, which maps it to reasoning_content) and one that
// does (Gemini).
func TestMatrix_VisibleThinkingSurvivesAsText(t *testing.T) {
	t.Parallel()
	pairs := []struct {
		ingress model.APIFormat
		target  model.APIFormat
	}{
		{model.APIFormatOpenAI, model.APIFormatAnthropic},
		{model.APIFormatGemini, model.APIFormatOpenAIResponses},
	}
	for _, p := range pairs {
		p := p
		t.Run(dialectName(p.ingress)+"->"+dialectName(p.target), func(t *testing.T) {
			t.Parallel()
			ingress := matrixDialects[p.ingress]
			targetD := matrixDialects[p.target]

			srv, ft := newFakeTarget(t, targetD.codec)
			ft.setResponse(cannedThinkingResponse())
			target := buildMatrixTarget(t, targetD, srv, "target-model", broadCaps()...)
			h := buildMatrixHandler(t, ingress, target)

			rr, resp := sendMatrixInvoke(t, h, ingress, portableFixtureRequest())
			if rr.Code != http.StatusOK {
				t.Fatalf("status = %d, body: %s", rr.Code, rr.Body.String())
			}
			tb := findFirstThinkingBlock(resp.Message.Blocks)
			if tb == nil {
				t.Fatalf("ingress-native response carries no ThinkingBlock, want visible thinking text to survive (blocks: %#v)", resp.Message.Blocks)
			}
			if !strings.Contains(tb.Thinking, "Step 1") {
				t.Errorf("ingress-native ThinkingBlock.Thinking = %q, want it to contain the target's visible reasoning text", tb.Thinking)
			}
		})
	}
}

// TestMatrix_ImageInboundReachesTarget proves an inline image in the
// harness's request reaches the target correctly, on two representative
// pairs.
func TestMatrix_ImageInboundReachesTarget(t *testing.T) {
	t.Parallel()
	pairs := []struct {
		ingress model.APIFormat
		target  model.APIFormat
	}{
		{model.APIFormatAnthropic, model.APIFormatOpenAI},
		{model.APIFormatGemini, model.APIFormatOpenAIResponses},
	}
	for _, p := range pairs {
		p := p
		t.Run(dialectName(p.ingress)+"->"+dialectName(p.target), func(t *testing.T) {
			t.Parallel()
			ingress := matrixDialects[p.ingress]
			targetD := matrixDialects[p.target]

			srv, ft := newFakeTarget(t, targetD.codec)
			ft.setResponse(portableCannedTextResponse())
			target := buildMatrixTarget(t, targetD, srv, "target-model", broadCaps()...)
			h := buildMatrixHandler(t, ingress, target)

			rr, _ := sendMatrixInvoke(t, h, ingress, imageFixtureRequest())
			if rr.Code != http.StatusOK {
				t.Fatalf("status = %d, body: %s", rr.Code, rr.Body.String())
			}

			decoded := ft.lastRequest(t).Request
			img := findFirstImage(decoded.Messages)
			if img == nil {
				t.Fatalf("target-decoded messages carry no ImageBlock")
			}
			if string(img.Source.Data) != string(tinyPNGBytes) {
				t.Errorf("target-decoded image bytes = %x, want %x (byte-exact)", img.Source.Data, tinyPNGBytes)
			}
		})
	}
}

// --- opaque thinking state: same-dialect replay survives --------------------

// TestMatrix_ThinkingOpaqueState_SameDialectReplaySurvives proves that when
// ingress and target dialects MATCH, a ThinkingBlock's opaque continuation
// state (Anthropic's Signature; Gemini's ProviderState/thoughtSignature)
// that the target sends back survives being replayed verbatim when the
// harness immediately continues the conversation with it.
func TestMatrix_ThinkingOpaqueState_SameDialectReplaySurvives(t *testing.T) {
	t.Parallel()

	t.Run("anthropic_signature", func(t *testing.T) {
		t.Parallel()
		d := matrixDialects[model.APIFormatAnthropic]
		srv, ft := newFakeTarget(t, d.codec)
		target := buildMatrixTarget(t, d, srv, "target-model", broadCaps()...)
		h := buildMatrixHandler(t, d, target)

		ft.setResponse(&inference.Response{
			Message: &content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					&content.ThinkingBlock{Thinking: "step by step", Signature: "sig-anthropic-777"},
					&content.TextBlock{Text: "42"},
				},
			}},
			FinishReason: stream.FinishReasonStop,
		})
		rr1, resp1 := sendMatrixInvoke(t, h, d, portableFixtureRequest())
		if rr1.Code != http.StatusOK {
			t.Fatalf("turn 1 status = %d, body: %s", rr1.Code, rr1.Body.String())
		}
		tb := findFirstThinkingBlock(resp1.Message.Blocks)
		if tb == nil || tb.Signature != "sig-anthropic-777" {
			t.Fatalf("turn 1 ingress-native ThinkingBlock = %#v, want Signature=sig-anthropic-777", tb)
		}

		ft.setResponse(portableCannedTextResponse())
		followUp := inference.Request{
			Messages: content.AgenticMessages{
				&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "continue"}}}},
				resp1.Message,
				&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "go on"}}}},
			},
		}
		rr2, _ := sendMatrixInvoke(t, h, d, followUp)
		if rr2.Code != http.StatusOK {
			t.Fatalf("turn 2 status = %d, body: %s", rr2.Code, rr2.Body.String())
		}
		got := findFirstThinkingBlockInMessages(ft.lastRequest(t).Request.Messages)
		if got == nil || got.Signature != "sig-anthropic-777" {
			t.Errorf("same-dialect replay: target-decoded ThinkingBlock = %#v, want Signature=sig-anthropic-777 preserved byte-for-byte", got)
		}
	})

	t.Run("gemini_providerstate", func(t *testing.T) {
		t.Parallel()
		d := matrixDialects[model.APIFormatGemini]
		srv, ft := newFakeTarget(t, d.codec)
		target := buildMatrixTarget(t, d, srv, "target-model", broadCaps()...)
		h := buildMatrixHandler(t, d, target)

		ft.setResponse(&inference.Response{
			Message: &content.AIMessage{Message: content.Message{
				Role: content.RoleAssistant,
				Blocks: []content.Block{
					content.NewThinkingBlock("step by step", "", json.RawMessage(`"opaque-thought-sig-999"`), "gemini"),
					&content.TextBlock{Text: "42"},
				},
			}},
			FinishReason: stream.FinishReasonStop,
		})
		rr1, resp1 := sendMatrixInvoke(t, h, d, portableFixtureRequest())
		if rr1.Code != http.StatusOK {
			t.Fatalf("turn 1 status = %d, body: %s", rr1.Code, rr1.Body.String())
		}
		tb := findFirstThinkingBlock(resp1.Message.Blocks)
		if tb == nil {
			t.Fatalf("turn 1 ingress-native response carries no ThinkingBlock")
		}
		var gotSig string
		if err := json.Unmarshal(tb.ProviderState, &gotSig); err != nil || gotSig != "opaque-thought-sig-999" {
			t.Fatalf("turn 1 ingress-native ThinkingBlock.ProviderState = %s, want JSON string opaque-thought-sig-999", tb.ProviderState)
		}

		ft.setResponse(portableCannedTextResponse())
		followUp := inference.Request{
			Messages: content.AgenticMessages{
				&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "continue"}}}},
				resp1.Message,
				&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "go on"}}}},
			},
		}
		rr2, _ := sendMatrixInvoke(t, h, d, followUp)
		if rr2.Code != http.StatusOK {
			t.Fatalf("turn 2 status = %d, body: %s", rr2.Code, rr2.Body.String())
		}
		got := findFirstThinkingBlockInMessages(ft.lastRequest(t).Request.Messages)
		if got == nil {
			t.Fatalf("same-dialect replay: target-decoded messages carry no ThinkingBlock")
		}
		var gotSig2 string
		if err := json.Unmarshal(got.ProviderState, &gotSig2); err != nil || gotSig2 != "opaque-thought-sig-999" {
			t.Errorf("same-dialect replay: target-decoded ThinkingBlock.ProviderState = %s, want JSON string opaque-thought-sig-999 preserved byte-for-byte", got.ProviderState)
		}
	})
}

// --- opaque thinking state: cross-dialect forwarding is refused ------------

// TestMatrix_ThinkingOpaqueState_CrossDialectNotForwarded proves that when
// ingress and target dialects DIFFER, a ThinkingBlock's ingress-native
// opaque continuation state is NOT forwarded to the differently-dialected
// target as if it were meaningful there. The first 3 pairs pivot on
// Anthropic's Signature field, which is a structurally distinct Go field
// from every other dialect's ProviderState -- no bundled codec's outbound
// encoder ever reads Signature, and Anthropic's own outbound encoder never
// reads ProviderState, so these pairs demonstrate the property by
// construction.
//
// The final 2 pairs are the regression case for a real bug found in this
// task's own integration testing: geminiapi and openairesponses both store
// their opaque state as a bare JSON-marshaled string in ProviderState with
// no tag identifying which dialect produced it (see geminiapi/decode.go's
// providerStateFromThoughtSignature and openairesponses/decode.go's
// opaqueStateFromWire doc comments), so a ThinkingBlock decoded from one of
// them would round-trip undetected through the other's encoder as if it
// were that encoder's own native state. content.ThinkingBlock.
// ProviderStateFormat (and the guards added to each codec's forwarding
// sites) now closes this: a block is tagged with the dialect that produced
// it, and each codec only replays ProviderState toward its own wire field
// when that tag matches its own label. These two cases set the format tag
// to match INGRESS (so it survives that dialect's own outbound-encode step
// unmolested -- see geminiapi/encode.go's and openairesponses/encode.go's
// forwarding guards) while TARGET is the other, mismatched dialect, proving
// the target's own forwarding guard -- not just the ingress one -- refuses
// to treat the foreign tag as its own.
func TestMatrix_ThinkingOpaqueState_CrossDialectNotForwarded(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		ingress   model.APIFormat
		target    model.APIFormat
		block     *content.ThinkingBlock
		secretSub string
	}{
		{
			name:      "anthropic_signature_not_forwarded_to_gemini_target",
			ingress:   model.APIFormatAnthropic,
			target:    model.APIFormatGemini,
			block:     &content.ThinkingBlock{Thinking: "reasoning", Signature: "sig-anthropic-secret-abc"},
			secretSub: "sig-anthropic-secret-abc",
		},
		{
			// The "vice versa" of the pair above.
			name:      "gemini_providerstate_not_forwarded_to_anthropic_target",
			ingress:   model.APIFormatGemini,
			target:    model.APIFormatAnthropic,
			block:     content.NewThinkingBlock("reasoning", "", json.RawMessage(`"gemini-secret-thought-xyz"`), "gemini"),
			secretSub: "gemini-secret-thought-xyz",
		},
		{
			name:      "anthropic_signature_not_forwarded_to_responses_target",
			ingress:   model.APIFormatAnthropic,
			target:    model.APIFormatOpenAIResponses,
			block:     &content.ThinkingBlock{Thinking: "reasoning", Signature: "sig-anthropic-secret-def"},
			secretSub: "sig-anthropic-secret-def",
		},
		{
			// Regression case: a ThinkingBlock genuinely native to Gemini
			// (ProviderStateFormat "gemini", so it survives geminiapi's own
			// outbound-encode guard) replayed toward an OpenAI-Responses
			// target must NOT be treated as that target's own
			// encrypted_content.
			name:      "gemini_providerstate_not_forwarded_to_responses_target",
			ingress:   model.APIFormatGemini,
			target:    model.APIFormatOpenAIResponses,
			block:     content.NewThinkingBlock("reasoning", "", json.RawMessage(`"gemini-secret-cross-format-111"`), "gemini"),
			secretSub: "gemini-secret-cross-format-111",
		},
		{
			// The inverse of the pair above: a ThinkingBlock genuinely
			// native to OpenAI-Responses replayed toward a Gemini target
			// must NOT be treated as that target's own thoughtSignature.
			name:      "responses_providerstate_not_forwarded_to_gemini_target",
			ingress:   model.APIFormatOpenAIResponses,
			target:    model.APIFormatGemini,
			block:     content.NewThinkingBlock("reasoning", "", json.RawMessage(`"responses-secret-cross-format-222"`), "openai-responses"),
			secretSub: "responses-secret-cross-format-222",
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ingress := matrixDialects[tc.ingress]
			targetD := matrixDialects[tc.target]

			srv, ft := newFakeTarget(t, targetD.codec)
			ft.setResponse(portableCannedTextResponse())
			target := buildMatrixTarget(t, targetD, srv, "target-model", broadCaps()...)
			h := buildMatrixHandler(t, ingress, target)

			// A follow-up-shaped request: the harness is replaying an
			// assistant turn it previously received, natively, from THIS
			// ingress dialect -- carrying that dialect's own opaque
			// continuation state -- and this turn happens to route to a
			// DIFFERENT target dialect.
			req := inference.Request{
				Messages: content.AgenticMessages{
					&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "continue"}}}},
					&content.AIMessage{Message: content.Message{
						Role:   content.RoleAssistant,
						Blocks: []content.Block{tc.block, &content.TextBlock{Text: "ok"}},
					}},
					&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "go on"}}}},
				},
			}

			rr, _ := sendMatrixInvoke(t, h, ingress, req)
			if rr.Code != http.StatusOK {
				t.Fatalf("status = %d, body: %s", rr.Code, rr.Body.String())
			}

			if raw := ft.lastRawBody(t); strings.Contains(string(raw), tc.secretSub) {
				t.Errorf("LEAK: target %s's raw wire request body contains the ingress (%s) opaque secret %q:\n%s",
					tc.target, tc.ingress, tc.secretSub, raw)
			}
			if thinkingOpaqueSubstringPresent(ft.lastRequest(t).Request.Messages, tc.secretSub) {
				t.Errorf("LEAK: target %s's DECODED request carries the ingress (%s) opaque secret %q in a ThinkingBlock",
					tc.target, tc.ingress, tc.secretSub)
			}
		})
	}
}

// --- shared search helpers ---------------------------------------------

// allText concatenates every *content.TextBlock's text in blocks, in order.
func allText(blocks []content.Block) string {
	var sb strings.Builder
	for _, b := range blocks {
		if tb, ok := b.(*content.TextBlock); ok {
			sb.WriteString(tb.Text)
		}
	}
	return sb.String()
}

// findFirstToolUse returns the first *content.ToolUseBlock in blocks, or nil.
func findFirstToolUse(blocks []content.Block) *content.ToolUseBlock {
	for _, b := range blocks {
		if tu, ok := b.(*content.ToolUseBlock); ok {
			return tu
		}
	}
	return nil
}

// findFirstImage returns the first *content.ImageBlock found in any message
// across msgs, or nil.
func findFirstImage(msgs content.AgenticMessages) *content.ImageBlock {
	for _, m := range msgs {
		var blocks []content.Block
		switch m := m.(type) {
		case *content.UserMessage:
			blocks = m.Blocks
		case *content.AIMessage:
			blocks = m.Blocks
		case *content.ToolResultMessage:
			blocks = m.Blocks
		}
		for _, b := range blocks {
			if img, ok := b.(*content.ImageBlock); ok {
				return img
			}
		}
	}
	return nil
}

// messagesContainText reports whether any message across msgs carries a
// *content.TextBlock whose text contains substr.
func messagesContainText(msgs content.AgenticMessages, substr string) bool {
	for _, m := range msgs {
		var blocks []content.Block
		switch m := m.(type) {
		case *content.UserMessage:
			blocks = m.Blocks
		case *content.AIMessage:
			blocks = m.Blocks
		case *content.ToolResultMessage:
			blocks = m.Blocks
		}
		if strings.Contains(allText(blocks), substr) {
			return true
		}
	}
	return false
}

// hasToolUse reports whether msgs contains an AIMessage with a
// *content.ToolUseBlock matching id and name, whose Input contains
// inputSubstr somewhere in its raw JSON.
func hasToolUse(msgs content.AgenticMessages, id, name, inputSubstr string) bool {
	for _, m := range msgs {
		ai, ok := m.(*content.AIMessage)
		if !ok {
			continue
		}
		for _, b := range ai.Blocks {
			tu, ok := b.(*content.ToolUseBlock)
			if !ok {
				continue
			}
			if tu.ID == id && tu.Name == name && strings.Contains(string(tu.Input), inputSubstr) {
				return true
			}
		}
	}
	return false
}

// hasToolResult reports whether msgs contains a *content.ToolResultMessage
// matching toolUseID whose text contains textSubstr.
func hasToolResult(msgs content.AgenticMessages, toolUseID, textSubstr string) bool {
	for _, m := range msgs {
		tr, ok := m.(*content.ToolResultMessage)
		if !ok {
			continue
		}
		if tr.ToolUseID == toolUseID && strings.Contains(allText(tr.Blocks), textSubstr) {
			return true
		}
	}
	return false
}
