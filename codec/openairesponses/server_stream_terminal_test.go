package openairesponses_test

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/conformance"
	"github.com/looprig/inference/codec/openairesponses"
	stream "github.com/looprig/inference/stream"
)

// The two events this file pins are both things the frame gate structurally
// CANNOT catch. conformance.MustValidateStream validates the frames that ARE
// present against ResponseStreamEvent; a frame that was never written has no
// instance to validate, so a missing terminal keeps the gate green. That is why
// these assertions are on the served event SEQUENCE rather than on frame
// contents.

// TestServerStream_TextItemEmitsOutputTextDone pins the terminal for the
// output_text.* channel. A client that reconstructs assistant text from
// response.output_text.delta alone — the documented way to do it, and what a
// client that ignores content_part.* does — otherwise never learns the text
// finished: closeOpenItem sent content_part.done and output_item.done and
// nothing on the channel the deltas arrived on. The refusal branch already
// sends refusal.done, so the text branch was the only asymmetric one.
func TestServerStream_TextItemEmitsOutputTextDone(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	enc, err := (openairesponses.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "hello "}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "world"}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.Finish(stream.StreamResult{Model: "gpt-test", FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	body := rec.Body.String()
	frames := sseNamedFrames(body)

	var done *sseNamedFrame
	for i := range frames {
		if frames[i].Event == "response.output_text.done" {
			done = &frames[i]
		}
	}
	if done == nil {
		t.Fatalf("no response.output_text.done frame; served events = %v", sseEventTypes(t, body))
	}

	// ResponseTextDoneEvent.required is
	// [type,item_id,output_index,content_index,text,sequence_number,logprobs],
	// and `text` must be the whole accumulated text, not the last delta.
	var event struct {
		Type         string            `json:"type"`
		ItemID       string            `json:"item_id"`
		OutputIndex  *int              `json:"output_index"`
		ContentIndex *int              `json:"content_index"`
		Text         *string           `json:"text"`
		Logprobs     []json.RawMessage `json:"logprobs"`
		Sequence     *int              `json:"sequence_number"`
	}
	if err := json.Unmarshal(done.Data, &event); err != nil {
		t.Fatalf("decode response.output_text.done: %v", err)
	}
	if event.Text == nil || *event.Text != "hello world" {
		t.Errorf("response.output_text.done text = %v, want %q", event.Text, "hello world")
	}
	if event.ItemID == "" || event.OutputIndex == nil || event.ContentIndex == nil ||
		event.Sequence == nil || event.Logprobs == nil {
		t.Errorf("response.output_text.done is missing a required member: %s", done.Data)
	}

	// It must precede content_part.done: the part terminal closes the container
	// the text terminal belongs to.
	order := sseEventTypes(t, body)
	textDone, partDone := indexOf(order, "response.output_text.done"), indexOf(order, "response.content_part.done")
	if textDone < 0 || partDone < 0 || textDone > partDone {
		t.Errorf("response.output_text.done must precede response.content_part.done; order = %v", order)
	}

	conformance.MustValidateStream(t, "openai-responses", "stream_event", []byte(body))
}

// TestServerStream_EmitsResponseInProgress pins response.in_progress. It is
// legal to omit — ResponseStreamEvent is a union, not a sequence — but every
// real Responses provider sends it between response.created and the first item
// event, and a client that waits for it before rendering (a reasonable reading
// of "created" as "accepted, not yet generating") hangs against a gateway that
// never sends one.
func TestServerStream_EmitsResponseInProgress(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	enc, err := (openairesponses.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "hi"}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.Finish(stream.StreamResult{Model: "gpt-test", FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	body := rec.Body.String()
	order := sseEventTypes(t, body)
	created, inProgress := indexOf(order, "response.created"), indexOf(order, "response.in_progress")
	firstItem := indexOf(order, "response.output_item.added")
	if inProgress < 0 {
		t.Fatalf("no response.in_progress event; order = %v", order)
	}
	if created != 0 || inProgress != created+1 || inProgress > firstItem {
		t.Errorf("response.in_progress must sit between response.created and the first item; order = %v", order)
	}

	// The envelope must carry the same response id and creation time as
	// response.created: they describe one response, not two.
	frames := sseNamedFrames(body)
	ids := map[string]string{}
	createdAt := map[string]json.RawMessage{}
	for _, f := range frames {
		if f.Event != "response.created" && f.Event != "response.in_progress" {
			continue
		}
		var envelope struct {
			Response struct {
				ID        string          `json:"id"`
				CreatedAt json.RawMessage `json:"created_at"`
				Status    string          `json:"status"`
			} `json:"response"`
		}
		if err := json.Unmarshal(f.Data, &envelope); err != nil {
			t.Fatalf("decode %s: %v", f.Event, err)
		}
		if envelope.Response.Status != "in_progress" {
			t.Errorf("%s status = %q, want in_progress", f.Event, envelope.Response.Status)
		}
		ids[f.Event] = envelope.Response.ID
		createdAt[f.Event] = envelope.Response.CreatedAt
	}
	if ids["response.created"] != ids["response.in_progress"] {
		t.Errorf("response id differs between created (%q) and in_progress (%q)",
			ids["response.created"], ids["response.in_progress"])
	}
	if string(createdAt["response.created"]) != string(createdAt["response.in_progress"]) {
		t.Errorf("created_at differs between created (%s) and in_progress (%s)",
			createdAt["response.created"], createdAt["response.in_progress"])
	}

	conformance.MustValidateStream(t, "openai-responses", "stream_event", []byte(body))
}

// TestServerStream_TextDoneIsNotEmittedForOtherItemKinds is the positive
// control on the new terminal: output_text.done belongs to the text channel
// only. A refusal keeps refusal.done, reasoning keeps
// reasoning_summary_text.done, and a tool call keeps
// function_call_arguments.done — none of them may acquire a spurious text
// terminal, which would tell a client that text it never received has ended.
func TestServerStream_TextDoneIsNotEmittedForOtherItemKinds(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		chunk    content.Chunk
		wantDone string
	}{
		{"refusal", &content.RefusalChunk{Text: "I cannot"}, "response.refusal.done"},
		{"reasoning", &content.ThinkingChunk{Thinking: "step"}, "response.reasoning_summary_text.done"},
		{"tool call", &content.ToolUseChunk{Index: 0, ID: "call_1", Name: "read", InputJSON: `{}`}, "response.function_call_arguments.done"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			rec := httptest.NewRecorder()
			enc, err := (openairesponses.Codec{}).OpenStream(rec)
			if err != nil {
				t.Fatalf("OpenStream() error = %v", err)
			}
			if err := enc.WriteChunk(tc.chunk); err != nil {
				t.Fatalf("WriteChunk() error = %v", err)
			}
			if err := enc.Finish(stream.StreamResult{FinishReason: stream.FinishReasonStop}); err != nil {
				t.Fatalf("Finish() error = %v", err)
			}
			order := sseEventTypes(t, rec.Body.String())
			if indexOf(order, "response.output_text.done") >= 0 {
				t.Errorf("a %s item emitted response.output_text.done; order = %v", tc.name, order)
			}
			if indexOf(order, tc.wantDone) < 0 {
				t.Errorf("a %s item lost its own terminal %q; order = %v", tc.name, tc.wantDone, order)
			}
			conformance.MustValidateStream(t, "openai-responses", "stream_event", rec.Body.Bytes())
		})
	}
}

// TestTheStreamGateCannotSeeAMissingEvent measures the limit that made both
// defects above invisible, rather than asserting it. A served stream with the
// text terminal surgically removed still validates frame-by-frame, because
// ResponseStreamEvent constrains each frame's shape and says nothing about
// which frames must appear.
func TestTheStreamGateCannotSeeAMissingEvent(t *testing.T) {
	t.Parallel()

	rec := httptest.NewRecorder()
	enc, err := (openairesponses.Codec{}).OpenStream(rec)
	if err != nil {
		t.Fatalf("OpenStream() error = %v", err)
	}
	if err := enc.WriteChunk(&content.TextChunk{Text: "hello"}); err != nil {
		t.Fatalf("WriteChunk() error = %v", err)
	}
	if err := enc.Finish(stream.StreamResult{FinishReason: stream.FinishReasonStop}); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}

	var kept []string
	for _, block := range strings.Split(rec.Body.String(), "\n\n") {
		if block == "" || strings.Contains(block, "response.output_text.done") ||
			strings.Contains(block, "response.in_progress") {
			continue
		}
		kept = append(kept, block)
	}
	mutilated := strings.Join(kept, "\n\n") + "\n\n"

	if n := conformance.MustValidateStream(t, "openai-responses", "stream_event", []byte(mutilated)); n == 0 {
		t.Fatal("gate validated no frames at all; the measurement below proves nothing")
	}
	// Reaching here IS the finding: the gate accepted a stream whose text
	// channel never terminates. If a future schema refresh starts expressing
	// event-sequence requirements this test fails, and the encoder-side
	// assertions above become redundant rather than load-bearing.
}

func indexOf(values []string, want string) int {
	for i, v := range values {
		if v == want {
			return i
		}
	}
	return -1
}
