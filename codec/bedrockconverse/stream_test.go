package bedrockconverse_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference/codec/bedrockconverse"
	"github.com/looprig/inference/stream"
)

func eventFrame(event, payload string) []byte {
	headers := bytes.NewBuffer(nil)
	writeHeader := func(name, value string) {
		headers.WriteByte(byte(len(name)))
		headers.WriteString(name)
		headers.WriteByte(7)
		var length [2]byte
		binary.BigEndian.PutUint16(length[:], uint16(len(value)))
		headers.Write(length[:])
		headers.WriteString(value)
	}
	writeHeader(":message-type", "event")
	writeHeader(":event-type", event)

	data := []byte(payload)
	total := 16 + headers.Len() + len(data)
	frame := make([]byte, total)
	binary.BigEndian.PutUint32(frame[:4], uint32(total))
	binary.BigEndian.PutUint32(frame[4:8], uint32(headers.Len()))
	binary.BigEndian.PutUint32(frame[8:12], crc32.ChecksumIEEE(frame[:8]))
	copy(frame[12:], headers.Bytes())
	copy(frame[12+headers.Len():], data)
	binary.BigEndian.PutUint32(frame[total-4:], crc32.ChecksumIEEE(frame[:total-4]))
	return frame
}

func exceptionFrame(exceptionType, message string) []byte {
	headers := bytes.NewBuffer(nil)
	writeHeader := func(name, value string) {
		headers.WriteByte(byte(len(name)))
		headers.WriteString(name)
		headers.WriteByte(7)
		var length [2]byte
		binary.BigEndian.PutUint16(length[:], uint16(len(value)))
		headers.Write(length[:])
		headers.WriteString(value)
	}
	writeHeader(":message-type", "exception")
	writeHeader(":exception-type", exceptionType)
	payload := []byte(`{"message":"` + message + `"}`)
	total := 16 + headers.Len() + len(payload)
	frame := make([]byte, total)
	binary.BigEndian.PutUint32(frame[:4], uint32(total))
	binary.BigEndian.PutUint32(frame[4:8], uint32(headers.Len()))
	binary.BigEndian.PutUint32(frame[8:12], crc32.ChecksumIEEE(frame[:8]))
	copy(frame[12:], headers.Bytes())
	copy(frame[12+headers.Len():], payload)
	binary.BigEndian.PutUint32(frame[total-4:], crc32.ChecksumIEEE(frame[:total-4]))
	return frame
}

func appendFrames(frames ...[]byte) []byte {
	var body []byte
	for _, frame := range frames {
		body = append(body, frame...)
	}
	return body
}

func drainStream(t *testing.T, body []byte) ([]content.Chunk, stream.StreamResult, bool, error) {
	t.Helper()
	reader, err := (bedrockconverse.Codec{}).DecodeStream(&http.Response{Body: io.NopCloser(bytes.NewReader(body))})
	if err != nil {
		return nil, stream.StreamResult{}, false, err
	}
	defer reader.Close()
	var chunks []content.Chunk
	var terminal error
	for {
		chunk, err := reader.Next()
		if err != nil {
			terminal = err
			break
		}
		chunks = append(chunks, chunk)
	}
	result, ok := reader.Result()
	return chunks, result, ok, terminal
}

func TestDecodeStream_TextToolReasoningUsageAndTerminal(t *testing.T) {
	t.Parallel()

	body := appendFrames(
		eventFrame("messageStart", `{"role":"assistant"}`),
		eventFrame("contentBlockStart", `{"contentBlockIndex":0,"start":{}}`),
		eventFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"hello"}}`),
		eventFrame("contentBlockStop", `{"contentBlockIndex":0}`),
		eventFrame("contentBlockStart", `{"contentBlockIndex":1,"start":{"toolUse":{"toolUseId":"call-1","name":"lookup"}}}`),
		eventFrame("contentBlockDelta", `{"contentBlockIndex":1,"delta":{"toolUse":{"input":"{\"q\":\"go\"}"}}}`),
		eventFrame("contentBlockStop", `{"contentBlockIndex":1}`),
		eventFrame("contentBlockStart", `{"contentBlockIndex":2,"start":{}}`),
		eventFrame("contentBlockDelta", `{"contentBlockIndex":2,"delta":{"reasoningContent":{"text":"think"}}}`),
		eventFrame("contentBlockDelta", `{"contentBlockIndex":2,"delta":{"reasoningContent":{"signature":"sig"}}}`),
		eventFrame("contentBlockStop", `{"contentBlockIndex":2}`),
		eventFrame("messageStop", `{"stopReason":"tool_use"}`),
		eventFrame("metadata", `{"usage":{"inputTokens":11,"outputTokens":7,"cacheReadInputTokens":2,"cacheWriteInputTokens":1}}`),
	)
	chunks, result, ok, err := drainStream(t, body)
	if !errors.Is(err, io.EOF) {
		t.Fatalf("terminal error = %v, want io.EOF", err)
	}
	if !ok {
		t.Fatal("Result() unavailable after clean stream")
	}
	if len(chunks) != 5 {
		t.Fatalf("chunks = %#v, want text/tool/input/reasoning/signature", chunks)
	}
	if got := chunks[0].(*content.TextChunk).Text; got != "hello" {
		t.Errorf("text chunk = %q, want hello", got)
	}
	toolStart := chunks[1].(*content.ToolUseChunk)
	if toolStart.Index != 1 || toolStart.ID != "call-1" || toolStart.Name != "lookup" {
		t.Errorf("tool start = %#v, want call-1/lookup at 1", toolStart)
	}
	if got := chunks[2].(*content.ToolUseChunk).InputJSON; got != `{"q":"go"}` {
		t.Errorf("tool input chunk = %q, want complete input", got)
	}
	if got := chunks[3].(*content.ThinkingChunk).Thinking; got != "think" {
		t.Errorf("thinking chunk = %q, want think", got)
	}
	if got := chunks[4].(*content.ThinkingChunk).Signature; got != "sig" {
		t.Errorf("thinking signature chunk = %q, want sig", got)
	}
	if result.FinishReason != stream.FinishReasonToolUse {
		t.Errorf("FinishReason = %q, want tool_use", result.FinishReason)
	}
	wantUsage := &content.Usage{InputTokens: 11, OutputTokens: 7, CacheReadTokens: 2, CacheCreationTokens: 1}
	if result.Usage == nil || *result.Usage != *wantUsage {
		t.Errorf("Usage = %#v, want %#v", result.Usage, wantUsage)
	}
}

func TestDecodeStream_ToolInputFragments(t *testing.T) {
	t.Parallel()

	body := appendFrames(
		eventFrame("messageStart", `{"role":"assistant"}`),
		eventFrame("contentBlockStart", `{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"id","name":"tool"}}}`),
		eventFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"a\":"}}}`),
		eventFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"toolUse":{"input":"1}"}}}`),
		eventFrame("contentBlockStop", `{"contentBlockIndex":0}`),
		eventFrame("messageStop", `{"stopReason":"tool_use"}`),
	)
	chunks, _, _, err := drainStream(t, body)
	if !errors.Is(err, io.EOF) {
		t.Fatalf("terminal error = %v, want io.EOF", err)
	}
	if len(chunks) != 3 {
		t.Fatalf("chunks = %#v, want start plus two deltas", chunks)
	}
	if got := chunks[1].(*content.ToolUseChunk).InputJSON; got != `{"a":` {
		t.Errorf("first input fragment = %q", got)
	}
	if got := chunks[2].(*content.ToolUseChunk).InputJSON; got != "1}" {
		t.Errorf("second input fragment = %q", got)
	}
}

func TestDecodeStream_ExceptionAndMalformedPayload(t *testing.T) {
	t.Parallel()

	_, _, _, err := drainStream(t, exceptionFrame("modelStreamErrorException", "provider failed"))
	var apiErr *bedrockconverse.StreamAPIError
	if !errors.As(err, &apiErr) || apiErr.Type != "modelStreamErrorException" {
		t.Fatalf("exception error = %T (%v), want StreamAPIError", err, err)
	}

	body := appendFrames(
		eventFrame("messageStart", `{"role":"assistant"}`),
		eventFrame("contentBlockStart", `{`),
	)
	_, _, _, err = drainStream(t, body)
	var decodeErr *bedrockconverse.StreamDecodeError
	if !errors.As(err, &decodeErr) {
		t.Fatalf("malformed event error = %T (%v), want StreamDecodeError", err, err)
	}
}

func TestDecodeStream_InvalidOrderingAndDuplicateTerminal(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		body  []byte
		match string
	}{
		{
			name:  "delta before block start",
			body:  appendFrames(eventFrame("messageStart", `{"role":"assistant"}`), eventFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"bad"}}`)),
			match: "without start",
		},
		{
			name:  "duplicate message stop",
			body:  appendFrames(eventFrame("messageStart", `{"role":"assistant"}`), eventFrame("messageStop", `{"stopReason":"end_turn"}`), eventFrame("messageStop", `{"stopReason":"end_turn"}`)),
			match: "duplicate messageStop",
		},
		{
			name:  "multiple delta variants",
			body:  appendFrames(eventFrame("messageStart", `{"role":"assistant"}`), eventFrame("contentBlockStart", `{"contentBlockIndex":0,"start":{}}`), eventFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"bad","toolUse":{"input":"{}"}}}`)),
			match: "exactly one recognized variant",
		},
		{
			name:  "multiple reasoning delta variants",
			body:  appendFrames(eventFrame("messageStart", `{"role":"assistant"}`), eventFrame("contentBlockStart", `{"contentBlockIndex":0,"start":{}}`), eventFrame("contentBlockDelta", `{"contentBlockIndex":0,"delta":{"reasoningContent":{"text":"bad","signature":"sig"}}}`)),
			match: "reasoning content delta must contain exactly one",
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, _, _, err := drainStream(t, tc.body)
			var decodeErr *bedrockconverse.StreamDecodeError
			if !errors.As(err, &decodeErr) || !strings.Contains(err.Error(), tc.match) {
				t.Fatalf("error = %T (%v), want StreamDecodeError containing %q", err, err, tc.match)
			}
		})
	}
}

func TestDecodeStream_MissingMessageStopAndMissingMetadata(t *testing.T) {
	t.Parallel()

	missingStop := appendFrames(eventFrame("messageStart", `{"role":"assistant"}`), eventFrame("contentBlockStart", `{"contentBlockIndex":0,"start":{}}`), eventFrame("contentBlockStop", `{"contentBlockIndex":0}`))
	_, _, _, err := drainStream(t, missingStop)
	var decodeErr *bedrockconverse.StreamDecodeError
	if !errors.As(err, &decodeErr) || !strings.Contains(err.Error(), "messageStop") {
		t.Fatalf("missing stop error = %T (%v), want StreamDecodeError mentioning messageStop", err, err)
	}

	complete := appendFrames(eventFrame("messageStart", `{"role":"assistant"}`), eventFrame("messageStop", `{"stopReason":"end_turn"}`))
	_, result, ok, err := drainStream(t, complete)
	if !errors.Is(err, io.EOF) || !ok {
		t.Fatalf("missing metadata terminal = %v, result ok=%v; want clean EOF/result", err, ok)
	}
	if result.Usage != nil {
		t.Fatalf("Usage = %#v, want nil when metadata is absent", result.Usage)
	}
}

func TestDecodeStream_ClosesBodyAndRejectsNilResponse(t *testing.T) {
	t.Parallel()

	closed := false
	body := &closeSpy{Reader: bytes.NewReader(nil), closed: &closed}
	reader, err := (bedrockconverse.Codec{}).DecodeStream(&http.Response{Body: body})
	if err != nil {
		t.Fatalf("DecodeStream() error = %v", err)
	}
	_ = reader.Close()
	if !closed {
		t.Fatal("reader Close() did not close response body")
	}
	if _, err := (bedrockconverse.Codec{}).DecodeStream(nil); err == nil {
		t.Fatal("DecodeStream(nil) error = nil, want error")
	}
	if _, err := (bedrockconverse.Codec{}).DecodeStream(&http.Response{}); err == nil {
		t.Fatal("DecodeStream(response with nil Body) error = nil, want error")
	}
}

type closeSpy struct {
	io.Reader
	closed *bool
}

func (s *closeSpy) Close() error {
	*s.closed = true
	return nil
}
