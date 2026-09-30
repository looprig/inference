// Package inferencetest provides a scripted, in-memory inference.Client for
// running and testing agents without a provider, an API key or a network.
//
// A Client answers each call with the next Step of its script:
//
//	client := inferencetest.New(
//		inferencetest.ToolCall("read_file", `{"path":"go.mod"}`),
//		inferencetest.Text("The module is example.com/demo."),
//	)
//	loop.WithInference(client, inferencetest.Model())
//
// Both Invoke and Stream are implemented, and they agree: folding a Stream
// through core's content/streamaccumulator yields the same message Invoke
// returns. Stream emits the chunk shapes a real codec produces — text in
// several TextChunk deltas, a tool call as a ToolUseChunk carrying its ID and
// Name followed by InputJSON fragments, reasoning as indexed ThinkingChunk
// deltas — and completes with an authoritative stream.StreamResult carrying the
// finish reason, the model name and token usage.
//
// Like a real client, every call first validates the request
// (model.Model.Validate, then inference.ValidateRequestFeatures) and fails
// without consuming a step when it is invalid, so a request no provider would
// accept does not pass silently here. Use Model for a descriptor that
// validates.
//
// Every request that reaches the script is recorded, so a test can assert what
// the model saw (Requests), and a step can check the request it answers
// (Step.Expect). A Client is safe for concurrent use.
package inferencetest
