package bedrockconverse

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/internal/usagenorm"
	stream "github.com/looprig/inference/stream"
	usage "github.com/looprig/inference/usage"
)

// DecodeResponse parses a native Bedrock Converse response into the shared
// inference response. Bedrock does not echo the model ID in this envelope, so
// Model is intentionally left empty for the provider client to fill from the
// bound request.
func DecodeResponse(body []byte) (*inference.Response, error) {
	var wire converseResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, &DecodeError{Reason: "unmarshal response body", Err: err}
	}
	if wire.Output == nil || wire.Output.Message == nil {
		return nil, &DecodeError{Reason: "response is missing output.message"}
	}
	if wire.Output.Message.Role != roleAssistant {
		return nil, &DecodeError{Reason: "output.message role is not assistant"}
	}

	blocks, err := decodeContentBlocks(wire.Output.Message.Content)
	if err != nil {
		return nil, err
	}
	normalized, err := normalizeUsage(wire.Usage)
	if err != nil {
		return nil, err
	}
	var messageUsage *content.Usage
	if normalized != nil {
		copyUsage := *normalized
		messageUsage = &copyUsage
	}
	return &inference.Response{
		Message: &content.AIMessage{
			Message: content.Message{Role: content.RoleAssistant, Blocks: blocks},
			Usage:   messageUsage,
		},
		Usage:        normalized,
		FinishReason: mapFinishReason(wire.StopReason),
	}, nil
}

func normalizeUsage(wire *responseUsage) (*usage.Usage, error) {
	if wire == nil {
		return nil, nil
	}
	input, err := wire.InputTokens.TokenCount(usagenorm.FieldInputTokens)
	if err != nil {
		return nil, err
	}
	output, err := wire.OutputTokens.TokenCount(usagenorm.FieldOutputTokens)
	if err != nil {
		return nil, err
	}
	// AWS marks only inputTokens, outputTokens and totalTokens @required on
	// com.amazonaws.bedrockruntime#TokenUsage; cacheReadInputTokens and
	// cacheWriteInputTokens carry no @required trait, so a conforming response
	// may omit them or send them as null on any turn that read from or wrote to
	// no cache. OptionalTokenCount maps null to zero while keeping the strict
	// numeric validation for present values; TokenCount would discard an
	// otherwise complete response over an accounting field. Identical treatment
	// to codec/anthropicapi for the identical two fields.
	cacheRead, err := wire.CacheReadInputTokens.OptionalTokenCount(usagenorm.FieldCacheReadTokens)
	if err != nil {
		return nil, err
	}
	cacheWrite, err := wire.CacheWriteInputTokens.OptionalTokenCount(usagenorm.FieldCacheCreationTokens)
	if err != nil {
		return nil, err
	}
	normalized := usage.Usage{
		InputTokens:         input,
		OutputTokens:        output,
		CacheReadTokens:     cacheRead,
		CacheCreationTokens: cacheWrite,
	}
	return &normalized, nil
}

func decodeContentBlocks(blocks []converseContentBlock) ([]content.Block, error) {
	decoded := make([]content.Block, 0, len(blocks))
	for _, block := range blocks {
		decodedBlock, err := decodeContentBlock(block)
		if err != nil {
			return nil, err
		}
		decoded = append(decoded, decodedBlock)
	}
	return decoded, nil
}

func decodeContentBlock(block converseContentBlock) (content.Block, error) {
	if variants := contentBlockVariantCount(block); variants != 1 {
		return nil, &DecodeError{Reason: "content block must contain exactly one recognized variant"}
	}
	switch {
	case block.Text != nil:
		return &content.TextBlock{Text: *block.Text}, nil
	case block.Image != nil:
		return decodeImage(block.Image)
	case block.Document != nil:
		return decodeDocument(block.Document)
	case block.Audio != nil:
		return decodeAudio(block.Audio)
	case block.ReasoningContent != nil:
		reasoning := block.ReasoningContent
		reasoningVariants := 0
		if reasoning.ReasoningText != nil {
			reasoningVariants++
		}
		if len(reasoning.RedactedContent) > 0 {
			reasoningVariants++
		}
		if reasoningVariants != 1 {
			return nil, &DecodeError{Reason: "reasoningContent must contain exactly one recognized variant"}
		}
		if len(reasoning.RedactedContent) > 0 {
			encoded, _ := json.Marshal(base64.StdEncoding.EncodeToString(reasoning.RedactedContent))
			return content.NewThinkingBlock("", "", encoded, providerStateFormatBedrockRedacted), nil
		}
		if reasoning.ReasoningText.Text == nil {
			return nil, &DecodeError{Reason: "reasoningText is missing text"}
		}
		// Stamped with THIS dialect as it comes off the wire. Converse fronts
		// the same Claude models as the Anthropic Messages API and the
		// signatures are not interchangeable, so the label is the only thing
		// that keeps the two apart once the block is in the neutral transcript.
		signature := reasoning.ReasoningText.Signature
		return content.NewSignedThinkingBlock(
			*reasoning.ReasoningText.Text, signature, signatureFormatFor(signature), nil, ""), nil
	case block.ToolUse != nil:
		input, err := decodeToolInput(block.ToolUse.Input)
		if err != nil {
			return nil, err
		}
		if block.ToolUse.ToolUseID == "" || block.ToolUse.Name == "" {
			return nil, &DecodeError{Reason: "toolUse is missing toolUseId or name"}
		}
		return &content.ToolUseBlock{ID: block.ToolUse.ToolUseID, Name: block.ToolUse.Name, Input: input}, nil
	case block.ToolResult != nil:
		return decodeToolResult(block.ToolResult)
	default:
		return nil, &DecodeError{Reason: "content block has no recognized variant"}
	}
}

func contentBlockVariantCount(block converseContentBlock) int {
	count := 0
	if block.Text != nil {
		count++
	}
	if block.Image != nil {
		count++
	}
	if block.Document != nil {
		count++
	}
	if block.Audio != nil {
		count++
	}
	if block.ReasoningContent != nil {
		count++
	}
	if block.ToolUse != nil {
		count++
	}
	if block.ToolResult != nil {
		count++
	}
	return count
}

func decodeToolInput(raw json.RawMessage) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || !json.Valid(trimmed) || trimmed[0] != '{' {
		return nil, &DecodeError{Reason: "toolUse.input is not a JSON object"}
	}
	return append(json.RawMessage(nil), trimmed...), nil
}

func decodeImage(image *imageContent) (*content.ImageBlock, error) {
	if image == nil || image.Format == "" || len(image.Source.Bytes) == 0 {
		return nil, &DecodeError{Reason: "image content block is incomplete"}
	}
	if imageFormat(content.MediaType("image/"+strings.ToLower(image.Format))) == "" {
		return nil, &DecodeError{Reason: "image content block has unsupported format"}
	}
	return &content.ImageBlock{
		MediaType: content.MediaType("image/" + strings.ToLower(image.Format)),
		Source:    content.ImageSource{Data: append([]byte(nil), image.Source.Bytes...)},
	}, nil
}

func decodeDocument(document *documentContent) (*content.DocumentBlock, error) {
	if document == nil || document.Format == "" || document.Name == "" {
		return nil, &DecodeError{Reason: "document content block is incomplete"}
	}
	if !isDocumentFormat(document.Format) {
		return nil, &DecodeError{Reason: "document content block has unsupported format"}
	}
	if err := validateDocumentName(document.Name); err != nil {
		return nil, &DecodeError{Reason: "document content block has invalid name"}
	}
	decoded := &content.DocumentBlock{
		MediaType: documentMediaType(document.Format),
		Name:      document.Name,
	}
	// DocumentSource has four members. bytes and text map onto
	// content.DocumentBlock's two payload fields; s3Location and content have
	// no neutral counterpart at all, so they are named rather than folded into
	// the arity error — an operator reading "must contain exactly one variant"
	// against a source that plainly holds exactly one has no way to tell that
	// the member itself is the problem.
	hasBytes := len(document.Source.Bytes) > 0
	hasText := document.Source.Text != nil
	switch {
	case document.Source.S3Location != nil:
		return nil, &DecodeError{Reason: "document source s3Location has no neutral representation; content.DocumentBlock carries a payload, not a storage reference"}
	case len(document.Source.Content) > 0:
		return nil, &DecodeError{Reason: "document source content blocks have no neutral representation; content.DocumentBlock carries a single body, not a block list"}
	case hasBytes == hasText:
		return nil, &DecodeError{Reason: "document content block source must contain exactly one variant"}
	case hasBytes:
		decoded.Data = append([]byte(nil), document.Source.Bytes...)
	default:
		decoded.Text = *document.Source.Text
	}
	return decoded, nil
}

// decodeAudio maps Converse's AudioBlock onto a neutral audio block.
//
// The format map is audioFormat read backwards, and it is an allowlist for the
// same reason: AudioFormat carries members the shared vocabulary has no name
// for (pcm, opus, mka, mkv, mpga, x-aac), and AWS may add more. Synthesising
// "audio/pcm" the way decodeImage synthesises "image/png" would mint a media
// type no other codec in this module recognizes, which then fails at the far
// end of a cross-provider replay instead of here.
func decodeAudio(audio *audioContent) (*content.AudioBlock, error) {
	if audio == nil || audio.Format == "" {
		return nil, &DecodeError{Reason: "audio content block is incomplete"}
	}
	mediaType := audioMediaType(audio.Format)
	if mediaType == "" {
		return nil, &DecodeError{Reason: "audio content block format " + audio.Format + " has no neutral media type"}
	}
	if audio.Source.S3Location != nil {
		return nil, &DecodeError{Reason: "audio source s3Location has no neutral representation; content.AudioBlock carries a payload, not a storage reference"}
	}
	if len(audio.Source.Bytes) == 0 {
		return nil, &DecodeError{Reason: "audio content block source must contain exactly one variant"}
	}
	return &content.AudioBlock{MediaType: mediaType, Data: append([]byte(nil), audio.Source.Bytes...)}, nil
}

// audioMediaType maps an AudioFormat member back to a shared media type, or ""
// when the vocabulary has no name for it. Two enum members can select the same
// media type — "mp3" and "mpeg" are both audio/mpeg, "mp4" and "m4a" are both
// audio/mp4 — which is why the forward map documents which of each pair it
// emits.
func audioMediaType(format string) content.MediaType {
	switch strings.ToLower(format) {
	case audioFormatMP3, "mpeg":
		return content.MediaTypeAudioMPEG
	case audioFormatWAV:
		return content.MediaTypeAudioWAV
	case audioFormatOGG:
		return content.MediaTypeAudioOGG
	case audioFormatFLAC:
		return content.MediaTypeAudioFLAC
	case audioFormatAAC:
		return content.MediaTypeAudioAAC
	case audioFormatMP4, "m4a":
		return content.MediaTypeAudioMP4
	case audioFormatWebM:
		return content.MediaTypeAudioWebM
	default:
		return ""
	}
}

func decodeToolResult(result *toolResultContent) (*content.ToolResultBlock, error) {
	if result.ToolUseID == "" {
		return nil, &DecodeError{Reason: "toolResult is missing toolUseId"}
	}
	if len(result.Content) == 0 {
		return nil, &DecodeError{Reason: "toolResult content must not be empty"}
	}
	status := result.Status
	if status != "" && status != toolResultStatusSuccess && status != toolResultStatusError {
		return nil, &DecodeError{Reason: "toolResult has unknown status"}
	}
	blocks := make([]content.Block, 0, len(result.Content))
	for _, block := range result.Content {
		if variants := toolResultBlockVariantCount(block); variants != 1 {
			return nil, &DecodeError{Reason: "tool result content block must contain exactly one recognized variant"}
		}
		switch {
		case block.Text != nil:
			blocks = append(blocks, &content.TextBlock{Text: *block.Text})
		case block.Image != nil:
			image, err := decodeImage(block.Image)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, image)
		case block.Document != nil:
			document, err := decodeDocument(block.Document)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, document)
		}
	}
	return &content.ToolResultBlock{ToolUseID: result.ToolUseID, Content: blocks, IsError: status == toolResultStatusError}, nil
}

func toolResultBlockVariantCount(block toolResultBlock) int {
	count := 0
	if block.Text != nil {
		count++
	}
	if block.Image != nil {
		count++
	}
	if block.Document != nil {
		count++
	}
	return count
}

func documentMediaType(format string) content.MediaType {
	switch strings.ToLower(format) {
	case "pdf":
		return content.MediaTypeDocumentPDF
	case "txt":
		return content.MediaTypeDocumentText
	case "html":
		return content.MediaTypeDocumentHTML
	case "csv":
		return content.MediaTypeDocumentCSV
	case "md":
		return content.MediaTypeDocumentMarkdown
	case "docx":
		return content.MediaTypeDocumentDOCX
	case "xlsx":
		return content.MediaTypeDocumentXLSX
	case "doc":
		return content.MediaType("application/msword")
	case "xls":
		return content.MediaType("application/vnd.ms-excel")
	default:
		return content.MediaType("application/" + strings.ToLower(format))
	}
}

func isDocumentFormat(format string) bool {
	switch strings.ToLower(format) {
	case "pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md":
		return true
	default:
		return false
	}
}

func mapFinishReason(reason string) stream.FinishReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return stream.FinishReasonStop
	case "max_tokens", "model_context_window_exceeded":
		return stream.FinishReasonLength
	case "tool_use":
		return stream.FinishReasonToolUse
	case "content_filtered", "guardrail_intervened":
		return stream.FinishReasonContentFilter
	default:
		return stream.FinishReasonUnknown
	}
}
