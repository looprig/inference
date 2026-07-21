package inference_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	model "github.com/looprig/inference/model"
)

func imgBlock() content.Block {
	return &content.ImageBlock{MediaType: content.MediaTypeImagePNG, Source: content.ImageSource{Data: []byte{0x89}}}
}

// ValidateRequestFeatures must reject image blocks for a model that does not
// advertise AcceptsImages — fail secure at the neutral layer instead of
// relying on each caller (or provider) to notice.
func TestValidateRequestFeaturesImageCapability(t *testing.T) {
	t.Parallel()

	imgCaps := model.Capabilities{AcceptsImages: true}

	tests := []struct {
		name        string
		req         inference.Request
		wantImagErr bool
	}{
		{
			name: "image in user message without capability",
			req: inference.Request{
				Model: model.Model{Name: "text-only"},
				Messages: content.AgenticMessages{
					&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{imgBlock()}}},
				},
			},
			wantImagErr: true,
		},
		{
			name: "image in user message with capability",
			req: inference.Request{
				Model: model.Model{Name: "vision", Caps: imgCaps},
				Messages: content.AgenticMessages{
					&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{imgBlock()}}},
				},
			},
		},
		{
			name: "image in tool result message without capability",
			req: inference.Request{
				Model: model.Model{Name: "text-only"},
				Messages: content.AgenticMessages{
					&content.ToolResultMessage{
						Message:   content.Message{Role: content.RoleTool, Blocks: []content.Block{imgBlock()}},
						ToolUseID: "call_1",
					},
				},
			},
			wantImagErr: true,
		},
		{
			name: "image nested in a tool result block without capability",
			req: inference.Request{
				Model: model.Model{Name: "text-only"},
				Messages: content.AgenticMessages{
					&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{
						&content.ToolResultBlock{ToolUseID: "call_1", Content: []content.Block{imgBlock()}},
					}}},
				},
			},
			wantImagErr: true,
		},
		{
			name: "text only without capability",
			req: inference.Request{
				Model: model.Model{Name: "text-only"},
				Messages: content.AgenticMessages{
					&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{&content.TextBlock{Text: "hi"}}}},
				},
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := inference.ValidateRequestFeatures(tt.req)
			if tt.wantImagErr {
				var target *inference.ImageInputUnsupportedError
				if !errors.As(err, &target) {
					t.Fatalf("ValidateRequestFeatures() error = %T %v, want *ImageInputUnsupportedError", err, err)
				}
				if target.Model != tt.req.Model.Name {
					t.Errorf("ImageInputUnsupportedError.Model = %q, want %q", target.Model, tt.req.Model.Name)
				}
				return
			}
			if err != nil {
				t.Fatalf("ValidateRequestFeatures() error = %v, want nil", err)
			}
		})
	}
}

// The Model diagnostic on ImageInputUnsupportedError is bounded like the
// structured-output errors — a hostile model name must not be retained whole.
func TestImageInputUnsupportedErrorBoundsModelMetadata(t *testing.T) {
	t.Parallel()

	long := strings.Repeat("m", inference.MaxStructuredOutputDiagnosticBytes*2)
	req := inference.Request{
		Model: model.Model{Name: long},
		Messages: content.AgenticMessages{
			&content.UserMessage{Message: content.Message{Role: content.RoleUser, Blocks: []content.Block{imgBlock()}}},
		},
	}
	err := inference.ValidateRequestFeatures(req)
	var target *inference.ImageInputUnsupportedError
	if !errors.As(err, &target) {
		t.Fatalf("error = %T %v, want *ImageInputUnsupportedError", err, err)
	}
	if len(target.Model) > inference.MaxStructuredOutputDiagnosticBytes {
		t.Errorf("Model length = %d, want <= %d", len(target.Model), inference.MaxStructuredOutputDiagnosticBytes)
	}
}
