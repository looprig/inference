package inference_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/looprig/inference"
)

const (
	maxOutputNameBytes        = 64
	maxOutputDescriptionBytes = 4096
	maxOutputSchemaBytes      = 1 << 20
	maxOutputSchemaDepth      = 64
	maxOutputSchemaProperties = 1024
)

func validOutput(schema string) inference.OutputSchema {
	return inference.OutputSchema{
		Name:        "result_v1",
		Description: "A typed result.",
		Schema:      json.RawMessage(schema),
		Strict:      true,
	}
}

func TestOutputSchemaClone(t *testing.T) {
	t.Parallel()

	original := validOutput(`{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"],"additionalProperties":false}`)
	clone := original.Clone()

	if clone.Name != original.Name || clone.Description != original.Description || clone.Strict != original.Strict {
		t.Fatalf("Clone() metadata = %+v, want %+v", clone, original)
	}
	if string(clone.Schema) != string(original.Schema) {
		t.Fatalf("Clone().Schema = %q, want %q", clone.Schema, original.Schema)
	}
	clone.Schema[0] = '['
	if original.Schema[0] != '{' {
		t.Fatal("Clone().Schema aliases the original schema")
	}

	var zero inference.OutputSchema
	if got := zero.Clone(); got.Schema != nil {
		t.Fatalf("zero Clone().Schema = %#v, want nil", got.Schema)
	}
	empty := inference.OutputSchema{Schema: json.RawMessage{}}
	if got := empty.Clone(); got.Schema == nil {
		t.Fatal("Clone() changed an empty non-nil schema to nil")
	}
}

func TestValidateOutputSchemaAcceptsPortableSubset(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		output inference.OutputSchema
	}{
		{
			name: "nested objects arrays and scalar enums",
			output: validOutput(`{
				"type":"object",
				"description":"root",
				"properties":{
					"kind":{"type":"string","enum":["report","summary"]},
					"ready":{"type":"boolean","enum":[true,false]},
					"score":{"type":"number","enum":[1,2.5,-3e2]},
					"count":{"type":"integer","enum":[0,1.0,2e1]},
					"items":{"type":"array","items":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":false}}
				},
				"required":["kind","ready","score","count","items"],
				"additionalProperties":false
			}`),
		},
		{
			name:   "empty object",
			output: validOutput(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`),
		},
		{
			name: "provider safe name boundary",
			output: func() inference.OutputSchema {
				o := validOutput(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`)
				o.Name = "_" + strings.Repeat("a", maxOutputNameBytes-1)
				return o
			}(),
		},
		{
			name: "description boundary",
			output: func() inference.OutputSchema {
				o := validOutput(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`)
				o.Description = strings.Repeat("d", maxOutputDescriptionBytes)
				return o
			}(),
		},
		{
			name: "schema size boundary",
			output: func() inference.OutputSchema {
				o := validOutput(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`)
				o.Schema = append(append(json.RawMessage(nil), o.Schema...), []byte(strings.Repeat(" ", maxOutputSchemaBytes-len(o.Schema)))...)
				return o
			}(),
		},
		{
			name:   "depth boundary",
			output: validOutput(nestedArraySchema(maxOutputSchemaDepth - 2)),
		},
		{
			name:   "property count boundary",
			output: validOutput(objectWithProperties(maxOutputSchemaProperties)),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if err := inference.ValidateOutputSchema(tt.output); err != nil {
				t.Fatalf("ValidateOutputSchema() error = %v", err)
			}
		})
	}
}

func TestValidateOutputSchemaRejectsInvalidOutputMetadata(t *testing.T) {
	t.Parallel()

	base := validOutput(`{"type":"object","properties":{},"required":[],"additionalProperties":false}`)
	tests := []struct {
		name   string
		mutate func(*inference.OutputSchema)
		field  inference.SchemaValidationField
		reason inference.SchemaValidationReason
	}{
		{name: "empty name", mutate: func(o *inference.OutputSchema) { o.Name = "" }, field: inference.SchemaFieldName, reason: inference.SchemaReasonEmpty},
		{name: "starts with digit", mutate: func(o *inference.OutputSchema) { o.Name = "1result" }, field: inference.SchemaFieldName, reason: inference.SchemaReasonInvalid},
		{name: "contains dot", mutate: func(o *inference.OutputSchema) { o.Name = "result.v1" }, field: inference.SchemaFieldName, reason: inference.SchemaReasonInvalid},
		{name: "contains non ASCII", mutate: func(o *inference.OutputSchema) { o.Name = "résultat" }, field: inference.SchemaFieldName, reason: inference.SchemaReasonInvalid},
		{name: "name too long", mutate: func(o *inference.OutputSchema) { o.Name = "a" + strings.Repeat("b", maxOutputNameBytes) }, field: inference.SchemaFieldName, reason: inference.SchemaReasonTooLong},
		{name: "reserved name", mutate: func(o *inference.OutputSchema) { o.Name = inference.StructuredOutputToolName }, field: inference.SchemaFieldName, reason: inference.SchemaReasonReserved},
		{name: "description too long", mutate: func(o *inference.OutputSchema) { o.Description = strings.Repeat("d", maxOutputDescriptionBytes+1) }, field: inference.SchemaFieldDescription, reason: inference.SchemaReasonTooLong},
		{name: "description invalid UTF-8", mutate: func(o *inference.OutputSchema) { o.Description = string([]byte{0xff}) }, field: inference.SchemaFieldDescription, reason: inference.SchemaReasonInvalidUTF8},
		{name: "schema too large", mutate: func(o *inference.OutputSchema) {
			o.Schema = json.RawMessage(strings.Repeat(" ", maxOutputSchemaBytes+1))
		}, field: inference.SchemaFieldSchema, reason: inference.SchemaReasonTooLarge},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			output := base.Clone()
			tt.mutate(&output)
			assertSchemaValidationError(t, inference.ValidateOutputSchema(output), tt.field, tt.reason)
		})
	}
}

func TestValidateOutputSchemaRejectsInvalidSchemas(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		schema string
		field  inference.SchemaValidationField
		reason inference.SchemaValidationReason
	}{
		{name: "nil schema", schema: "", field: inference.SchemaFieldSchema, reason: inference.SchemaReasonMalformed},
		{name: "malformed JSON", schema: `{`, field: inference.SchemaFieldSchema, reason: inference.SchemaReasonMalformed},
		{name: "trailing JSON", schema: `{"type":"object","properties":{},"required":[],"additionalProperties":false}{}`, field: inference.SchemaFieldSchema, reason: inference.SchemaReasonMalformed},
		{name: "non object root", schema: `[]`, field: inference.SchemaFieldSchema, reason: inference.SchemaReasonRootNotObject},
		{name: "missing type", schema: `{"properties":{},"required":[],"additionalProperties":false}`, field: inference.SchemaFieldType, reason: inference.SchemaReasonMissing},
		{name: "description null", schema: `{"type":"object","description":null,"properties":{},"required":[],"additionalProperties":false}`, field: inference.SchemaFieldDescription, reason: inference.SchemaReasonInvalid},
		{name: "null type", schema: `{"type":null}`, field: inference.SchemaFieldType, reason: inference.SchemaReasonInvalid},
		{name: "union type", schema: `{"type":["object","null"]}`, field: inference.SchemaFieldType, reason: inference.SchemaReasonInvalid},
		{name: "unsupported type", schema: rootWithProperty(`{"type":"null"}`), field: inference.SchemaFieldType, reason: inference.SchemaReasonUnsupported},
		{name: "unknown keyword", schema: `{"type":"object","properties":{},"required":[],"additionalProperties":false,"title":"x"}`, field: inference.SchemaFieldKeyword, reason: inference.SchemaReasonUnknownKeyword},
		{name: "duplicate schema keyword", schema: `{"type":"object","type":"object","properties":{},"required":[],"additionalProperties":false}`, field: inference.SchemaFieldKeyword, reason: inference.SchemaReasonDuplicate},
		{name: "duplicate property name", schema: `{"type":"object","properties":{"x":{"type":"string"},"x":{"type":"boolean"}},"required":["x"],"additionalProperties":false}`, field: inference.SchemaFieldProperties, reason: inference.SchemaReasonDuplicate},
		{name: "escaped duplicate property name", schema: `{"type":"object","properties":{"x":{"type":"string"},"\u0078":{"type":"boolean"}},"required":["x"],"additionalProperties":false}`, field: inference.SchemaFieldProperties, reason: inference.SchemaReasonDuplicate},
		{name: "duplicate nested schema keyword", schema: `{"type":"object","properties":{"x":{"type":"object","properties":{"y":{"type":"string","type":"string"}},"required":["y"],"additionalProperties":false}},"required":["x"],"additionalProperties":false}`, field: inference.SchemaFieldKeyword, reason: inference.SchemaReasonDuplicate},
		{name: "ref keyword", schema: `{"type":"object","properties":{},"required":[],"additionalProperties":false,"$ref":"#"}`, field: inference.SchemaFieldKeyword, reason: inference.SchemaReasonUnknownKeyword},
		{name: "composition keyword", schema: `{"type":"object","properties":{},"required":[],"additionalProperties":false,"anyOf":[]}`, field: inference.SchemaFieldKeyword, reason: inference.SchemaReasonUnknownKeyword},
		{name: "numeric constraint", schema: `{"type":"object","properties":{"n":{"type":"number","minimum":0}},"required":["n"],"additionalProperties":false}`, field: inference.SchemaFieldKeyword, reason: inference.SchemaReasonUnknownKeyword},
		{name: "string constraint", schema: `{"type":"object","properties":{"s":{"type":"string","maxLength":3}},"required":["s"],"additionalProperties":false}`, field: inference.SchemaFieldKeyword, reason: inference.SchemaReasonUnknownKeyword},
		{name: "object missing additionalProperties", schema: `{"type":"object","properties":{},"required":[]}`, field: inference.SchemaFieldAdditionalProperties, reason: inference.SchemaReasonMissing},
		{name: "object additionalProperties true", schema: `{"type":"object","properties":{},"required":[],"additionalProperties":true}`, field: inference.SchemaFieldAdditionalProperties, reason: inference.SchemaReasonMustBeFalse},
		{name: "object additionalProperties schema", schema: `{"type":"object","properties":{},"required":[],"additionalProperties":{}}`, field: inference.SchemaFieldAdditionalProperties, reason: inference.SchemaReasonMustBeFalse},
		{name: "object properties wrong type", schema: `{"type":"object","properties":[],"required":[],"additionalProperties":false}`, field: inference.SchemaFieldProperties, reason: inference.SchemaReasonInvalid},
		{name: "object property is not schema", schema: `{"type":"object","properties":{"x":null},"required":["x"],"additionalProperties":false}`, field: inference.SchemaFieldSchema, reason: inference.SchemaReasonInvalid},
		{name: "object missing required", schema: `{"type":"object","properties":{"x":{"type":"string"}},"additionalProperties":false}`, field: inference.SchemaFieldRequired, reason: inference.SchemaReasonMissing},
		{name: "object required wrong type", schema: `{"type":"object","properties":{},"required":"x","additionalProperties":false}`, field: inference.SchemaFieldRequired, reason: inference.SchemaReasonInvalid},
		{name: "declared property omitted from required", schema: `{"type":"object","properties":{"x":{"type":"string"}},"required":[],"additionalProperties":false}`, field: inference.SchemaFieldRequired, reason: inference.SchemaReasonMissing},
		{name: "required duplicate", schema: `{"type":"object","properties":{"x":{"type":"string"}},"required":["x","x"],"additionalProperties":false}`, field: inference.SchemaFieldRequired, reason: inference.SchemaReasonDuplicate},
		{name: "required unknown property", schema: `{"type":"object","properties":{},"required":["x"],"additionalProperties":false}`, field: inference.SchemaFieldRequired, reason: inference.SchemaReasonUnknownProperty},
		{name: "array missing items", schema: rootWithProperty(`{"type":"array"}`), field: inference.SchemaFieldItems, reason: inference.SchemaReasonMissing},
		{name: "array tuple items", schema: rootWithProperty(`{"type":"array","items":[{"type":"string"}]}`), field: inference.SchemaFieldItems, reason: inference.SchemaReasonInvalid},
		{name: "array null items", schema: rootWithProperty(`{"type":"array","items":null}`), field: inference.SchemaFieldItems, reason: inference.SchemaReasonInvalid},
		{name: "enum on object", schema: `{"type":"object","properties":{},"required":[],"additionalProperties":false,"enum":[]}`, field: inference.SchemaFieldEnum, reason: inference.SchemaReasonUnsupported},
		{name: "enum on array", schema: rootWithProperty(`{"type":"array","items":{"type":"string"},"enum":[]}`), field: inference.SchemaFieldEnum, reason: inference.SchemaReasonUnsupported},
		{name: "enum wrong JSON type", schema: rootWithProperty(`{"type":"string","enum":"x"}`), field: inference.SchemaFieldEnum, reason: inference.SchemaReasonInvalid},
		{name: "enum null", schema: rootWithProperty(`{"type":"string","enum":null}`), field: inference.SchemaFieldEnum, reason: inference.SchemaReasonInvalid},
		{name: "enum empty", schema: rootWithProperty(`{"type":"string","enum":[]}`), field: inference.SchemaFieldEnum, reason: inference.SchemaReasonInvalid},
		{name: "string enum mismatch", schema: rootWithProperty(`{"type":"string","enum":[1]}`), field: inference.SchemaFieldEnum, reason: inference.SchemaReasonTypeMismatch},
		{name: "string enum null mismatch", schema: rootWithProperty(`{"type":"string","enum":[null]}`), field: inference.SchemaFieldEnum, reason: inference.SchemaReasonTypeMismatch},
		{name: "boolean enum mismatch", schema: rootWithProperty(`{"type":"boolean","enum":["true"]}`), field: inference.SchemaFieldEnum, reason: inference.SchemaReasonTypeMismatch},
		{name: "number enum mismatch", schema: rootWithProperty(`{"type":"number","enum":[true]}`), field: inference.SchemaFieldEnum, reason: inference.SchemaReasonTypeMismatch},
		{name: "integer enum fractional", schema: rootWithProperty(`{"type":"integer","enum":[1.5]}`), field: inference.SchemaFieldEnum, reason: inference.SchemaReasonTypeMismatch},
		{name: "scalar properties keyword", schema: rootWithProperty(`{"type":"string","properties":{}}`), field: inference.SchemaFieldProperties, reason: inference.SchemaReasonUnsupported},
		{name: "scalar items keyword", schema: rootWithProperty(`{"type":"string","items":{"type":"string"}}`), field: inference.SchemaFieldItems, reason: inference.SchemaReasonUnsupported},
		{name: "schema too deep", schema: nestedArraySchema(maxOutputSchemaDepth - 1), field: inference.SchemaFieldSchema, reason: inference.SchemaReasonTooDeep},
		{name: "too many properties", schema: objectWithProperties(maxOutputSchemaProperties + 1), field: inference.SchemaFieldProperties, reason: inference.SchemaReasonTooManyProperties},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assertSchemaValidationError(t, inference.ValidateOutputSchema(validOutput(tt.schema)), tt.field, tt.reason)
		})
	}
}

func TestValidateOutputSchemaRejectsNonStringRequiredMembers(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		properties string
		required   string
	}{
		{
			name:       "null cannot name empty property",
			properties: `"":{"type":"string"}`,
			required:   `null`,
		},
		{name: "boolean", required: `true`},
		{name: "number", required: `1`},
		{name: "object", required: `{"value":"schema-secret-object"}`},
		{name: "array", required: `["schema-secret-array"]`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			const secret = "schema-secret-required-member"
			schema := fmt.Sprintf(
				`{"type":"object","description":%q,"properties":{%s},"required":[%s],"additionalProperties":false}`,
				secret,
				tt.properties,
				tt.required,
			)
			err := inference.ValidateOutputSchema(validOutput(schema))
			assertSchemaValidationError(t, err, inference.SchemaFieldRequired, inference.SchemaReasonInvalid)
			if strings.Contains(err.Error(), secret) || len(err.Error()) > inference.MaxStructuredOutputDiagnosticBytes {
				t.Fatalf("validation diagnostic is unbounded or exposes schema bytes: %q", err)
			}
		})
	}
}

func TestValidateOutputSchemaDoesNotExposeSchemaBytes(t *testing.T) {
	t.Parallel()

	const secret = "schema-secret-that-must-not-be-logged"
	err := inference.ValidateOutputSchema(validOutput(`{"type":"object","properties":{},"required":[],"additionalProperties":false,"` + secret + `":true}`))
	if err == nil {
		t.Fatal("ValidateOutputSchema() error = nil, want validation error")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("error contains raw schema content: %q", err)
	}
	if len(err.Error()) > 256 {
		t.Fatalf("error length = %d, want bounded metadata", len(err.Error()))
	}
}

func assertSchemaValidationError(t *testing.T, err error, field inference.SchemaValidationField, reason inference.SchemaValidationReason) {
	t.Helper()
	if err == nil {
		t.Fatal("ValidateOutputSchema() error = nil, want validation error")
	}
	var validationErr *inference.SchemaValidationError
	if !errors.As(err, &validationErr) {
		t.Fatalf("error type = %T, want *inference.SchemaValidationError", err)
	}
	if validationErr.Field != field || validationErr.ReasonCode != reason {
		t.Fatalf("validation error = {Field:%q ReasonCode:%q}, want {Field:%q ReasonCode:%q}", validationErr.Field, validationErr.ReasonCode, field, reason)
	}
}

func rootWithProperty(propertySchema string) string {
	return `{"type":"object","properties":{"value":` + propertySchema + `},"required":["value"],"additionalProperties":false}`
}

func nestedArraySchema(arrayCount int) string {
	node := `{"type":"string"}`
	for range arrayCount {
		node = `{"type":"array","items":` + node + `}`
	}
	return rootWithProperty(node)
}

func objectWithProperties(count int) string {
	var properties strings.Builder
	var required strings.Builder
	for i := range count {
		if i > 0 {
			properties.WriteByte(',')
			required.WriteByte(',')
		}
		name := fmt.Sprintf("p%d", i)
		fmt.Fprintf(&properties, "%q:{\"type\":\"string\"}", name)
		fmt.Fprintf(&required, "%q", name)
	}
	return `{"type":"object","properties":{` + properties.String() + `},"required":[` + required.String() + `],"additionalProperties":false}`
}
