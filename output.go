package inference

import (
	"bytes"
	"encoding/json"
	"strconv"
	"strings"
	"unicode/utf8"
)

const (
	StructuredOutputToolName = "_looprig_final_output"
	StructuredOutputRevision = "structured-output/v1"

	maxOutputNameBytes        = 64
	maxOutputDescriptionBytes = 4096
	maxOutputSchemaBytes      = 1 << 20
	maxOutputSchemaDepth      = 64
	maxOutputSchemaProperties = 1024
)

// OutputSchema is a provider-neutral request for one schema-constrained JSON
// object. Description must be valid UTF-8 and is limited to 4096 bytes. Schema
// must satisfy the portable subset checked by ValidateOutputSchema.
type OutputSchema struct {
	Name        string
	Description string
	Schema      json.RawMessage
	Strict      bool
}

// Clone returns an independent copy of the output schema.
func (o OutputSchema) Clone() OutputSchema {
	clone := o
	if o.Schema != nil {
		clone.Schema = make(json.RawMessage, len(o.Schema))
		copy(clone.Schema, o.Schema)
	}
	return clone
}

// ValidateOutputSchema validates output against the bounded portable JSON
// Schema subset shared by provider codecs.
func ValidateOutputSchema(output OutputSchema) error {
	if err := validateOutputName(output.Name); err != nil {
		return err
	}
	if !utf8.ValidString(output.Description) {
		return schemaError(SchemaFieldDescription, SchemaReasonInvalidUTF8)
	}
	if len(output.Description) > maxOutputDescriptionBytes {
		return schemaError(SchemaFieldDescription, SchemaReasonTooLong)
	}
	if len(output.Schema) > maxOutputSchemaBytes {
		return schemaError(SchemaFieldSchema, SchemaReasonTooLarge)
	}
	if !utf8.Valid(output.Schema) || !json.Valid(output.Schema) {
		return schemaError(SchemaFieldSchema, SchemaReasonMalformed)
	}
	if firstJSONByte(output.Schema) != '{' {
		return schemaError(SchemaFieldSchema, SchemaReasonRootNotObject)
	}

	validator := schemaValidator{}
	return validator.validateNode(output.Schema, 1, true)
}

func validateOutputName(name string) error {
	if name == "" {
		return schemaError(SchemaFieldName, SchemaReasonEmpty)
	}
	if len(name) > maxOutputNameBytes {
		return schemaError(SchemaFieldName, SchemaReasonTooLong)
	}
	if name == StructuredOutputToolName {
		return schemaError(SchemaFieldName, SchemaReasonReserved)
	}
	for i := range len(name) {
		ch := name[i]
		if i == 0 {
			if !isASCIIAlpha(ch) && ch != '_' {
				return schemaError(SchemaFieldName, SchemaReasonInvalid)
			}
			continue
		}
		if !isASCIIAlpha(ch) && !isASCIIDigit(ch) && ch != '_' && ch != '-' {
			return schemaError(SchemaFieldName, SchemaReasonInvalid)
		}
	}
	return nil
}

func isASCIIAlpha(ch byte) bool {
	return ch >= 'A' && ch <= 'Z' || ch >= 'a' && ch <= 'z'
}

func isASCIIDigit(ch byte) bool {
	return ch >= '0' && ch <= '9'
}

type rawSchemaNode struct {
	typeValue            json.RawMessage
	descriptionValue     json.RawMessage
	propertiesValue      json.RawMessage
	itemsValue           json.RawMessage
	enumValue            json.RawMessage
	requiredValue        json.RawMessage
	additionalProperties json.RawMessage
}

type schemaProperty struct {
	name   string
	schema json.RawMessage
}

type schemaValidator struct {
	propertyCount int
}

func (v *schemaValidator) validateNode(raw json.RawMessage, depth int, root bool) error {
	if depth > maxOutputSchemaDepth {
		return schemaError(SchemaFieldSchema, SchemaReasonTooDeep)
	}
	if firstJSONByte(raw) != '{' {
		return schemaError(SchemaFieldSchema, SchemaReasonInvalid)
	}

	node, err := decodeRawSchemaNode(raw)
	if err != nil {
		return err
	}
	if node.typeValue == nil {
		return schemaError(SchemaFieldType, SchemaReasonMissing)
	}
	var schemaType string
	if err := json.Unmarshal(node.typeValue, &schemaType); err != nil || schemaType == "" {
		return schemaError(SchemaFieldType, SchemaReasonInvalid)
	}
	if root && schemaType != "object" {
		return schemaError(SchemaFieldSchema, SchemaReasonRootNotObject)
	}
	if node.descriptionValue != nil {
		var description string
		if firstJSONByte(node.descriptionValue) != '"' {
			return schemaError(SchemaFieldDescription, SchemaReasonInvalid)
		}
		if err := json.Unmarshal(node.descriptionValue, &description); err != nil || !utf8.ValidString(description) {
			return schemaError(SchemaFieldDescription, SchemaReasonInvalid)
		}
		if len(description) > maxOutputDescriptionBytes {
			return schemaError(SchemaFieldDescription, SchemaReasonTooLong)
		}
	}

	switch schemaType {
	case "object":
		return v.validateObject(node, depth)
	case "array":
		return v.validateArray(node, depth)
	case "string", "boolean", "number", "integer":
		return validateScalar(node, schemaType)
	default:
		return schemaError(SchemaFieldType, SchemaReasonUnsupported)
	}
}

func (v *schemaValidator) validateObject(node rawSchemaNode, depth int) error {
	if node.itemsValue != nil {
		return schemaError(SchemaFieldItems, SchemaReasonUnsupported)
	}
	if node.enumValue != nil {
		return schemaError(SchemaFieldEnum, SchemaReasonUnsupported)
	}
	if node.additionalProperties == nil {
		return schemaError(SchemaFieldAdditionalProperties, SchemaReasonMissing)
	}
	if !bytes.Equal(bytes.TrimSpace(node.additionalProperties), []byte("false")) {
		return schemaError(SchemaFieldAdditionalProperties, SchemaReasonMustBeFalse)
	}

	properties, err := decodeProperties(node.propertiesValue)
	if err != nil {
		return err
	}
	v.propertyCount += len(properties)
	if v.propertyCount > maxOutputSchemaProperties {
		return schemaError(SchemaFieldProperties, SchemaReasonTooManyProperties)
	}

	required, err := decodeRequired(node.requiredValue, len(properties) > 0)
	if err != nil {
		return err
	}
	known := make(map[string]struct{}, len(properties))
	for _, property := range properties {
		known[property.name] = struct{}{}
	}
	seenRequired := make(map[string]struct{}, len(required))
	for _, name := range required {
		if _, duplicate := seenRequired[name]; duplicate {
			return schemaError(SchemaFieldRequired, SchemaReasonDuplicate)
		}
		seenRequired[name] = struct{}{}
		if _, exists := known[name]; !exists {
			return schemaError(SchemaFieldRequired, SchemaReasonUnknownProperty)
		}
	}
	if len(seenRequired) != len(known) {
		return schemaError(SchemaFieldRequired, SchemaReasonMissing)
	}

	for _, property := range properties {
		if err := v.validateNode(property.schema, depth+1, false); err != nil {
			return err
		}
	}
	return nil
}

func (v *schemaValidator) validateArray(node rawSchemaNode, depth int) error {
	if node.propertiesValue != nil {
		return schemaError(SchemaFieldProperties, SchemaReasonUnsupported)
	}
	if node.requiredValue != nil {
		return schemaError(SchemaFieldRequired, SchemaReasonUnsupported)
	}
	if node.additionalProperties != nil {
		return schemaError(SchemaFieldAdditionalProperties, SchemaReasonUnsupported)
	}
	if node.enumValue != nil {
		return schemaError(SchemaFieldEnum, SchemaReasonUnsupported)
	}
	if node.itemsValue == nil {
		return schemaError(SchemaFieldItems, SchemaReasonMissing)
	}
	if firstJSONByte(node.itemsValue) != '{' {
		return schemaError(SchemaFieldItems, SchemaReasonInvalid)
	}
	return v.validateNode(node.itemsValue, depth+1, false)
}

func validateScalar(node rawSchemaNode, schemaType string) error {
	if node.propertiesValue != nil {
		return schemaError(SchemaFieldProperties, SchemaReasonUnsupported)
	}
	if node.itemsValue != nil {
		return schemaError(SchemaFieldItems, SchemaReasonUnsupported)
	}
	if node.requiredValue != nil {
		return schemaError(SchemaFieldRequired, SchemaReasonUnsupported)
	}
	if node.additionalProperties != nil {
		return schemaError(SchemaFieldAdditionalProperties, SchemaReasonUnsupported)
	}
	if node.enumValue == nil {
		return nil
	}

	var values []json.RawMessage
	if err := json.Unmarshal(node.enumValue, &values); err != nil || len(values) == 0 {
		return schemaError(SchemaFieldEnum, SchemaReasonInvalid)
	}
	for _, value := range values {
		if !enumValueMatches(value, schemaType) {
			return schemaError(SchemaFieldEnum, SchemaReasonTypeMismatch)
		}
	}
	return nil
}

func enumValueMatches(raw json.RawMessage, schemaType string) bool {
	switch schemaType {
	case "string":
		var value string
		return firstJSONByte(raw) == '"' && json.Unmarshal(raw, &value) == nil
	case "boolean":
		var value bool
		return json.Unmarshal(raw, &value) == nil && (bytes.Equal(raw, []byte("true")) || bytes.Equal(raw, []byte("false")))
	case "number", "integer":
		var value json.Number
		if json.Unmarshal(raw, &value) != nil || value == "" {
			return false
		}
		return schemaType == "number" || jsonNumberIsIntegral(value.String())
	default:
		return false
	}
}

func jsonNumberIsIntegral(number string) bool {
	mantissa := number
	exponent := 0
	if index := strings.IndexAny(mantissa, "eE"); index >= 0 {
		parsed, err := strconv.Atoi(mantissa[index+1:])
		if err != nil {
			return false
		}
		exponent = parsed
		mantissa = mantissa[:index]
	}
	mantissa = strings.TrimPrefix(mantissa, "-")
	fractionDigits := 0
	if index := strings.IndexByte(mantissa, '.'); index >= 0 {
		fractionDigits = len(mantissa) - index - 1
		mantissa = mantissa[:index] + mantissa[index+1:]
	}
	remainingFraction := fractionDigits - exponent
	if remainingFraction <= 0 {
		return true
	}
	if remainingFraction > len(mantissa) {
		return strings.Trim(mantissa, "0") == ""
	}
	return strings.Trim(mantissa[len(mantissa)-remainingFraction:], "0") == ""
}

func decodeRawSchemaNode(raw json.RawMessage) (rawSchemaNode, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return rawSchemaNode{}, schemaError(SchemaFieldSchema, SchemaReasonInvalid)
	}

	var node rawSchemaNode
	for keyword, value := range fields {
		switch keyword {
		case "type":
			node.typeValue = value
		case "description":
			node.descriptionValue = value
		case "properties":
			node.propertiesValue = value
		case "items":
			node.itemsValue = value
		case "enum":
			node.enumValue = value
		case "required":
			node.requiredValue = value
		case "additionalProperties":
			node.additionalProperties = value
		default:
			return rawSchemaNode{}, schemaError(SchemaFieldKeyword, SchemaReasonUnknownKeyword)
		}
	}
	return node, nil
}

func decodeProperties(raw json.RawMessage) ([]schemaProperty, error) {
	if raw == nil {
		return nil, nil
	}
	if firstJSONByte(raw) != '{' {
		return nil, schemaError(SchemaFieldProperties, SchemaReasonInvalid)
	}
	var values map[string]json.RawMessage
	if err := json.Unmarshal(raw, &values); err != nil || values == nil {
		return nil, schemaError(SchemaFieldProperties, SchemaReasonInvalid)
	}
	properties := make([]schemaProperty, 0, len(values))
	for name, schema := range values {
		properties = append(properties, schemaProperty{name: name, schema: schema})
	}
	return properties, nil
}

func decodeRequired(raw json.RawMessage, required bool) ([]string, error) {
	if raw == nil {
		if required {
			return nil, schemaError(SchemaFieldRequired, SchemaReasonMissing)
		}
		return nil, nil
	}
	var values []string
	if err := json.Unmarshal(raw, &values); err != nil || values == nil {
		return nil, schemaError(SchemaFieldRequired, SchemaReasonInvalid)
	}
	return values, nil
}

func firstJSONByte(raw json.RawMessage) byte {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return 0
	}
	return trimmed[0]
}

func schemaError(field SchemaValidationField, reason SchemaValidationReason) error {
	return &SchemaValidationError{Field: field, Reason: reason}
}
