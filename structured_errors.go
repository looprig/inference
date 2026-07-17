package inference

// SchemaValidationField identifies the output-schema component that failed
// validation. Values are stable classifications suitable for errors.As callers.
type SchemaValidationField string

const (
	SchemaFieldName                 SchemaValidationField = "Name"
	SchemaFieldDescription          SchemaValidationField = "Description"
	SchemaFieldSchema               SchemaValidationField = "Schema"
	SchemaFieldKeyword              SchemaValidationField = "Keyword"
	SchemaFieldType                 SchemaValidationField = "Type"
	SchemaFieldProperties           SchemaValidationField = "Properties"
	SchemaFieldItems                SchemaValidationField = "Items"
	SchemaFieldEnum                 SchemaValidationField = "Enum"
	SchemaFieldRequired             SchemaValidationField = "Required"
	SchemaFieldAdditionalProperties SchemaValidationField = "AdditionalProperties"
)

// SchemaValidationReason identifies why an output-schema component failed
// validation. It intentionally contains no caller-provided content.
type SchemaValidationReason string

const (
	SchemaReasonEmpty             SchemaValidationReason = "empty"
	SchemaReasonInvalid           SchemaValidationReason = "invalid"
	SchemaReasonReserved          SchemaValidationReason = "reserved"
	SchemaReasonTooLong           SchemaValidationReason = "too long"
	SchemaReasonInvalidUTF8       SchemaValidationReason = "invalid UTF-8"
	SchemaReasonMalformed         SchemaValidationReason = "malformed"
	SchemaReasonTooLarge          SchemaValidationReason = "too large"
	SchemaReasonRootNotObject     SchemaValidationReason = "root is not an object schema"
	SchemaReasonUnknownKeyword    SchemaValidationReason = "unknown keyword"
	SchemaReasonMissing           SchemaValidationReason = "missing"
	SchemaReasonUnsupported       SchemaValidationReason = "unsupported"
	SchemaReasonMustBeFalse       SchemaValidationReason = "must be false"
	SchemaReasonDuplicate         SchemaValidationReason = "duplicate"
	SchemaReasonUnknownProperty   SchemaValidationReason = "unknown property"
	SchemaReasonTypeMismatch      SchemaValidationReason = "type mismatch"
	SchemaReasonTooDeep           SchemaValidationReason = "too deep"
	SchemaReasonTooManyProperties SchemaValidationReason = "too many properties"
)

// SchemaValidationError reports a stable, bounded validation classification.
// It never retains schema bytes, property names, descriptions, or JSON decoder
// errors because those values may contain sensitive caller input.
type SchemaValidationError struct {
	Field  SchemaValidationField
	Reason SchemaValidationReason
}

func (e *SchemaValidationError) Error() string {
	return "inference: invalid output schema field " + string(e.Field) + ": " + string(e.Reason)
}

// StructuredOutputUnsupportedError reports that a model does not advertise
// native structured output. Model is diagnostic metadata only.
type StructuredOutputUnsupportedError struct {
	Model string
}

func (e *StructuredOutputUnsupportedError) Error() string {
	return "inference: structured output unsupported"
}

// StructuredOutputWithToolsUnsupportedError reports that a model does not
// advertise the distinct native structured-output-with-tools capability.
// Model is diagnostic metadata only.
type StructuredOutputWithToolsUnsupportedError struct {
	Model string
}

func (e *StructuredOutputWithToolsUnsupportedError) Error() string {
	return "inference: structured output with tools unsupported"
}

// StructuredOutputConflictError reports an invalid request feature
// combination. Feature is a bounded classification, never a schema or tool
// payload supplied by the caller.
type StructuredOutputConflictError struct {
	Feature string
}

func (e *StructuredOutputConflictError) Error() string {
	return "inference: structured output feature conflict: " + e.Feature
}
