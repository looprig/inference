package main

import (
	"fmt"
	"strings"
)

// OpenAPI 3.1 -> JSON Schema 2020-12.
//
// This is very nearly the identity conversion: 3.1 adopted 2020-12 as its
// schema dialect, so the work is (a) rebasing "#/components/schemas/X" onto
// "#/$defs/X", (b) discarding annotation and vendor keywords that only add
// bulk, and (c) repairing the two OpenAPI 3.0 leftovers that both published
// documents still carry — "nullable" and the boolean form of the exclusive
// bounds. Both repairs are recorded in the report; the second is not optional,
// because a boolean exclusiveMinimum makes the document fail to compile.

const componentsPrefix = "#/components/schemas/"

type openAPIConverter struct {
	schemas map[string]any
	rep     *docReport
	// current is the definition being converted. It is the resolution target
	// for the 2019-09 $recursiveRef idiom that both documents still use.
	current string
	// err records the first structural surprise in the source document. The
	// walk returns values rather than errors, so it is checked after each
	// definition instead of unwinding mid-tree.
	err error
}

func newOpenAPIConverter(doc any, rep *docReport) (converter, error) {
	root, ok := doc.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("document root is %T, want an object", doc)
	}
	components, ok := root["components"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("document has no components object")
	}
	schemas, ok := components["schemas"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("document has no components.schemas object")
	}
	return &openAPIConverter{schemas: schemas, rep: rep}, nil
}

func (c *openAPIConverter) has(name string) bool {
	_, ok := c.schemas[name]
	return ok
}

func (c *openAPIConverter) convert(name string, b *builder) (any, error) {
	c.current = name
	converted := c.node(c.schemas[name], b, defPointer(name))
	if c.err != nil {
		return nil, c.err
	}
	return converted, nil
}

// unionMembers reads a union root. An explicit OpenAPI discriminator mapping is
// used when present; otherwise the discriminator value is recovered from each
// member's own pinned property, which is how OpenAI describes its response
// stream events.
func (c *openAPIConverter) unionMembers(name string) ([]unionMember, bool) {
	root, ok := c.schemas[name].(map[string]any)
	if !ok {
		return nil, false
	}
	branches, ok := root["oneOf"].([]any)
	if !ok {
		branches, ok = root["anyOf"].([]any)
	}
	if !ok {
		return nil, false
	}

	property := ""
	mapping := map[string]string{}
	if disc, ok := root["discriminator"].(map[string]any); ok {
		property, _ = disc["propertyName"].(string)
		if raw, ok := disc["mapping"].(map[string]any); ok {
			for value, ref := range raw {
				if text, ok := ref.(string); ok {
					mapping[strings.TrimPrefix(text, componentsPrefix)] = value
				}
			}
		}
	}

	var members []unionMember
	for _, branch := range branches {
		node, ok := branch.(map[string]any)
		if !ok {
			continue
		}
		ref, ok := node["$ref"].(string)
		if !ok || !strings.HasPrefix(ref, componentsPrefix) {
			continue
		}
		def := strings.TrimPrefix(ref, componentsPrefix)
		value := mapping[def]
		if value == "" {
			value = c.pinnedValue(def, property)
		}
		members = append(members, unionMember{def: def, value: value})
	}
	return members, len(members) > 0
}

// pinnedValue recovers the single constant a member fixes its discriminator
// property to. Both published specs express this as an enum of one or a const.
func (c *openAPIConverter) pinnedValue(def, property string) string {
	node, ok := c.schemas[def].(map[string]any)
	if !ok {
		return ""
	}
	props, ok := node["properties"].(map[string]any)
	if !ok {
		return ""
	}
	candidates := []string{property}
	if property == "" {
		candidates = []string{"type", "object"}
	}
	for _, key := range candidates {
		field, ok := props[key].(map[string]any)
		if !ok {
			continue
		}
		if text, ok := field["const"].(string); ok {
			return text
		}
		if values, ok := field["enum"].([]any); ok && len(values) == 1 {
			if text, ok := values[0].(string); ok {
				return text
			}
		}
	}
	return ""
}

// schemaKeywords classifies the 2020-12 applicator keywords so the walk knows
// which values are subschemas. Recursing blindly would misread a property
// literally named "$ref" or "items" as an applicator.
var (
	subschemaKeywords = map[string]bool{
		"additionalProperties": true, "contains": true, "contentSchema": true,
		"else": true, "if": true, "items": true, "not": true,
		"propertyNames": true, "then": true, "unevaluatedItems": true,
		"unevaluatedProperties": true,
	}
	subschemaListKeywords = map[string]bool{
		"allOf": true, "anyOf": true, "oneOf": true, "prefixItems": true,
	}
	subschemaMapKeywords = map[string]bool{
		"$defs": true, "definitions": true, "dependentSchemas": true,
		"patternProperties": true, "properties": true,
	}
	// droppedKeywords are annotations, vendor extensions and OpenAPI-only
	// document structure. None of them assert anything under 2020-12, and
	// dropping them removes the great majority of both documents' bulk.
	droppedKeywords = map[string]bool{
		"default": true, "deprecated": true, "description": true,
		"discriminator": true, "example": true, "examples": true,
		"externalDocs": true, "readOnly": true, "summary": true,
		"writeOnly": true, "xml": true,
	}
)

// node converts one schema node. ptr is the pointer of the node within the
// emitted document, used only for reporting.
func (c *openAPIConverter) node(value any, b *builder, ptr string) any {
	switch typed := value.(type) {
	case bool:
		return typed
	case map[string]any:
		return c.object(typed, b, ptr)
	default:
		return value
	}
}

func (c *openAPIConverter) object(in map[string]any, b *builder, ptr string) any {
	out := make(map[string]any, len(in))

	for _, key := range sortedKeys(in) {
		value := in[key]
		switch {
		case strings.HasPrefix(key, "x-"):
			c.rep.dropKeyword(key)
		case droppedKeywords[key]:
			c.rep.dropKeyword(key)
		case key == "$ref":
			ref, ok := value.(string)
			if !ok || !strings.HasPrefix(ref, componentsPrefix) {
				// Every reference in both documents is a component reference.
				// Anything else would silently become unresolvable, so fail
				// generation rather than emit a dangling pointer.
				if c.err == nil {
					c.err = fmt.Errorf("%s: unsupported $ref %v", ptr, value)
				}
				continue
			}
			out["$ref"] = b.ref(strings.TrimPrefix(ref, componentsPrefix))["$ref"]
		case key == "$recursiveAnchor":
			// Paired with $recursiveRef below; the anchor itself carries no
			// assertion once the reference is rewritten.
			c.rep.dropKeyword(key)
		case key == "$recursiveRef":
			// The 2019-09 recursion idiom, still used by OpenAI for the
			// self-nesting CompoundFilter. Within a single definition it means
			// exactly "recurse into the definition that carries the anchor",
			// which is a plain local reference in 2020-12.
			if target, ok := value.(string); !ok || target != "#" {
				if c.err == nil {
					c.err = fmt.Errorf("%s: unsupported $recursiveRef %v", ptr, value)
				}
				continue
			}
			c.rep.note(fmt.Sprintf("$recursiveRef in %q was rewritten to a local reference to that definition",
				c.current))
			out["$ref"] = b.ref(c.current)["$ref"]
		case subschemaKeywords[key]:
			out[key] = c.node(value, b, ptr+"/"+key)
		case subschemaListKeywords[key]:
			list, ok := value.([]any)
			if !ok {
				out[key] = value
				continue
			}
			converted := make([]any, len(list))
			for i, item := range list {
				converted[i] = c.node(item, b, fmt.Sprintf("%s/%s/%d", ptr, key, i))
			}
			out[key] = converted
		case subschemaMapKeywords[key]:
			children, ok := value.(map[string]any)
			if !ok {
				out[key] = value
				continue
			}
			converted := make(map[string]any, len(children))
			for _, name := range sortedKeys(children) {
				converted[name] = c.node(children[name], b, ptr+"/"+key+"/"+name)
			}
			out[key] = converted
		case key == "format":
			if text, ok := value.(string); ok {
				c.rep.noteFormat(text)
			}
			out[key] = value
		default:
			out[key] = value
		}
	}

	c.repairExclusiveBounds(out, ptr)
	c.repairOverlappingOneOf(in, out, ptr)
	return c.repairNullable(in, out, ptr)
}

// repairOverlappingOneOf relaxes a oneOf whose branches are not mutually
// exclusive into an anyOf.
//
// oneOf asserts "exactly one branch matches". Both published documents use it
// as a loose "one of these shapes" in places where the shapes genuinely
// overlap: OpenAI's InputItem offers EasyInputMessage alongside Item, and Item
// in turn offers InputMessage, so an ordinary user message matches two branches
// at once. The live API accepts that payload, so keeping oneOf would make the
// gate reject a legal request — the one failure mode a conformance gate must
// never have.
//
// The relaxation is allowlisted rather than inferred from an inability to
// prove exclusivity. Failure to prove that branches are exclusive is not proof
// that they overlap; widening every such union silently discards constraints
// from scalar and nested unions that are perfectly sound.
func (c *openAPIConverter) repairOverlappingOneOf(in, out map[string]any, ptr string) {
	branches, ok := in["oneOf"].([]any)
	if !ok || len(branches) < 2 {
		return
	}
	if c.provablyExclusive(branches) {
		return
	}
	if !knownOverlappingOneOf[ptr] {
		return
	}
	converted, ok := out["oneOf"]
	if !ok {
		return
	}
	delete(out, "oneOf")
	out["anyOf"] = converted
	c.rep.OverlappingOneOf = append(c.rep.OverlappingOneOf, ptr)
}

// knownOverlappingOneOf contains only overlaps demonstrated against the
// published API shapes. InputItem includes EasyInputMessage directly and Item,
// which itself includes InputMessage; an ordinary input message therefore
// matches two branches. Additions require a fixture proving the overlap.
var knownOverlappingOneOf = map[string]bool{
	"#/$defs/InputItem": true,
}

// provablyExclusive reports whether some shared property pins every branch to a
// distinct constant, which is what makes a oneOf sound.
func (c *openAPIConverter) provablyExclusive(branches []any) bool {
	pinned := make([]map[string]string, 0, len(branches))
	for _, branch := range branches {
		node, ok := branch.(map[string]any)
		if !ok {
			return false
		}
		values := c.pinnedValues(node)
		if len(values) == 0 {
			return false
		}
		pinned = append(pinned, values)
	}

	for property := range pinned[0] {
		seen := make(map[string]bool, len(pinned))
		distinct := true
		for _, values := range pinned {
			value, ok := values[property]
			if !ok || seen[value] {
				distinct = false
				break
			}
			seen[value] = true
		}
		if distinct {
			return true
		}
	}
	return false
}

// pinnedValues collects every property a schema fixes to a single string,
// following one level of reference. A branch that pins nothing — most often
// because it is itself a nested union — yields an empty map, which is what
// makes its enclosing oneOf unprovable.
func (c *openAPIConverter) pinnedValues(node map[string]any) map[string]string {
	if ref, ok := node["$ref"].(string); ok && strings.HasPrefix(ref, componentsPrefix) {
		target, ok := c.schemas[strings.TrimPrefix(ref, componentsPrefix)].(map[string]any)
		if !ok {
			return nil
		}
		node = target
	}
	properties, ok := node["properties"].(map[string]any)
	if !ok {
		return nil
	}
	out := map[string]string{}
	for name, raw := range properties {
		field, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if text, ok := field["const"].(string); ok {
			out[name] = text
			continue
		}
		if values, ok := field["enum"].([]any); ok && len(values) == 1 {
			if text, ok := values[0].(string); ok {
				out[name] = text
			}
		}
	}
	return out
}

// repairExclusiveBounds rewrites the OpenAPI 3.0 boolean form of the exclusive
// bounds into the 2020-12 numeric form. This is a correctness fix, not a
// cosmetic one: a boolean exclusiveMinimum is not a valid 2020-12 schema and
// the document would not compile.
func (c *openAPIConverter) repairExclusiveBounds(out map[string]any, ptr string) {
	for _, pair := range [][2]string{{"exclusiveMinimum", "minimum"}, {"exclusiveMaximum", "maximum"}} {
		exclusive, inclusive := pair[0], pair[1]
		flag, isBool := out[exclusive].(bool)
		if !isBool {
			continue
		}
		delete(out, exclusive)
		c.rep.BooleanExclusiveBounds = append(c.rep.BooleanExclusiveBounds, ptr)
		if !flag {
			continue
		}
		bound, ok := out[inclusive]
		if !ok {
			c.rep.dropConstraint(fmt.Sprintf("%s: %s was true with no %s to make exclusive", ptr, exclusive, inclusive))
			continue
		}
		out[exclusive] = bound
		delete(out, inclusive)
	}
}

// repairNullable widens a schema that carries the OpenAPI 3.0 "nullable: true"
// leftover so that it admits null.
//
// Both published 3.1 documents still emit it — 111 occurrences in OpenAI's,
// 208 in Anthropic's — and a 2020-12 validator ignores the keyword entirely.
// Ignoring it is not the harmless under-constraint it looks like: for
// {"type":"string","nullable":true} a validator that drops "nullable" enforces
// type:string and REJECTS the null the provider is documented to send. So the
// widening is required to avoid falsely rejecting legal payloads, and every
// site is reported because the widened position no longer distinguishes "null
// is allowed here" from "anything is allowed here".
func (c *openAPIConverter) repairNullable(in, out map[string]any, ptr string) any {
	raw, present := in["nullable"]
	delete(out, "nullable")
	if !present {
		return out
	}
	c.rep.dropKeyword("nullable")
	if flag, _ := raw.(bool); !flag {
		return out
	}
	c.rep.NullableWidened = append(c.rep.NullableWidened, ptr)

	widened := false
	if declared, ok := out["type"]; ok {
		out["type"] = withNullType(declared)
		widened = true
	}
	if values, ok := out["enum"].([]any); ok {
		out["enum"] = withNullValue(values)
		widened = true
	}
	if widened {
		return out
	}
	// No type and no enum to widen: the node is a bare reference or a
	// composition, so admit null alongside whatever it already says.
	return map[string]any{"anyOf": []any{out, map[string]any{"type": "null"}}}
}

func withNullType(declared any) any {
	switch typed := declared.(type) {
	case string:
		if typed == "null" {
			return typed
		}
		return []any{typed, "null"}
	case []any:
		for _, item := range typed {
			if text, ok := item.(string); ok && text == "null" {
				return typed
			}
		}
		return append(append([]any{}, typed...), "null")
	default:
		return declared
	}
}

func withNullValue(values []any) []any {
	for _, item := range values {
		if item == nil {
			return values
		}
	}
	return append(append([]any{}, values...), nil)
}
