package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// Smithy 2.0 AST -> JSON Schema 2020-12, for the rest-json protocol.
//
// The source is AWS's own model distribution (aws/api-models-aws), not an SDK's
// vendored copy. That matters beyond provenance: the official model keeps its
// @pattern traits anchored, so a constraint such as ToolUseId's
// "^[a-zA-Z0-9_.:-]+$" rejects an id with a stray space anywhere in it, where
// the unanchored copy would have accepted it.
//
// Shape mapping:
//
//	structure -> object, with @required members in "required"
//	union     -> object plus a oneOf of single-member requirements, which is
//	             exactly "one and only one member is set"
//	enum      -> string with the @enumValue list
//	list      -> array; map -> object with additionalProperties
//	blob      -> base64 string (rest-json encodes blobs that way)
//	timestamp -> epoch seconds, the rest-json default
//	document  -> unconstrained, which is what a Smithy document is
//
// Constraint traits (@length, @range, @pattern) are carried across on both
// shapes and members. HTTP-bound members (@httpLabel, @httpQuery, @httpHeader,
// @httpResponseCode) are omitted, because they are not part of the JSON body
// this gate validates.
type smithyConverter struct {
	shapes map[string]any
	// namespace is the service namespace, used to turn shape IDs into short
	// definition names.
	namespace string
	rep       *docReport
	err       error
}

func newSmithyConverter(doc any, rep *docReport) (converter, error) {
	root, ok := doc.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("document root is %T, want an object", doc)
	}
	if version, _ := root["smithy"].(string); !strings.HasPrefix(version, "2.") {
		return nil, fmt.Errorf("smithy version %v is not 2.x", root["smithy"])
	}
	shapes, ok := root["shapes"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("smithy document has no shapes object")
	}

	namespace := ""
	for id, shape := range shapes {
		node, ok := shape.(map[string]any)
		if !ok {
			continue
		}
		if kind, _ := node["type"].(string); kind == "service" {
			namespace, _, _ = strings.Cut(id, "#")
			break
		}
	}
	if namespace == "" {
		return nil, fmt.Errorf("smithy document declares no service shape")
	}

	rep.note("blob members are base64 strings under rest-json; the encoding is recorded " +
		"as contentEncoding, which draft 2020-12 treats as an annotation, so a malformed " +
		"base64 body is not rejected.")
	rep.note("timestamps use the rest-json default of epoch seconds and are checked only " +
		"as numbers.")
	return &smithyConverter{shapes: shapes, namespace: namespace, rep: rep}, nil
}

// shapeID reconstructs the full shape ID from a short definition name.
func (c *smithyConverter) shapeID(name string) string { return c.namespace + "#" + name }

func (c *smithyConverter) has(name string) bool {
	_, ok := c.shapes[c.shapeID(name)]
	return ok
}

func (c *smithyConverter) convert(name string, b *builder) (any, error) {
	shape, _ := c.shapes[c.shapeID(name)].(map[string]any)
	if shape == nil {
		return nil, fmt.Errorf("shape %q is not an object", name)
	}
	converted := c.shape(name, shape, b)
	if c.err != nil {
		return nil, c.err
	}
	return converted, nil
}

// unionMembers reports the members of a Smithy union. The discriminator is the
// member name itself, since the rest-json encoding of a union is an object with
// exactly one member property set.
func (c *smithyConverter) unionMembers(name string) ([]unionMember, bool) {
	shape, ok := c.shapes[c.shapeID(name)].(map[string]any)
	if !ok {
		return nil, false
	}
	if kind, _ := shape["type"].(string); kind != "union" {
		return nil, false
	}
	members, ok := shape["members"].(map[string]any)
	if !ok {
		return nil, false
	}
	out := make([]unionMember, 0, len(members))
	for _, memberName := range sortedKeys(members) {
		member, ok := members[memberName].(map[string]any)
		if !ok {
			continue
		}
		target, _ := member["target"].(string)
		short, isLocal := strings.CutPrefix(target, c.namespace+"#")
		if !isLocal {
			// A prelude-targeted union member has no named shape to focus on.
			continue
		}
		out = append(out, unionMember{def: short, value: memberName})
	}
	return out, len(out) > 0
}

func (c *smithyConverter) shape(name string, shape map[string]any, b *builder) any {
	kind, _ := shape["type"].(string)
	traits := traitsOf(shape)
	out := map[string]any{}

	switch kind {
	case "structure":
		c.structure(name, shape, b, out)
	case "union":
		c.union(name, shape, b, out)
	case "enum":
		out["type"] = "string"
		out["enum"] = c.enumValues(shape)
	case "string":
		out["type"] = "string"
		c.applyString(out, traits, name)
	case "integer", "long":
		out["type"] = "integer"
		c.applyRange(out, traits, name)
	case "float", "double":
		out["type"] = "number"
		c.applyRange(out, traits, name)
	case "boolean":
		out["type"] = "boolean"
	case "blob":
		out["type"] = "string"
		out["contentEncoding"] = "base64"
		c.applyBlobLength(out, traits, name)
	case "timestamp":
		out["type"] = "number"
		if format, ok := traits["smithy.api#timestampFormat"].(string); ok && format != "epoch-seconds" {
			c.rep.dropConstraint(fmt.Sprintf("%s: @timestampFormat %q is not modelled", name, format))
		}
	case "document":
		// A Smithy document is arbitrary JSON by definition.
		return true
	case "list":
		out["type"] = "array"
		if member, ok := shape["member"].(map[string]any); ok {
			out["items"] = c.memberSchema(name+".member", member, b)
		}
		c.applyLength(out, traits, name, "minItems", "maxItems")
		if _, sparse := traits["smithy.api#sparse"]; sparse {
			out["items"] = map[string]any{"anyOf": []any{out["items"], map[string]any{"type": "null"}}}
		}
	case "map":
		out["type"] = "object"
		if value, ok := shape["value"].(map[string]any); ok {
			out["additionalProperties"] = c.memberSchema(name+".value", value, b)
		}
		if key, ok := shape["key"].(map[string]any); ok {
			if names := c.keySchema(name+".key", key); names != nil {
				out["propertyNames"] = names
			}
		}
		c.applyLength(out, traits, name, "minProperties", "maxProperties")
	default:
		if c.err == nil {
			c.err = fmt.Errorf("shape %q has unsupported type %q", name, kind)
		}
	}
	return out
}

// structure renders a structure's JSON body. Members bound to the HTTP envelope
// are not part of the body and are dropped; a member marked @httpPayload makes
// the body that member alone.
func (c *smithyConverter) structure(name string, shape map[string]any, b *builder, out map[string]any) {
	members, _ := shape["members"].(map[string]any)
	properties := map[string]any{}
	var required []string

	for _, memberName := range sortedKeys(members) {
		member, ok := members[memberName].(map[string]any)
		if !ok {
			continue
		}
		traits := traitsOf(member)
		if bound := httpBinding(traits); bound != "" {
			c.rep.note(fmt.Sprintf("%s.%s is carried in the HTTP %s, not the JSON body, and is not modelled",
				name, memberName, bound))
			continue
		}
		if _, payload := traits["smithy.api#httpPayload"]; payload {
			c.rep.note(fmt.Sprintf("%s has an @httpPayload member; the body is that member alone", name))
			for key, value := range c.memberSchema(name+"."+memberName, member, b) {
				out[key] = value
			}
			return
		}
		properties[memberName] = c.memberSchema(name+"."+memberName, member, b)
		if _, req := traits["smithy.api#required"]; req {
			required = append(required, memberName)
		}
	}

	out["type"] = "object"
	if len(properties) > 0 {
		out["properties"] = properties
	}
	if len(required) > 0 {
		sort.Strings(required)
		out["required"] = toAnySlice(required)
	}
}

// union renders a Smithy union: an object whose members are all optional
// individually, constrained by a oneOf that admits exactly one of them. Two
// members present make two branches match, which oneOf rejects; none present
// makes no branch match, which oneOf also rejects.
func (c *smithyConverter) union(name string, shape map[string]any, b *builder, out map[string]any) {
	members, _ := shape["members"].(map[string]any)
	properties := map[string]any{}
	var branches []any

	for _, memberName := range sortedKeys(members) {
		member, ok := members[memberName].(map[string]any)
		if !ok {
			continue
		}
		properties[memberName] = c.memberSchema(name+"."+memberName, member, b)
		branches = append(branches, map[string]any{"required": []any{memberName}})
	}

	out["type"] = "object"
	out["properties"] = properties
	out["oneOf"] = branches
}

// memberSchema renders a member reference plus any constraint traits carried on
// the member itself. Sibling keywords next to $ref are applied in 2020-12, so
// the constraint intersects the target shape rather than replacing it.
func (c *smithyConverter) memberSchema(path string, member map[string]any, b *builder) map[string]any {
	target, _ := member["target"].(string)
	out := c.targetSchema(path, target, b)
	traits := traitsOf(member)

	switch c.targetKind(target) {
	case "string", "enum":
		c.applyString(out, traits, path)
	case "blob":
		c.applyBlobLength(out, traits, path)
	case "list":
		c.applyLength(out, traits, path, "minItems", "maxItems")
	case "map":
		c.applyLength(out, traits, path, "minProperties", "maxProperties")
	case "integer", "long", "float", "double":
		c.applyRange(out, traits, path)
	}
	return out
}

// targetSchema resolves a member target to either a local reference or an
// inlined prelude shape.
func (c *smithyConverter) targetSchema(path, target string, b *builder) map[string]any {
	if short, ok := strings.CutPrefix(target, c.namespace+"#"); ok {
		return b.ref(short)
	}
	switch target {
	case "smithy.api#String":
		return map[string]any{"type": "string"}
	case "smithy.api#Boolean":
		return map[string]any{"type": "boolean"}
	case "smithy.api#Integer", "smithy.api#Long":
		return map[string]any{"type": "integer"}
	case "smithy.api#Float", "smithy.api#Double":
		return map[string]any{"type": "number"}
	case "smithy.api#Blob":
		return map[string]any{"type": "string", "contentEncoding": "base64"}
	case "smithy.api#Timestamp":
		return map[string]any{"type": "number"}
	case "smithy.api#Document":
		// Unconstrained by definition; an empty schema accepts any JSON value.
		return map[string]any{}
	default:
		if c.err == nil {
			c.err = fmt.Errorf("%s: unsupported target %q", path, target)
		}
		return map[string]any{}
	}
}

// targetKind reports the shape type a target resolves to, for deciding which
// constraint keywords a member trait maps onto.
func (c *smithyConverter) targetKind(target string) string {
	if shape, ok := c.shapes[target].(map[string]any); ok {
		kind, _ := shape["type"].(string)
		return kind
	}
	switch target {
	case "smithy.api#String":
		return "string"
	case "smithy.api#Blob":
		return "blob"
	case "smithy.api#Integer", "smithy.api#Long":
		return "integer"
	case "smithy.api#Float", "smithy.api#Double":
		return "double"
	default:
		return ""
	}
}

// keySchema constrains a map's property names, but only when the key shape
// actually says something; an unconstrained propertyNames schema is noise.
func (c *smithyConverter) keySchema(path string, key map[string]any) map[string]any {
	target, _ := key["target"].(string)
	shape, ok := c.shapes[target].(map[string]any)
	if !ok {
		return nil
	}
	out := map[string]any{}
	c.applyString(out, traitsOf(shape), path)
	if len(out) == 0 {
		return nil
	}
	return out
}

func (c *smithyConverter) enumValues(shape map[string]any) []any {
	members, _ := shape["members"].(map[string]any)
	values := make([]string, 0, len(members))
	for _, memberName := range sortedKeys(members) {
		member, ok := members[memberName].(map[string]any)
		if !ok {
			continue
		}
		value, ok := traitsOf(member)["smithy.api#enumValue"].(string)
		if !ok {
			value = memberName
		}
		values = append(values, value)
	}
	sort.Strings(values)
	return toAnySlice(values)
}

// applyString carries @length and @pattern onto a string schema. A pattern the
// Go regexp engine cannot compile is dropped and reported rather than emitted,
// because an uncompilable pattern would fail the whole document at load time.
func (c *smithyConverter) applyString(out map[string]any, traits map[string]any, path string) {
	c.applyLength(out, traits, path, "minLength", "maxLength")
	pattern, ok := traits["smithy.api#pattern"].(string)
	if !ok {
		return
	}
	if _, err := regexp.Compile(pattern); err != nil {
		c.rep.dropConstraint(fmt.Sprintf("%s: @pattern %q is not a valid RE2 expression (%v)", path, pattern, err))
		return
	}
	out["pattern"] = pattern
}

// applyBlobLength maps a blob's @length onto the base64 text. The trait bounds
// the decoded byte count, not the encoded string, so only a lower bound of at
// least one transfers soundly: it rules out the empty string. An upper bound
// would have to be inflated by the base64 expansion and is reported instead of
// guessed.
func (c *smithyConverter) applyBlobLength(out map[string]any, traits map[string]any, path string) {
	length, ok := traits["smithy.api#length"].(map[string]any)
	if !ok {
		return
	}
	if min, ok := numberOf(length["min"]); ok {
		if value, err := min.Int64(); err == nil && value >= 1 {
			out["minLength"] = json.Number("1")
		}
	}
	if _, ok := length["max"]; ok {
		c.rep.dropConstraint(fmt.Sprintf(
			"%s: @length max bounds the decoded blob, not its base64 text, and is not modelled", path))
	}
}

func (c *smithyConverter) applyLength(out map[string]any, traits map[string]any, path, minKey, maxKey string) {
	length, ok := traits["smithy.api#length"].(map[string]any)
	if !ok {
		return
	}
	if min, ok := numberOf(length["min"]); ok {
		out[minKey] = min
	}
	if max, ok := numberOf(length["max"]); ok {
		out[maxKey] = max
	}
	_ = path
}

func (c *smithyConverter) applyRange(out map[string]any, traits map[string]any, path string) {
	bounds, ok := traits["smithy.api#range"].(map[string]any)
	if !ok {
		return
	}
	if min, ok := numberOf(bounds["min"]); ok {
		out["minimum"] = min
	}
	if max, ok := numberOf(bounds["max"]); ok {
		out["maximum"] = max
	}
	_ = path
}

func traitsOf(node map[string]any) map[string]any {
	traits, _ := node["traits"].(map[string]any)
	if traits == nil {
		return map[string]any{}
	}
	return traits
}

// httpBinding names the HTTP envelope location a member is bound to, or "" when
// the member belongs in the JSON body.
func httpBinding(traits map[string]any) string {
	for trait, location := range map[string]string{
		"smithy.api#httpLabel":         "path",
		"smithy.api#httpQuery":         "query string",
		"smithy.api#httpQueryParams":   "query string",
		"smithy.api#httpHeader":        "header",
		"smithy.api#httpPrefixHeaders": "headers",
		"smithy.api#httpResponseCode":  "status line",
	} {
		if _, ok := traits[trait]; ok {
			return location
		}
	}
	return ""
}

func numberOf(value any) (json.Number, bool) {
	number, ok := value.(json.Number)
	return number, ok
}

func toAnySlice(values []string) []any {
	out := make([]any, len(values))
	for i, v := range values {
		out[i] = v
	}
	return out
}
