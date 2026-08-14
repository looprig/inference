package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Google discovery -> JSON Schema 2020-12.
//
// The discovery format is a small, closed subset of draft-03 JSON Schema plus
// Google's own annotations, so the conversion is mechanical: references become
// local pointers, the pseudo-type "any" becomes no constraint at all, numeric
// bounds arrive as strings and are parsed back into numbers, and everything
// else is documentation.
//
// The important thing to know about the result is what the format cannot
// express. Discovery has no response-side required list: proto3 field presence
// is optional by construction and Google publishes requiredness only as prose
// in each field's description. The gemini document therefore constrains types,
// enums, nesting and array-ness, and cannot detect a missing field. That is
// recorded on the document rather than papered over by guessing.
type discoveryConverter struct {
	schemas map[string]any
	rep     *docReport
	err     error
}

func newDiscoveryConverter(doc any, rep *docReport) (converter, error) {
	root, ok := doc.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("document root is %T, want an object", doc)
	}
	schemas, ok := root["schemas"].(map[string]any)
	if !ok {
		return nil, fmt.Errorf("discovery document has no schemas object")
	}
	rep.note("Google's discovery format declares almost no required properties, on either " +
		"the request or the response side: proto3 field presence is optional by construction " +
		"and Google publishes requiredness only as prose in each field's description. This " +
		"document therefore constrains types, enums, array-ness and nesting, and cannot " +
		"detect a missing field.")
	rep.note("int64 and uint64 fields are typed as strings, which is what proto3 JSON " +
		"emits; a fixture that writes them as numbers is correctly rejected.")
	return &discoveryConverter{schemas: schemas, rep: rep}, nil
}

func (c *discoveryConverter) has(name string) bool {
	_, ok := c.schemas[name]
	return ok
}

func (c *discoveryConverter) convert(name string, b *builder) (any, error) {
	converted := c.node(c.schemas[name], b, defPointer(name))
	if c.err != nil {
		return nil, c.err
	}
	return converted, nil
}

// unionMembers reports that discovery has no union construct. Gemini's response
// message is a plain structure, so no target declares one.
func (c *discoveryConverter) unionMembers(string) ([]unionMember, bool) { return nil, false }

// discoveryDropped are the discovery-only annotations. None of them constrain
// an instance.
var discoveryDropped = map[string]bool{
	"annotations": true, "default": true, "deprecated": true,
	"description": true, "enumDeprecated": true, "enumDescriptions": true,
	"id": true, "readOnly": true, "variant": true,
}

func (c *discoveryConverter) node(value any, b *builder, ptr string) any {
	node, ok := value.(map[string]any)
	if !ok {
		return value
	}
	out := make(map[string]any, len(node))

	for _, key := range sortedKeys(node) {
		item := node[key]
		switch key {
		case "$ref":
			name, ok := item.(string)
			if !ok {
				if c.err == nil {
					c.err = fmt.Errorf("%s: $ref is %T, want a definition name", ptr, item)
				}
				continue
			}
			out["$ref"] = b.ref(name)["$ref"]
		case "type":
			text, _ := item.(string)
			// Discovery's "any" is the absence of a type, not a type.
			if text == "any" {
				c.rep.note("some fields are typed \"any\" in discovery and are left unconstrained")
				continue
			}
			out["type"] = item
		case "format":
			if text, ok := item.(string); ok {
				c.rep.noteFormat(text)
			}
			out["format"] = item
		case "items":
			out["items"] = c.node(item, b, ptr+"/items")
		case "additionalProperties":
			out["additionalProperties"] = c.node(item, b, ptr+"/additionalProperties")
		case "properties":
			children, ok := item.(map[string]any)
			if !ok {
				continue
			}
			converted := make(map[string]any, len(children))
			for _, name := range sortedKeys(children) {
				converted[name] = c.node(children[name], b, ptr+"/properties/"+name)
			}
			out["properties"] = converted
		case "minimum", "maximum":
			// Discovery serialises bounds as strings; 2020-12 requires numbers.
			number, ok := discoveryNumber(item)
			if !ok {
				c.rep.dropConstraint(fmt.Sprintf("%s: %s %v is not a number", ptr, key, item))
				continue
			}
			out[key] = number
		case "enum", "pattern", "required":
			out[key] = item
		default:
			if discoveryDropped[key] {
				c.rep.dropKeyword(key)
				continue
			}
			c.rep.dropKeyword(key)
		}
	}
	return out
}

// discoveryNumber parses a bound that discovery encodes as a string.
func discoveryNumber(value any) (json.Number, bool) {
	switch typed := value.(type) {
	case json.Number:
		return typed, true
	case string:
		trimmed := strings.TrimSpace(typed)
		if trimmed == "" {
			return "", false
		}
		number := json.Number(trimmed)
		if _, err := number.Float64(); err != nil {
			return "", false
		}
		return number, true
	default:
		return "", false
	}
}
