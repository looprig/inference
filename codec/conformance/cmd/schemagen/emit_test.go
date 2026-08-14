package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCountShapesDoesNotCountPropertiesMapAsSchema(t *testing.T) {
	t.Parallel()

	rep := &docReport{}
	countShapes(rep, map[string]any{
		"Envelope": map[string]any{
			"type": "object",
			"properties": map[string]any{
				// A field may itself be named "properties". The containing map is
				// a property-name map, not another schema node.
				"properties": map[string]any{"type": "string"},
			},
			"required": []any{"properties"},
		},
	})

	if rep.ObjectSchemas != 1 {
		t.Fatalf("ObjectSchemas = %d, want 1 (the properties-name map is not a schema)", rep.ObjectSchemas)
	}
}

func TestDocumentDescriptionNamesCurrentGeneratorPath(t *testing.T) {
	t.Parallel()

	got := documentDescription(target{format: "openai", kind: "request", direction: directionRequest, source: "openai"})
	if !strings.Contains(got, "codec/conformance/cmd/schemagen") {
		t.Fatalf("description does not name current generator path: %q", got)
	}
	if strings.Contains(got, "providers/internal/conformance") {
		t.Fatalf("description still names removed generator path: %q", got)
	}
}

func TestWriteTreePrunesStaleGeneratedSchemas(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	stale := filepath.Join(dir, "old", "removed.schema.json")
	keep := filepath.Join(dir, "operator-note.txt")
	if err := os.MkdirAll(filepath.Dir(stale), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(stale, []byte("stale"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(keep, []byte("keep"), 0o600); err != nil {
		t.Fatal(err)
	}

	if err := writeTree(dir, map[string][]byte{"new/current.schema.json": []byte("current")}); err != nil {
		t.Fatalf("writeTree() error = %v", err)
	}
	if _, err := os.Stat(stale); !os.IsNotExist(err) {
		t.Fatalf("stale generated schema still exists; Stat error = %v", err)
	}
	if _, err := os.Stat(keep); err != nil {
		t.Fatalf("writeTree removed unrelated file: %v", err)
	}
}

func TestWriteTreeDoesNotFollowOutputSymlinkOutsideRoot(t *testing.T) {
	t.Parallel()

	parent := t.TempDir()
	outDir := filepath.Join(parent, "schema")
	outside := filepath.Join(parent, "outside")
	if err := os.MkdirAll(outDir, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(outside, 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(outDir, "escape")); err != nil {
		t.Fatal(err)
	}

	err := writeTree(outDir, map[string][]byte{
		"escape/current.schema.json": []byte("must stay confined"),
	})
	if err == nil {
		t.Fatal("writeTree followed an output symlink outside its root")
	}
	if _, statErr := os.Stat(filepath.Join(outside, "current.schema.json")); !os.IsNotExist(statErr) {
		t.Fatalf("writeTree created a schema outside its output root; Stat error = %v", statErr)
	}
}

func TestUnprovenOneOfIsPreservedUnlessKnownToOverlap(t *testing.T) {
	t.Parallel()

	rep := &docReport{}
	conv := &openAPIConverter{rep: rep, schemas: map[string]any{}}
	in := map[string]any{"oneOf": []any{
		map[string]any{"type": "string"},
		map[string]any{"type": "number"},
	}}
	out := map[string]any{"oneOf": in["oneOf"]}
	conv.repairOverlappingOneOf(in, out, "#/$defs/Unrelated")

	if _, ok := out["oneOf"]; !ok {
		t.Fatal("an unproven union was relaxed without evidence that its branches overlap")
	}
	if len(rep.OverlappingOneOf) != 0 {
		t.Fatalf("reported relaxations = %v, want none", rep.OverlappingOneOf)
	}
}

func TestKnownOverlappingOneOfIsRelaxed(t *testing.T) {
	t.Parallel()

	rep := &docReport{}
	conv := &openAPIConverter{rep: rep, schemas: map[string]any{}}
	in := map[string]any{"oneOf": []any{
		map[string]any{"$ref": componentsPrefix + "EasyInputMessage"},
		map[string]any{"$ref": componentsPrefix + "Item"},
	}}
	out := map[string]any{"oneOf": in["oneOf"]}
	conv.repairOverlappingOneOf(in, out, "#/$defs/InputItem")

	if _, ok := out["anyOf"]; !ok {
		t.Fatal("known InputItem overlap kept oneOf, which rejects ordinary input messages")
	}
	if got := rep.OverlappingOneOf; len(got) != 1 || got[0] != "#/$defs/InputItem" {
		t.Fatalf("reported relaxations = %v, want InputItem", got)
	}
}
