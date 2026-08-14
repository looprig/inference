package conformance

import (
	"encoding/json"
	"io/fs"
	"path"
	"regexp"
	"strings"
	"testing"
)

// Provenance is what separates "these schemas came from the vendors" from
// "these schemas came from somewhere". These tests hold the record to the
// standard the claim needs: every source named, hashed and dated; every
// derived document accounted for; and the one source that is not first-party
// hosted flagged as such with its chain of custody intact.

var (
	sha256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
	datePattern   = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)
)

// firstPartyHosted names the sources served by the vendor's own
// infrastructure. Anything absent from this set must explain itself.
var firstPartyHosted = map[string]bool{
	"openai":                true,
	"anthropic-stats":       true,
	"gemini":                true,
	"bedrock":               true,
	"jsonschema-test-suite": true,
}

func TestProvenanceRecordsEverySource(t *testing.T) {
	t.Parallel()

	prov, err := LoadProvenance()
	if err != nil {
		t.Fatalf("LoadProvenance() error = %v", err)
	}
	want := []string{"openai", "anthropic", "anthropic-stats", "gemini", "bedrock", "jsonschema-test-suite"}
	for _, key := range want {
		source, ok := prov.Sources[key]
		if !ok {
			t.Errorf("provenance is missing source %q", key)
			continue
		}
		if !strings.HasPrefix(source.URL, "https://") {
			t.Errorf("%s: url = %q, want an https URL", key, source.URL)
		}
		if source.Publisher == "" {
			t.Errorf("%s: publisher is empty", key)
		}
		if !datePattern.MatchString(source.Retrieved) {
			t.Errorf("%s: retrieved = %q, want YYYY-MM-DD", key, source.Retrieved)
		}
		if !sha256Pattern.MatchString(source.SHA256) {
			t.Errorf("%s: sha256 = %q, want 64 lowercase hex digits", key, source.SHA256)
		}
		if source.Bytes <= 0 {
			t.Errorf("%s: bytes = %d, want a positive size", key, source.Bytes)
		}
	}
	for key := range prov.Sources {
		if !contains(want, key) {
			t.Errorf("provenance records an unexpected source %q", key)
		}
	}
}

// TestNonFirstPartyHostedSourcesExplainThemselves is the check the Anthropic
// situation exists for: there is no Anthropic-hosted OpenAPI document, so the
// gate consumes one Anthropic points at from its own SDK repository. That is
// acceptable only while it is stated, and while both ends of the pointer are
// recorded so a substitution is visible.
func TestNonFirstPartyHostedSourcesExplainThemselves(t *testing.T) {
	t.Parallel()

	prov, err := LoadProvenance()
	if err != nil {
		t.Fatalf("LoadProvenance() error = %v", err)
	}

	for key, source := range prov.Sources {
		if firstPartyHosted[key] {
			continue
		}
		if key != "anthropic" {
			t.Errorf("%s: source is not first-party hosted and is not the documented exception", key)
			continue
		}
		if source.Hosting == "" {
			t.Errorf("%s: hosting note is empty; a non-first-party-hosted source must state why", key)
		}
		if source.PointerSource == "" || source.PointerHash == "" {
			t.Errorf("%s: pointer_source/pointer_hash missing; the chain of custody is unverifiable", key)
		}
	}

	// The pointer must still resolve to the document we actually converted.
	stats, ok := prov.Sources["anthropic-stats"]
	if !ok {
		t.Fatal("provenance is missing the anthropic-stats pointer file")
	}
	pointed := stats.Fields["openapi_spec_url"]
	if pointed == "" {
		t.Fatal("anthropic-stats records no openapi_spec_url")
	}
	if got := prov.Sources["anthropic"].URL; got != pointed {
		t.Fatalf("anthropic source url = %q, but .stats.yml points at %q", got, pointed)
	}
	if got := prov.Sources["anthropic"].PointerHash; got != stats.Fields["openapi_spec_hash"] {
		t.Fatalf("recorded pointer_hash %q does not match .stats.yml openapi_spec_hash %q",
			got, stats.Fields["openapi_spec_hash"])
	}
}

// TestUnstableSourcesCarryACanonicalHash covers the Google discovery endpoint,
// which reorders its JSON between responses: without a canonical hash a refresh
// would look like a change every time and real drift would be invisible in the
// noise.
func TestUnstableSourcesCarryACanonicalHash(t *testing.T) {
	t.Parallel()

	prov, err := LoadProvenance()
	if err != nil {
		t.Fatalf("LoadProvenance() error = %v", err)
	}
	gemini := prov.Sources["gemini"]
	if gemini == nil {
		t.Fatal("provenance is missing the gemini source")
	}
	if !sha256Pattern.MatchString(gemini.CanonicalSHA256) {
		t.Fatalf("gemini canonical_sha256 = %q, want 64 lowercase hex digits", gemini.CanonicalSHA256)
	}
	if gemini.CanonicalNote == "" {
		t.Fatal("gemini canonical_note is empty; the reason for the second hash must be recorded")
	}
}

// TestEveryDocumentIsIndexedAndReported makes the three generated artefacts
// agree. A schema nobody can reach, or a document with no honesty entry, is a
// generation bug that would otherwise sit unnoticed in the tree.
func TestEveryDocumentIsIndexedAndReported(t *testing.T) {
	t.Parallel()

	onDisk := map[string]bool{}
	err := fs.WalkDir(SchemaFS(), schemaRoot, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".schema.json") {
			return err
		}
		onDisk[strings.TrimPrefix(p, schemaRoot+"/")] = true
		return nil
	})
	if err != nil {
		t.Fatalf("walk schema tree: %v", err)
	}
	if len(onDisk) == 0 {
		t.Fatal("the embedded schema tree contains no documents")
	}

	indexed := map[string]bool{}
	for _, format := range Formats() {
		for _, kind := range Kinds(format) {
			entry, _ := Lookup(format, kind)
			indexed[entry.Document] = true
			if !onDisk[entry.Document] {
				t.Errorf("index entry %s/%s points at %q, which is not in the tree", format, kind, entry.Document)
			}
		}
	}
	for document := range onDisk {
		if !indexed[document] {
			t.Errorf("%s is in the tree but no index entry reaches it", document)
		}
	}

	report := loadUnenforced(t)
	for document := range onDisk {
		entry, ok := report[document]
		if !ok {
			t.Errorf("%s has no entry in schema/unenforced.json", document)
			continue
		}
		if entry.Summary == "" {
			t.Errorf("%s has an unenforced entry with no summary", document)
		}
		if entry.Defs == 0 {
			t.Errorf("%s reports zero definitions", document)
		}
	}
}

// TestReportNamesTheKnownWeaknesses asserts that the honesty ledger actually
// says the things the package documentation promises it says. A report that
// silently stopped recording, say, the nullable widenings would make the gate
// look stronger than it is.
func TestReportNamesTheKnownWeaknesses(t *testing.T) {
	t.Parallel()

	report := loadUnenforced(t)

	chunk, ok := report["openai/chat_completion_chunk.schema.json"]
	if !ok {
		t.Fatal("unenforced.json has no entry for the OpenAI chunk document")
	}
	if len(chunk.NullableWidened) == 0 {
		t.Error("the OpenAI chunk document records no nullable widenings, but the source spec still uses nullable")
	}

	gemini, ok := report["gemini/generate_content_response.schema.json"]
	if !ok {
		t.Fatal("unenforced.json has no entry for the Gemini document")
	}
	if gemini.ObjectsWithoutRequired != gemini.ObjectSchemas {
		t.Errorf("gemini reports %d/%d object shapes without required properties; discovery declares none, so these must match",
			gemini.ObjectsWithoutRequired, gemini.ObjectSchemas)
	}
	if !mentions(gemini.Notes, "no required properties") {
		t.Errorf("the Gemini entry does not state that discovery declares no required properties: %v", gemini.Notes)
	}

	bedrock, ok := report["bedrock-converse/converse_response.schema.json"]
	if !ok {
		t.Fatal("unenforced.json has no entry for the Bedrock document")
	}
	if !mentions(bedrock.Notes, "base64") {
		t.Errorf("the Bedrock entry does not state that base64 blob bodies are unchecked: %v", bedrock.Notes)
	}

	// The gate never invents a closure: every closed object is one the vendor's
	// own specification closes. What is checkable here is that the accounting
	// adds up, and that no closure sits next to a composition keyword — that
	// combination cannot see the composed properties and would reject legal
	// payloads, which is the one failure mode worth failing the build over.
	for document, entry := range report {
		if entry.OpenObjects+entry.ClosedObjects != entry.ObjectSchemas {
			t.Errorf("%s: %d open + %d closed != %d object shapes",
				document, entry.OpenObjects, entry.ClosedObjects, entry.ObjectSchemas)
		}
		if entry.CompositionClosures != 0 {
			t.Errorf("%s: %d schema(s) close additionalProperties while composing with allOf/anyOf/oneOf; "+
				"those positions can reject a legal payload", document, entry.CompositionClosures)
		}
	}
}

// TestRequestDocumentsAreStricterThanResponseDocuments pins the reason the
// request half exists. Anthropic closes almost every object in its request body
// and almost none in its response, so an encoder that emits an undeclared field
// is caught outbound and would not be caught inbound.
func TestRequestDocumentsAreStricterThanResponseDocuments(t *testing.T) {
	t.Parallel()

	report := loadUnenforced(t)

	request, ok := report["anthropic/create_message_request.schema.json"]
	if !ok {
		t.Fatal("unenforced.json has no entry for the Anthropic request document")
	}
	response, ok := report["anthropic/message.schema.json"]
	if !ok {
		t.Fatal("unenforced.json has no entry for the Anthropic response document")
	}
	if request.ClosedObjects <= response.ClosedObjects {
		t.Errorf("anthropic request closes %d object shapes and the response closes %d; "+
			"the request document is supposed to be the stricter one",
			request.ClosedObjects, response.ClosedObjects)
	}
	if request.ClosedObjects < request.ObjectSchemas/2 {
		t.Errorf("anthropic request closes only %d of %d object shapes; the specification closes almost all of them",
			request.ClosedObjects, request.ObjectSchemas)
	}
}

// TestOverlappingUnionsAreRecorded keeps the oneOf relaxation visible. Those
// positions no longer assert that a value matches exactly one variant, and that
// is a real, if necessary, loss of strength.
func TestOverlappingUnionsAreRecorded(t *testing.T) {
	t.Parallel()

	report := loadUnenforced(t)
	entry, ok := report["openai-responses/create_response_request.schema.json"]
	if !ok {
		t.Fatal("unenforced.json has no entry for the OpenAI responses request document")
	}
	if len(entry.OverlappingOneOf) == 0 {
		t.Error("no oneOf relaxations recorded, but OpenAI's InputItem union overlaps by construction")
	}
	// The discriminated unions must NOT have been relaxed: they are what makes
	// an undefined variant detectable.
	for _, document := range []string{"anthropic/message.schema.json", "anthropic/stream_event.schema.json"} {
		if got := report[document].OverlappingOneOf; len(got) != 0 {
			t.Errorf("%s relaxed %d union(s); Anthropic discriminates every one of them by type", document, len(got))
		}
	}
}

// unenforcedEntry mirrors one document's entry in schema/unenforced.json. It is
// declared here rather than exported from the package because it is a reporting
// artefact, not part of the gate's contract.
type unenforcedEntry struct {
	Summary                string         `json:"summary"`
	Defs                   int            `json:"defs"`
	ObjectSchemas          int            `json:"object_schemas"`
	OpenObjects            int            `json:"open_objects"`
	ObjectsWithoutRequired int            `json:"objects_with_properties_but_no_required"`
	ClosedObjects          int            `json:"closed_objects"`
	CompositionClosures    int            `json:"composition_closures"`
	OverlappingOneOf       []string       `json:"overlapping_oneof_relaxed_to_anyof"`
	NullableWidened        []string       `json:"nullable_widened"`
	DroppedConstraints     []string       `json:"dropped_constraints"`
	FormatAnnotations      map[string]int `json:"format_annotations_not_asserted"`
	Notes                  []string       `json:"notes"`
}

func loadUnenforced(t *testing.T) map[string]unenforcedEntry {
	t.Helper()

	raw, err := fs.ReadFile(SchemaFS(), path.Join(schemaRoot, "unenforced.json"))
	if err != nil {
		t.Fatalf("read unenforced.json: %v", err)
	}
	var doc struct {
		Documents map[string]unenforcedEntry `json:"documents"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("parse unenforced.json: %v", err)
	}
	if len(doc.Documents) == 0 {
		t.Fatal("unenforced.json records no documents")
	}
	return doc.Documents
}

func mentions(notes []string, fragment string) bool {
	for _, note := range notes {
		if strings.Contains(note, fragment) {
			return true
		}
	}
	return false
}

func contains(list []string, value string) bool {
	for _, item := range list {
		if item == value {
			return true
		}
	}
	return false
}
