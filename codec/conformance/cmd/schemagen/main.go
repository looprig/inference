// Command schemagen derives the checked-in JSON Schema 2020-12 documents that
// back the provider conformance gate.
//
// It reads the four official upstream API descriptions, converts each one from
// its own dialect into standalone 2020-12 documents (one per api-format and
// message kind, with every reference rebased into a local $defs so the
// documents validate offline), and writes them plus an index, a provenance
// record and a report of everything the derived schemas do not enforce.
//
// Refresh path (the only supported way to regenerate; never hand-edit the
// output):
//
//	go run ./codec/conformance/cmd/schemagen \
//	    -fetch -specs /tmp/looprig-inference-conformance-specs \
//	    -out codec/conformance/schema
//
// -fetch downloads each source from the URL recorded in provenance.json before
// converting. Without it the tool converts whatever is already in -specs, which
// keeps regeneration reproducible offline. Either way the resulting tree is
// committed as-is: upstream drift shows up as a git diff, and the SHA-256 of
// every source document (plus a key-sorted canonical hash, for sources whose
// byte encoding is not stable) is recorded in provenance.json.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	yaml "go.yaml.in/yaml/v3"

	"github.com/looprig/inference/codec/conformance"
)

// source describes one upstream API description and how to read it.
type source struct {
	// key names the source in provenance.json and on the command line.
	key string
	// url is the exact document fetched by -fetch.
	url string
	// file is the name the document is stored under inside -specs.
	file string
	// dialect selects the converter.
	dialect string
	// publisher records who publishes the bytes we consume. Every source but
	// Anthropic's is served by the vendor's own first-party infrastructure.
	publisher string
	// hosting is empty for first-party-hosted sources and otherwise explains
	// the chain of custody.
	hosting string
	// canonical requests a key-sorted canonical hash in addition to the raw
	// byte hash. Only meaningful for sources whose serialisation is unstable.
	canonical bool
}

// sources is the authoritative list of upstream documents. Adding a provider
// format starts here.
var sources = []source{
	{
		key:       "openai",
		url:       "https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml",
		file:      "openai.yaml",
		dialect:   dialectOpenAPI,
		publisher: "OpenAI, from its own openai/openai-openapi repository",
	},
	{
		key:     "anthropic",
		url:     "https://storage.googleapis.com/stainless-sdk-openapi-specs/anthropic/anthropic-086fd8de69e181b730041d853827045c2df13e50b16ea2e1d4cb97b793d90caf.yml",
		file:    "anthropic.yml",
		dialect: dialectOpenAPI,
		publisher: "Anthropic, via the openapi_spec_url published in its own " +
			"anthropics/anthropic-sdk-go repository (.stats.yml)",
		hosting: "ONLY non-first-party-HOSTED source. Anthropic publishes no OpenAPI " +
			"document on its own infrastructure: api.anthropic.com/openapi.json returns " +
			"404 and the docs.anthropic.com / docs.claude.com / platform.claude.com " +
			"openapi.json paths all redirect to a 404 page. The document below is the " +
			"one Anthropic itself points its official Go SDK at, so the pointer is " +
			"first-party even though the bytes are served by Stainless. The pointer " +
			"file is recorded alongside it as anthropic-stats so substitution of either " +
			"the pointer or the document is detectable.",
	},
	{
		key:       "anthropic-stats",
		url:       "https://raw.githubusercontent.com/anthropics/anthropic-sdk-go/main/.stats.yml",
		file:      "anthropic-stats.yml",
		dialect:   dialectPointer,
		publisher: "Anthropic, from its own anthropics/anthropic-sdk-go repository",
		hosting: "Pointer file only; it is not converted. It records the " +
			"openapi_spec_url the anthropic source is fetched from and Anthropic's " +
			"own openapi_spec_hash for that document.",
	},
	{
		key:       "gemini",
		url:       "https://generativelanguage.googleapis.com/$discovery/rest?version=v1beta",
		file:      "gemini-discovery.json",
		dialect:   dialectDiscovery,
		publisher: "Google, from the production generativelanguage.googleapis.com discovery endpoint",
		canonical: true,
	},
	{
		key:       "bedrock",
		url:       "https://raw.githubusercontent.com/aws/api-models-aws/main/models/bedrock-runtime/service/2023-09-30/bedrock-runtime-2023-09-30.json",
		file:      "bedrock-runtime.json",
		dialect:   dialectSmithy,
		publisher: "AWS, from its own aws/api-models-aws model distribution",
	},
}

const (
	directionRequest  = conformance.DirectionRequest
	directionResponse = conformance.DirectionResponse
)

const (
	dialectOpenAPI   = "openapi-3.1"
	dialectDiscovery = "google-discovery-v1"
	dialectSmithy    = "smithy-2.0-ast"
	dialectPointer   = "pointer"
)

// target names one emitted schema document: an api-format, a message kind, and
// the source definition that is its root.
type target struct {
	// format is the Looprig api-format label the gate is keyed by.
	format string
	// kind is the message kind within that format.
	kind string
	// source is the key of the upstream document it is derived from.
	source string
	// root is the definition name in that document.
	root string
	// union, when set, names the property whose value discriminates the
	// members of a root that is a union of message shapes. "" means the root
	// is not a union; smithyMemberKey means the union is keyed by which single
	// member property is present (the Smithy union encoding).
	union string
	// direction says whether the kind describes what Looprig sends or what the
	// provider returns. It is recorded in the index so a test that validates an
	// encoded request against a response schema fails on the mistake rather
	// than on the payload.
	direction string
}

// smithyMemberKey marks a union whose branch is chosen by the name of the one
// property that is present, rather than by a discriminator property's value.
const smithyMemberKey = "\x00member-key"

// targets is the full set of emitted documents.
//
// Request kinds matter more than response kinds, and are listed first for that
// reason. A response schema tells us whether we understood the provider; a
// request schema tells us whether the provider will understand us, and it does
// so before the bytes leave the process. Request schemas are also the stricter
// half of every specification: they carry the required lists and the
// additionalProperties:false closures that response schemas routinely omit,
// which is exactly where an encoder bug hides.
var targets = []target{
	{format: "openai", kind: "chat_completion_request", source: "openai", root: "CreateChatCompletionRequest", direction: directionRequest},
	{format: "openai-responses", kind: "create_response_request", source: "openai", root: "CreateResponse", direction: directionRequest},
	{format: "anthropic", kind: "create_message_request", source: "anthropic", root: "CreateMessageParams", direction: directionRequest},
	{format: "gemini", kind: "generate_content_request", source: "gemini", root: "GenerateContentRequest", direction: directionRequest},
	{format: "bedrock-converse", kind: "converse_request", source: "bedrock", root: "ConverseRequest", direction: directionRequest},

	{format: "openai", kind: "chat_completion", source: "openai", root: "CreateChatCompletionResponse", direction: directionResponse},
	{format: "openai", kind: "chat_completion_chunk", source: "openai", root: "CreateChatCompletionStreamResponse", direction: directionResponse},

	{format: "openai-responses", kind: "response", source: "openai", root: "Response", direction: directionResponse},
	{format: "openai-responses", kind: "stream_event", source: "openai", root: "ResponseStreamEvent", union: "type", direction: directionResponse},

	{format: "anthropic", kind: "message", source: "anthropic", root: "Message", direction: directionResponse},
	{format: "anthropic", kind: "stream_event", source: "anthropic", root: "MessageStreamEvent", union: "type", direction: directionResponse},

	{format: "gemini", kind: "generate_content_response", source: "gemini", root: "GenerateContentResponse", direction: directionResponse},

	{format: "bedrock-converse", kind: "converse_response", source: "bedrock", root: "ConverseResponse", direction: directionResponse},
	{format: "bedrock-converse", kind: "converse_stream_output", source: "bedrock", root: "ConverseStreamOutput", union: smithyMemberKey, direction: directionResponse},
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("schemagen: ")

	specs := flag.String("specs", "", "directory holding the upstream API descriptions")
	out := flag.String("out", "", "directory to write the derived schema tree into")
	fetch := flag.Bool("fetch", false, "download each source from its recorded URL into -specs first")
	suite := flag.String("suite", "", "vendored JSON-Schema-Test-Suite tree (default: <out>/../testdata/jsonschema-suite)")
	flag.Parse()

	if *specs == "" || *out == "" {
		flag.Usage()
		log.Fatal("both -specs and -out are required")
	}
	if *suite == "" {
		*suite = filepath.Join(*out, "..", "testdata", "jsonschema-suite")
	}
	if err := run(*specs, *out, *suite, *fetch); err != nil {
		log.Fatal(err)
	}
}

func run(specsDir, outDir, suiteDir string, fetch bool) error {
	if err := os.MkdirAll(specsDir, 0o750); err != nil {
		return err
	}
	if fetch {
		for _, s := range sources {
			if err := download(s, specsDir); err != nil {
				return fmt.Errorf("fetch %s: %w", s.key, err)
			}
		}
	}

	previous := loadPreviousProvenance(outDir)
	today := time.Now().UTC().Format("2006-01-02")

	docs := map[string]any{}
	prov := conformance.Provenance{
		Comment: provenanceComment,
		Sources: map[string]*conformance.Source{},
	}

	for _, s := range sources {
		raw, err := os.ReadFile(filepath.Join(specsDir, s.file)) // #nosec G304 -- operator-supplied spec directory
		if err != nil {
			return fmt.Errorf("read %s: %w", s.key, err)
		}
		rec := &conformance.Source{
			URL:       s.url,
			File:      s.file,
			Dialect:   s.dialect,
			Publisher: s.publisher,
			Hosting:   s.hosting,
			Bytes:     len(raw),
			SHA256:    hashHex(raw),
		}
		parsed, err := parseSource(s, raw)
		if err != nil {
			return fmt.Errorf("parse %s: %w", s.key, err)
		}
		if s.canonical {
			canon, err := canonicalHash(parsed)
			if err != nil {
				return fmt.Errorf("canonicalise %s: %w", s.key, err)
			}
			rec.CanonicalSHA256 = canon
			rec.CanonicalNote = "the endpoint does not serve byte-stable JSON; compare canonical_sha256 across refreshes"
		}
		rec.Retrieved = retrievedDate(previous, s.key, rec.SHA256, today)
		if s.key == "anthropic-stats" {
			rec.Fields = statsFields(raw)
		}
		prov.Sources[s.key] = rec
		docs[s.key] = parsed
	}
	if err := crossCheckAnthropicPointer(prov.Sources); err != nil {
		return err
	}
	suiteRec, err := suiteRecord(suiteDir, previous, today)
	if err != nil {
		return err
	}
	prov.Sources[suiteProvenanceKey] = suiteRec

	rep := newReport()
	idx := index{}
	written := map[string][]byte{}

	for _, tgt := range targets {
		src := sourceByKey(tgt.source)
		doc, ok := docs[tgt.source]
		if !ok {
			return fmt.Errorf("target %s/%s: unknown source %q", tgt.format, tgt.kind, tgt.source)
		}
		rel := filepath.ToSlash(filepath.Join(tgt.format, tgt.kind+".schema.json"))
		docRep := rep.forDocument(rel)
		conv, err := newConverter(src.dialect, doc, docRep)
		if err != nil {
			return fmt.Errorf("target %s/%s: %w", tgt.format, tgt.kind, err)
		}
		built, entry, err := buildDocument(conv, tgt, rel, docRep)
		if err != nil {
			return fmt.Errorf("target %s/%s: %w", tgt.format, tgt.kind, err)
		}
		encoded, err := marshalIndent(built)
		if err != nil {
			return err
		}
		written[rel] = encoded
		idx.add(tgt.format, tgt.kind, entry)
	}

	if err := writeTree(outDir, written); err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(outDir, "index.json"), idx); err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(outDir, "provenance.json"), prov); err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(outDir, "unenforced.json"), rep.finish()); err != nil {
		return err
	}

	for _, rel := range sortedKeys(written) {
		log.Printf("%-52s %7d bytes", rel, len(written[rel]))
	}
	return nil
}

func sourceByKey(key string) source {
	for _, s := range sources {
		if s.key == key {
			return s
		}
	}
	panic("schemagen: unknown source " + key)
}

// parseSource decodes one upstream document. JSON is decoded with UseNumber so
// that integer constraints survive the round trip verbatim; YAML is decoded
// through yaml and then normalised to JSON-compatible values.
func parseSource(s source, raw []byte) (any, error) {
	switch {
	case strings.HasSuffix(s.file, ".json"), looksLikeJSON(raw):
		dec := json.NewDecoder(strings.NewReader(string(raw)))
		dec.UseNumber()
		var v any
		if err := dec.Decode(&v); err != nil {
			return nil, err
		}
		return v, nil
	default:
		var v any
		if err := yaml.Unmarshal(raw, &v); err != nil {
			return nil, err
		}
		return normalizeYAML(v)
	}
}

func looksLikeJSON(raw []byte) bool {
	trimmed := strings.TrimLeft(string(raw), " \t\r\n")
	return strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[")
}

// normalizeYAML converts a yaml-decoded tree into the same value shapes the
// JSON decoder produces, so one converter can serve both dialects. Non-string
// mapping keys are an error rather than a silent coercion: an OpenAPI document
// must not have them, and quietly stringifying one would hide a spec change.
func normalizeYAML(v any) (any, error) {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			nv, err := normalizeYAML(val)
			if err != nil {
				return nil, err
			}
			out[k] = nv
		}
		return out, nil
	case map[any]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			ks, ok := k.(string)
			if !ok {
				return nil, fmt.Errorf("non-string mapping key %v (%T)", k, k)
			}
			nv, err := normalizeYAML(val)
			if err != nil {
				return nil, err
			}
			out[ks] = nv
		}
		return out, nil
	case []any:
		out := make([]any, len(t))
		for i, val := range t {
			nv, err := normalizeYAML(val)
			if err != nil {
				return nil, err
			}
			out[i] = nv
		}
		return out, nil
	default:
		return v, nil
	}
}

func download(s source, dir string) error {
	req, err := http.NewRequest(http.MethodGet, s.url, nil)
	if err != nil {
		return err
	}
	client := &http.Client{Timeout: 2 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: %s", s.url, resp.Status)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	log.Printf("fetched %-16s %8d bytes", s.key, len(body))
	return os.WriteFile(filepath.Join(dir, s.file), body, 0o600)
}

func hashHex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// canonicalHash hashes a key-sorted, whitespace-free encoding of a parsed
// document. Google's discovery endpoint reorders keys between responses, so the
// raw byte hash alone cannot distinguish a re-fetch from a real change.
func canonicalHash(v any) (string, error) {
	buf, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return hashHex(buf), nil
}

// retrievedDate keeps the recorded date stable when a refresh produced
// byte-identical input, so regeneration does not create diff noise.
func retrievedDate(previous *conformance.Provenance, key, sha, today string) string {
	if previous == nil {
		return today
	}
	if rec, ok := previous.Sources[key]; ok && rec.SHA256 == sha && rec.Retrieved != "" {
		return rec.Retrieved
	}
	return today
}

func loadPreviousProvenance(outDir string) *conformance.Provenance {
	raw, err := os.ReadFile(filepath.Join(outDir, "provenance.json")) // #nosec G304 -- operator-supplied output directory
	if err != nil {
		return nil
	}
	var p conformance.Provenance
	if err := json.Unmarshal(raw, &p); err != nil {
		return nil
	}
	return &p
}

// statsFields extracts the key/value pairs of Anthropic's .stats.yml so the
// pointer it publishes is recorded verbatim next to our own hash of the
// document that pointer resolves to.
func statsFields(raw []byte) map[string]string {
	out := map[string]string{}
	for _, line := range strings.Split(string(raw), "\n") {
		key, value, ok := strings.Cut(strings.TrimSpace(line), ":")
		if !ok || strings.HasPrefix(key, "#") {
			continue
		}
		out[strings.TrimSpace(key)] = strings.TrimSpace(value)
	}
	return out
}

// crossCheckAnthropicPointer fails the generation if the Anthropic document we
// converted is not the one Anthropic's own SDK repository points at. This is
// the substitution check that the non-first-party hosting makes necessary.
func crossCheckAnthropicPointer(recs map[string]*conformance.Source) error {
	stats, ok := recs["anthropic-stats"]
	if !ok || stats.Fields == nil {
		return errors.New("anthropic-stats pointer file is missing")
	}
	want := stats.Fields["openapi_spec_url"]
	got := recs["anthropic"].URL
	if want == "" {
		return errors.New("anthropic-stats has no openapi_spec_url")
	}
	if want != got {
		return fmt.Errorf("anthropic source URL %q no longer matches openapi_spec_url %q in .stats.yml; "+
			"update the sources table and re-fetch", got, want)
	}
	recs["anthropic"].PointerSource = sourceByKey("anthropic-stats").url
	recs["anthropic"].PointerHash = stats.Fields["openapi_spec_hash"]
	recs["anthropic"].PointerHashNote = "openapi_spec_hash as published by Anthropic in .stats.yml. It is " +
		"Stainless's hash of its internal canonical spec, not of the served bytes, so it will not equal " +
		"sha256 above; it is recorded so a change to Anthropic's published pointer is visible."
	return nil
}

func writeTree(outDir string, files map[string][]byte) error {
	// A removed target must remove its old generated document too. Otherwise a
	// regeneration can leave an unindexed schema behind and make the checked-in
	// tree depend on whatever happened to exist before the command ran.
	if err := filepath.WalkDir(outDir, func(full string, entry fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return nil
			}
			return err
		}
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".schema.json") {
			return nil
		}
		rel, err := filepath.Rel(outDir, full)
		if err != nil {
			return err
		}
		if _, current := files[filepath.ToSlash(rel)]; current {
			return nil
		}
		return os.Remove(full)
	}); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	for _, rel := range sortedKeys(files) {
		full := filepath.Join(outDir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0o750); err != nil {
			return err
		}
		if err := os.WriteFile(full, files[rel], 0o600); err != nil {
			return err
		}
	}
	return nil
}

func writeJSON(path string, v any) error {
	buf, err := marshalIndent(v)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return err
	}
	return os.WriteFile(path, buf, 0o600)
}

// marshalIndent encodes with HTML escaping disabled so that patterns and
// descriptions survive unmangled, and with a trailing newline so the files are
// well-formed text.
func marshalIndent(v any) ([]byte, error) {
	var sb strings.Builder
	enc := json.NewEncoder(&sb)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return []byte(sb.String()), nil
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// provenanceComment heads schema/provenance.json.
const provenanceComment = "Chain of custody for every upstream API description the conformance schemas are " +
	"derived from. Generated by cmd/schemagen; do not edit by hand. Each entry records the exact URL " +
	"fetched, who publishes it, the date the recorded bytes were last observed, and their SHA-256. " +
	"Re-run the generator with -fetch to refresh: an upstream change shows up as a diff in these hashes " +
	"and in the derived schema tree. Sources whose serialisation is not byte-stable also carry a " +
	"canonical_sha256 over a key-sorted re-encoding, which is the hash to compare across refreshes."
