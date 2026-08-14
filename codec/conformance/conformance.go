package conformance

import (
	"bytes"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"path"
	"regexp"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// schemaTree holds the derived, checked-in schema documents. Embedding keeps
// the gate usable from any package's tests without a path relative to this
// directory, and guarantees the documents cannot be fetched at test time.
//
//go:embed schema
var schemaTree embed.FS

// schemaRoot is the directory prefix inside schemaTree.
const schemaRoot = "schema"

// resourceBase is the synthetic base URI the documents are registered under.
// It is never dereferenced: every reference inside a document is local, which
// is the whole point of rebasing them into $defs at generation time.
const resourceBase = "https://schemas.looprig.dev/llm/conformance/"

// registry is the process-wide compiled-schema cache. Documents are parsed and
// registered once; individual entry points are compiled on first use and kept,
// because a fixture suite validates the same handful of kinds thousands of
// times.
type registry struct {
	err      error
	index    Index
	compiler *jsonschema.Compiler

	mu       sync.Mutex
	compiled map[string]*jsonschema.Schema
}

var (
	registryOnce sync.Once
	shared       *registry
)

func load() *registry {
	registryOnce.Do(func() { shared = newRegistry() })
	return shared
}

func newRegistry() *registry {
	r := &registry{compiled: map[string]*jsonschema.Schema{}}

	raw, err := schemaTree.ReadFile(path.Join(schemaRoot, "index.json"))
	if err != nil {
		r.err = fmt.Errorf("conformance: read index: %w", err)
		return r
	}
	if err := json.Unmarshal(raw, &r.index); err != nil {
		r.err = fmt.Errorf("conformance: parse index: %w", err)
		return r
	}

	compiler := jsonschema.NewCompiler()
	compiler.DefaultDraft(jsonschema.Draft2020)
	// Refuse every load: a document that needs to resolve an external URI is a
	// generation bug, and silently reaching the network would break hermeticity.
	compiler.UseLoader(offlineLoader{})

	err = fs.WalkDir(schemaTree, schemaRoot, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, ".schema.json") {
			return err
		}
		body, err := schemaTree.ReadFile(p)
		if err != nil {
			return err
		}
		doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(body))
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		rel := strings.TrimPrefix(p, schemaRoot+"/")
		return compiler.AddResource(resourceBase+rel, doc)
	})
	if err != nil {
		r.err = fmt.Errorf("conformance: register schemas: %w", err)
		return r
	}
	r.compiler = compiler
	return r
}

// offlineLoader fails any attempt to resolve a schema the gate did not embed.
type offlineLoader struct{}

func (offlineLoader) Load(url string) (any, error) {
	return nil, fmt.Errorf("conformance: refusing to load %q; every schema reference must be local", url)
}

// schemaFor compiles, and caches, the subschema at pointer within document.
func (r *registry) schemaFor(document, pointer string) (*jsonschema.Schema, error) {
	if r.err != nil {
		return nil, r.err
	}
	loc := resourceBase + document + pointer

	r.mu.Lock()
	defer r.mu.Unlock()
	if sch, ok := r.compiled[loc]; ok {
		return sch, nil
	}
	sch, err := r.compiler.Compile(loc)
	if err != nil {
		return nil, fmt.Errorf("conformance: compile %s: %w", loc, err)
	}
	r.compiled[loc] = sch
	return sch, nil
}

// resolve looks up one kind. A kind may name a union member directly, as
// "stream_event/message_start", to validate an already-unwrapped frame.
func (r *registry) resolve(format, kind string) (*Entry, string, string, error) {
	if r.err != nil {
		return nil, "", "", r.err
	}
	kinds, ok := r.index[format]
	if !ok {
		return nil, "", "", fmt.Errorf("conformance: unknown api-format %q (have %s)", format, strings.Join(r.formats(), ", "))
	}
	base, member, hasMember := strings.Cut(kind, "/")
	entry, ok := kinds[base]
	if !ok {
		return nil, "", "", fmt.Errorf("conformance: unknown kind %q for api-format %q (have %s)",
			base, format, strings.Join(sortedIndexKeys(kinds), ", "))
	}
	if !hasMember {
		return entry, entry.Root, "", nil
	}
	if entry.Union == nil {
		return nil, "", "", fmt.Errorf("conformance: kind %q of api-format %q is not a union; %q names no member",
			base, format, kind)
	}
	pointer, ok := entry.Union.Members[member]
	if !ok {
		return nil, "", "", fmt.Errorf("conformance: union %s/%s has no member %q (have %s)",
			format, base, member, strings.Join(sortedIndexKeys(entry.Union.Members), ", "))
	}
	return entry, pointer, member, nil
}

func (r *registry) formats() []string { return sortedIndexKeys(r.index) }

func sortedIndexKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// MustValidate fails t unless payload is a legal message of the given kind for
// the given api-format. The payload is always the encoded body: the bytes on
// the wire, not a Go value. Call it on every fixture before the fixture reaches
// a Looprig decoder, and on every encoded request before it reaches a live API.
//
// On failure it reports the violating instance path and the schema keyword
// that rejected it, not a generic "does not validate": a fixture suite is only
// maintainable if a rejection tells you which byte to fix.
func MustValidate(t testing.TB, format, kind string, payload []byte) {
	t.Helper()
	if err := Validate(format, kind, payload); err != nil {
		t.Fatalf("%v", err)
	}
}

// MustValidateRequest fails t unless body is a legal request body of the given
// kind. body is what the encoder produced — the marshalled HTTP body, with any
// values the provider carries outside it (a Bedrock modelId in the URI path,
// for instance) already excluded, exactly as the schema models it.
//
// It is the same check as MustValidate with one addition: it refuses a kind
// that is not a request. Holding an encoded request against a response schema
// would appear to pass while proving nothing, so the direction mismatch is
// reported as the mistake it is.
func MustValidateRequest(t testing.TB, format, kind string, body []byte) {
	t.Helper()
	mustValidateDirected(t, DirectionRequest, format, kind, body)
}

// MustValidateResponse is MustValidateRequest's counterpart for inbound
// provider messages.
func MustValidateResponse(t testing.TB, format, kind string, payload []byte) {
	t.Helper()
	mustValidateDirected(t, DirectionResponse, format, kind, payload)
}

func mustValidateDirected(t testing.TB, want, format, kind string, payload []byte) {
	t.Helper()
	var err error
	if want == DirectionRequest {
		err = ValidateRequest(format, kind, payload)
	} else {
		err = ValidateResponse(format, kind, payload)
	}
	if err != nil {
		t.Fatalf("%v", err)
	}
}

// ValidateRequest validates an outbound request against a request schema and
// the semantic constraints that JSON Schema cannot express. Semantic checks
// run only after schema validation so the more precise instance-path
// diagnostic is never masked.
func ValidateRequest(format, kind string, body []byte) error {
	if err := checkDirection(DirectionRequest, format, kind); err != nil {
		return err
	}
	if err := Validate(format, kind, body); err != nil {
		return err
	}
	return checkRequestSemantics(format, kind, body)
}

// ValidateResponse validates an inbound provider payload against a response
// schema and refuses request kinds.
func ValidateResponse(format, kind string, payload []byte) error {
	if err := checkDirection(DirectionResponse, format, kind); err != nil {
		return err
	}
	return Validate(format, kind, payload)
}

// checkDirection reports a kind whose direction is not the one the caller
// intended.
func checkDirection(want, format, kind string) error {
	base, _, _ := strings.Cut(kind, "/")
	entry, ok := load().index[format][base]
	if !ok {
		// Leave the "unknown kind" diagnostic to resolve, which lists what does
		// exist.
		return nil
	}
	if entry.Direction != want {
		return fmt.Errorf("conformance: %s/%s is a %s kind, but it is being validated as a %s; "+
			"a %s body held against a %s schema proves nothing",
			format, kind, entry.Direction, want, want, entry.Direction)
	}
	return nil
}

// Validate is the non-testing form of MustValidate. It returns nil when the
// payload conforms and a diagnostic error otherwise.
func Validate(format, kind string, payload []byte) error {
	r := load()
	entry, pointer, member, err := r.resolve(format, kind)
	if err != nil {
		return err
	}

	instance, err := jsonschema.UnmarshalJSON(bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("conformance: %s/%s payload is not valid JSON: %w", format, kind, err)
	}

	selected := ""
	if member == "" && entry.Union != nil {
		// Validating against the whole union reports every branch's failure at
		// once, which is unreadable. If the payload says which member it is,
		// hold it to exactly that member.
		if name, ptr, narrowed, ok := focusUnion(entry.Union, instance); ok {
			pointer, selected, instance = ptr, name, narrowed
		}
	}

	schema, err := r.schemaFor(entry.Document, pointer)
	if err != nil {
		return err
	}
	if err := schema.Validate(instance); err != nil {
		var verr *jsonschema.ValidationError
		if !errors.As(err, &verr) {
			return fmt.Errorf("conformance: %s/%s: %w", format, kind, err)
		}
		return &Failure{
			Format:   format,
			Kind:     kind,
			Document: entry.Document,
			Pointer:  pointer,
			Selected: selected,
			Union:    entry.Union,
			Err:      verr,
		}
	}
	return nil
}

// focusUnion picks the member a payload claims to be, and returns the value to
// hold against that member's schema. It reports ok=false whenever the payload
// makes no claim the index recognises, in which case the caller falls back to
// the whole union, whose own error says so.
//
// The narrowed value differs by style, because the two encodings put the member
// in different places. A property-style union (OpenAI, Anthropic) tags the
// message itself, so the member schema describes the whole object. A Smithy
// union wraps the member: {"contentBlockDelta": {...}} means the member schema
// describes the wrapped value, not the wrapper.
func focusUnion(u *Union, instance any) (name, pointer string, narrowed any, ok bool) {
	obj, isObj := instance.(map[string]any)
	if !isObj {
		return "", "", nil, false
	}
	switch u.Style {
	case UnionStyleProperty:
		value, present := obj[u.Property]
		if !present {
			return "", "", nil, false
		}
		text, isText := value.(string)
		if !isText {
			return "", "", nil, false
		}
		pointer, ok = u.Members[text]
		return text, pointer, instance, ok
	case UnionStyleMemberKey:
		// Exactly one member property may be set. Zero or several is itself a
		// violation of the union, so hand those back to the union schema.
		var found string
		for key := range obj {
			if _, isMember := u.Members[key]; !isMember {
				continue
			}
			if found != "" {
				return "", "", nil, false
			}
			found = key
		}
		if found == "" {
			return "", "", nil, false
		}
		return found, u.Members[found], obj[found], true
	default:
		return "", "", nil, false
	}
}

// Failure is a schema violation rendered for a test log.
type Failure struct {
	Format   string
	Kind     string
	Document string
	Pointer  string
	// Selected names the union member the payload claimed to be, when the gate
	// narrowed the check to it.
	Selected string
	Union    *Union
	Err      *jsonschema.ValidationError
}

func (f *Failure) Unwrap() error { return f.Err }

func (f *Failure) Error() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "conformance: %s/%s fixture is not a legal provider payload\n", f.Format, f.Kind)
	fmt.Fprintf(&sb, "  schema:  %s%s\n", f.Document, f.Pointer)
	if f.Selected != "" && f.Union != nil {
		switch f.Union.Style {
		case UnionStyleProperty:
			fmt.Fprintf(&sb, "  member:  selected by %s=%q\n", f.Union.Property, f.Selected)
		case UnionStyleMemberKey:
			fmt.Fprintf(&sb, "  member:  selected by the present member property %q\n", f.Selected)
		}
	}
	if f.Selected != "" && f.Union != nil && f.Union.Style == UnionStyleMemberKey {
		sb.WriteString("           (instance paths below are relative to that member's value)\n")
	}
	units := focusUnionLeaves(leafViolations(f.Err.DetailedOutput()))
	fmt.Fprintf(&sb, "  %d violation(s):\n", len(units))
	for _, u := range units {
		at := u.InstanceLocation
		if at == "" {
			at = "(root)"
		}
		fmt.Fprintf(&sb, "    at %s: %s\n", at, u.Error)
		fmt.Fprintf(&sb, "        keyword %s\n", u.KeywordLocation)
	}
	return strings.TrimRight(sb.String(), "\n")
}

// focusUnionLeaves drops the branches of a nested union that the payload was
// never claiming to be.
//
// Content blocks are unions twenty branches wide, and a single missing property
// makes every branch fail: nineteen of them because the payload's "type" is not
// theirs, one of them for the real reason. Reporting all twenty buries the
// answer. Where at least one branch failed for a reason OTHER than its
// discriminator, only those branches are kept — they are the ones the payload
// claimed to be. Where every branch failed on its discriminator the payload
// names a variant that does not exist, and all of them are kept, because that
// full list is then the useful message.
//
// This is presentation only. Nothing is hidden that changes the verdict, and a
// union the heuristic cannot read falls through unpruned.
func focusUnionLeaves(units []*jsonschema.OutputUnit) []*jsonschema.OutputUnit {
	return focusUnionLeavesFrom(units, 0)
}

// focusUnionLeavesFrom prunes the outermost union at or after keyword-location
// offset from. Recursing with the offset advanced past the branch just entered
// is what lets it descend through nested unions instead of re-reading the one
// it already handled.
func focusUnionLeavesFrom(units []*jsonschema.OutputUnit, from int) []*jsonschema.OutputUnit {
	if len(units) < 2 {
		return units
	}

	// Find the outermost union all these violations pass through.
	prefix, ok := shallowestUnionPrefix(units, from)
	if !ok {
		return units
	}

	branches := map[string][]*jsonschema.OutputUnit{}
	var order []string
	var passthrough []*jsonschema.OutputUnit
	for _, unit := range units {
		key, ok := branchKey(unit.KeywordLocation, prefix)
		if !ok {
			passthrough = append(passthrough, unit)
			continue
		}
		if _, seen := branches[key]; !seen {
			order = append(order, key)
		}
		branches[key] = append(branches[key], unit)
	}
	if len(order) < 2 {
		return units
	}

	root := shallowestInstance(units)
	var kept []string
	for _, key := range order {
		if !hasDiscriminatorFailure(branches[key], root) {
			kept = append(kept, key)
		}
	}
	if len(kept) == 0 || len(kept) == len(order) {
		kept = order
	}

	out := passthrough
	for _, key := range kept {
		out = append(out, focusUnionLeavesFrom(branches[key], len(key))...)
	}
	return out
}

// unionBranch matches a oneOf/anyOf branch step in a keyword location.
var unionBranch = regexp.MustCompile(`/(oneOf|anyOf)/\d+`)

// shallowestUnionPrefix returns the keyword-location prefix of the outermost
// union at or after offset from that the violations pass through.
func shallowestUnionPrefix(units []*jsonschema.OutputUnit, from int) (string, bool) {
	best := ""
	for _, unit := range units {
		if len(unit.KeywordLocation) <= from {
			continue
		}
		loc := unionBranch.FindStringIndex(unit.KeywordLocation[from:])
		if loc == nil {
			continue
		}
		candidate := unit.KeywordLocation[:from+loc[1]]
		trimmed := candidate[:strings.LastIndex(candidate, "/")]
		if best == "" || len(trimmed) < len(best) {
			best = trimmed
		}
	}
	return best, best != ""
}

// branchKey returns the prefix identifying which branch of prefix's union a
// violation came from.
func branchKey(location, prefix string) (string, bool) {
	if !strings.HasPrefix(location, prefix+"/") {
		return "", false
	}
	rest := location[len(prefix)+1:]
	index, _, _ := strings.Cut(rest, "/")
	if index == "" {
		return "", false
	}
	for _, r := range index {
		if r < '0' || r > '9' {
			return "", false
		}
	}
	return prefix + "/" + index, true
}

// hasDiscriminatorFailure reports whether a branch rejected the payload because
// its discriminator names a different variant.
//
// One such failure is enough to disqualify the branch: a wrong-variant branch
// reports its missing properties and its closed-object violations too, so
// demanding that the discriminator be its ONLY complaint would disqualify
// nothing. What matters is that the correct branch has no discriminator
// complaint at all, because its discriminator matched.
//
// A discriminator check is a const or enum assertion on an immediate property
// of the value being matched. An enum failure deeper inside the branch is a
// real error rather than a variant mismatch, which is what stops a bad nested
// enum from suppressing the branch that found it.
func hasDiscriminatorFailure(branch []*jsonschema.OutputUnit, root string) bool {
	for _, unit := range branch {
		keyword := unit.KeywordLocation
		if !strings.HasSuffix(keyword, "/const") && !strings.HasSuffix(keyword, "/enum") {
			continue
		}
		if isImmediateProperty(unit.InstanceLocation, root) {
			return true
		}
	}
	return false
}

// isImmediateProperty reports whether location names a direct property of root.
func isImmediateProperty(location, root string) bool {
	rest, ok := strings.CutPrefix(location, root+"/")
	return ok && !strings.Contains(rest, "/")
}

// shallowestInstance returns the least deeply nested instance location among a
// set of violations, which is the value the enclosing union is matching.
func shallowestInstance(units []*jsonschema.OutputUnit) string {
	best := ""
	depth := -1
	for _, unit := range units {
		d := strings.Count(unit.InstanceLocation, "/")
		if depth == -1 || d < depth {
			best, depth = unit.InstanceLocation, d
		}
	}
	return best
}

// leafViolations reduces the validator's nested output to its leaves.
//
// The interior units are structural — "'allOf' failed", "validation failed" at
// a $ref — and carry no information a fixture author can act on. The leaves are
// the ones that say "missing property 'usage'" or "value must be 'assistant'",
// which is the whole reason for preferring a diagnostic over a boolean.
func leafViolations(unit *jsonschema.OutputUnit) []*jsonschema.OutputUnit {
	var out []*jsonschema.OutputUnit
	var walk func(*jsonschema.OutputUnit)
	walk = func(u *jsonschema.OutputUnit) {
		if u == nil {
			return
		}
		if len(u.Errors) == 0 {
			if u.Error != nil {
				out = append(out, u)
			}
			return
		}
		for i := range u.Errors {
			walk(&u.Errors[i])
		}
	}
	walk(unit)
	if len(out) == 0 && unit != nil {
		out = append(out, unit)
	}
	return out
}

// MustValidateStream fails t unless every data frame of an SSE body is a legal
// message of the given kind. It returns the number of frames validated.
//
// Frames are validated one at a time, against the event union, so a suite can
// assert that each individual chunk a provider would emit is legal rather than
// only that the concatenation parses. A body with no frames fails: an empty
// fixture is not evidence of anything.
func MustValidateStream(t testing.TB, format, kind string, body []byte) int {
	t.Helper()

	frames, err := ParseSSE(body)
	if err != nil {
		t.Fatalf("conformance: %s/%s stream fixture: %v", format, kind, err)
	}
	if len(frames) == 0 {
		t.Fatalf("conformance: %s/%s stream fixture contains no data frames", format, kind)
	}

	entry, _, _, err := load().resolve(format, kind)
	if err != nil {
		t.Fatalf("%v", err)
	}

	for i, frame := range frames {
		if err := Validate(format, kind, frame.Data); err != nil {
			t.Fatalf("frame %d (event %q):\n%v", i, frame.Event, err)
		}
		// An SSE "event:" name that disagrees with the payload's own
		// discriminator is a fixture that could not have come off the wire,
		// even though each half is individually legal.
		if frame.Event == "" || entry.Union == nil || entry.Union.Style != UnionStyleProperty {
			continue
		}
		var probe map[string]any
		if err := json.Unmarshal(frame.Data, &probe); err != nil {
			continue
		}
		if got, ok := probe[entry.Union.Property].(string); ok && got != frame.Event {
			t.Fatalf("conformance: %s/%s frame %d: SSE event name %q disagrees with payload %s=%q",
				format, kind, i, frame.Event, entry.Union.Property, got)
		}
	}
	return len(frames)
}

// Frame is one server-sent event.
type Frame struct {
	// Event is the value of the "event:" field, empty when absent.
	Event string
	// Data is the concatenated "data:" payload.
	Data []byte
}

// terminator is the sentinel OpenAI sends to close a stream. It is not JSON and
// is skipped rather than validated.
const terminator = "[DONE]"

// ParseSSE splits a server-sent-event body into its data frames. It implements
// only what provider streams use — "event:", "data:", blank-line separation and
// comment lines — and rejects anything it does not understand rather than
// guessing, so a malformed stream fixture fails the gate instead of silently
// validating a subset of its frames.
func ParseSSE(body []byte) ([]Frame, error) {
	var (
		frames []Frame
		event  string
		data   []string
	)
	flush := func() error {
		defer func() { event, data = "", nil }()
		if data == nil {
			// A dispatch with no data buffer is ignored by the SSE
			// specification, which is what a bare "retry:" produces. A NAMED
			// event with no payload is different: no provider sends one, so a
			// fixture containing one is broken and says so.
			if event != "" {
				return fmt.Errorf("event %q carries no data", event)
			}
			return nil
		}
		joined := strings.Join(data, "\n")
		if strings.TrimSpace(joined) == terminator {
			return nil
		}
		if strings.TrimSpace(joined) == "" {
			return fmt.Errorf("event %q has an empty data payload", event)
		}
		frames = append(frames, Frame{Event: event, Data: []byte(joined)})
		return nil
	}

	for n, raw := range strings.Split(strings.ReplaceAll(string(body), "\r\n", "\n"), "\n") {
		line := strings.TrimSuffix(raw, "\r")
		switch {
		case line == "":
			if err := flush(); err != nil {
				return nil, fmt.Errorf("line %d: %w", n+1, err)
			}
		case strings.HasPrefix(line, ":"):
			// Comment / keep-alive.
		case strings.HasPrefix(line, "data:"):
			data = append(data, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
		case strings.HasPrefix(line, "event:"):
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "id:"), strings.HasPrefix(line, "retry:"):
			// Reconnection metadata; it carries no payload of its own.
		default:
			return nil, fmt.Errorf("line %d: %q is not a server-sent-event field", n+1, line)
		}
	}
	if err := flush(); err != nil {
		return nil, err
	}
	return frames, nil
}

// Formats lists every api-format the gate knows.
func Formats() []string { return load().formats() }

// Kinds lists every message kind registered for an api-format.
func Kinds(format string) []string { return sortedIndexKeys(load().index[format]) }

// Lookup exposes one index entry, for tests that assert the gate's coverage.
func Lookup(format, kind string) (*Entry, bool) {
	entry, ok := load().index[format][kind]
	return entry, ok
}

// LoadProvenance reads the checked-in provenance record.
func LoadProvenance() (Provenance, error) {
	var p Provenance
	raw, err := schemaTree.ReadFile(path.Join(schemaRoot, "provenance.json"))
	if err != nil {
		return p, err
	}
	err = json.Unmarshal(raw, &p)
	return p, err
}

// SchemaFS exposes the embedded schema tree so tests can assert on the files
// themselves without reaching outside the package directory.
func SchemaFS() fs.FS { return schemaTree }
