package conformance

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// The conformance gate delegates every assertion to a third-party JSON Schema
// implementation, because no first-party Go implementation exists: the JSON
// Schema organisation publishes a language-agnostic test suite and no Go
// validator, so every Go option (santhosh-tekuri, invopop, qri-io, ...) is
// third-party and hand-rolling one would defeat the purpose of the gate.
//
// Rather than trust the library's own claims, this file runs the
// organisation's official draft2020-12 suite against it on every `go test`
// run, from a checked-in copy. A regression in the validator is therefore a
// build failure here, not a silent weakening of every provider fixture
// assertion. The vendored copy, its upstream commit, and the refresh path are
// recorded in schema/provenance.json under "jsonschema-test-suite".
//
// The suite is split exactly as the specification splits it:
//
//   - Required vocabulary (tests/draft2020-12/*.json). These are the keywords
//     the gate actually asserts. TestOfficialJSONSchemaSuite runs every file
//     with no skip list and no allowlist; all of them must pass.
//   - Optional format assertion (tests/draft2020-12/optional/format/*.json).
//     "format" is an annotation in draft 2020-12 unless a validator opts in,
//     and the gate deliberately does not opt in (see doc.go). Those cases are
//     still executed, and their exact outcome is pinned to a checked-in record
//     so that neither a regression nor an improvement can pass unnoticed.
//
// tests/draft2020-12/optional/ files other than the format directory are not
// vendored: they cover cross-draft, bignum and ECMAScript-regex behaviour that
// the gate's schemas never exercise. That exclusion is a vendoring decision
// visible in testdata/, not a hidden skip inside a passing test run.
const suiteRoot = "testdata/jsonschema-suite"

// suiteRemotesPrefix is the base URL the suite uses for its remote-reference
// fixtures. The upstream project serves the remotes/ tree from a local HTTP
// server on this port; we resolve the same URLs from the checked-in tree so
// the run stays offline.
const suiteRemotesPrefix = "http://localhost:1234/"

// optionalFormatDir is the suite-relative prefix of the optional format cases.
const optionalFormatDir = "optional/format/"

// formatSupportRecord is the checked-in, exact record of how the validator
// behaves on the optional format-assertion cases.
const formatSupportRecord = "testdata/format-assertion-support.json"

// suiteCase mirrors one entry of a JSON-Schema-Test-Suite file.
type suiteCase struct {
	Description string `json:"description"`
	Schema      any    `json:"schema"`
	Tests       []struct {
		Description string `json:"description"`
		Data        any    `json:"data"`
		Valid       bool   `json:"valid"`
	} `json:"tests"`
}

// formatSupport is the serialised shape of formatSupportRecord: for every
// optional format file, the number of cases executed and the exact description
// of every case whose outcome disagrees with the suite.
type formatSupport struct {
	Comment string                    `json:"comment"`
	Files   map[string]formatFileStat `json:"files"`
}

type formatFileStat struct {
	Cases       int      `json:"cases"`
	Unsupported []string `json:"unsupported"`
}

// suiteRemoteLoader resolves the suite's http://localhost:1234/ references from
// the checked-in remotes tree. Anything else is refused so a missing fixture
// surfaces as a test failure instead of a network fetch.
type suiteRemoteLoader string

func (l suiteRemoteLoader) Load(url string) (any, error) {
	rest, ok := strings.CutPrefix(url, suiteRemotesPrefix)
	if !ok {
		return nil, fmt.Errorf("conformance: suite loader refuses non-local URL %q", url)
	}
	file, err := os.Open(filepath.Join(string(l), "remotes", filepath.FromSlash(rest))) // #nosec G304 -- checked-in suite tree
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	return jsonschema.UnmarshalJSON(file)
}

// TestOfficialJSONSchemaSuite runs the required draft2020-12 vocabulary from
// the JSON Schema organisation's own suite against the validator this package
// uses, with the same compiler settings the gate uses. There is no skip list:
// every vendored required file must pass.
func TestOfficialJSONSchemaSuite(t *testing.T) {
	t.Parallel()

	files := requiredSuiteFiles(t)
	if len(files) < 40 {
		t.Fatalf("requiredSuiteFiles() returned %d files, want the full draft2020-12 required suite", len(files))
	}

	for _, rel := range files {
		t.Run(rel, func(t *testing.T) {
			t.Parallel()
			for _, failure := range runSuiteFile(t, rel) {
				t.Errorf("%s", failure)
			}
		})
	}
	t.Logf("official JSON Schema draft2020-12 required suite: %d files, 0 skipped", len(files))
}

// TestOptionalFormatAssertionSupport executes the optional format-assertion
// cases and compares the result, case by case, against the checked-in record.
// The gate does not enable format assertion, so these divergences do not
// weaken it; pinning them keeps the claim "we know exactly what this validator
// does not implement" true rather than aspirational.
func TestOptionalFormatAssertionSupport(t *testing.T) {
	t.Parallel()

	want := loadFormatSupport(t)
	files := optionalFormatFiles(t)
	if len(files) == 0 {
		t.Fatal("optionalFormatFiles() returned nothing, want the vendored optional/format suite")
	}

	got := make(map[string]formatFileStat, len(files))
	for _, rel := range files {
		cases, failures := runOptionalFormatFile(t, rel)
		stat := formatFileStat{Cases: cases, Unsupported: failures}
		if stat.Unsupported == nil {
			stat.Unsupported = []string{}
		}
		got[rel] = stat
	}

	for rel, gotStat := range got {
		wantStat, ok := want.Files[rel]
		if !ok {
			t.Errorf("%s: no entry in %s; add one recording %d unsupported case(s)",
				rel, formatSupportRecord, len(gotStat.Unsupported))
			continue
		}
		if gotStat.Cases != wantStat.Cases {
			t.Errorf("%s: ran %d cases, record says %d; the vendored suite changed",
				rel, gotStat.Cases, wantStat.Cases)
		}
		if !slices.Equal(gotStat.Unsupported, wantStat.Unsupported) {
			t.Errorf("%s: unsupported cases changed\n got: %v\nwant: %v", rel, gotStat.Unsupported, wantStat.Unsupported)
		}
	}
	for rel := range want.Files {
		if _, ok := got[rel]; !ok {
			t.Errorf("%s: %s records a file that is no longer vendored", rel, formatSupportRecord)
		}
	}
}

// requiredSuiteFiles lists the required-vocabulary suite files, relative to the
// tests/draft2020-12 root.
func requiredSuiteFiles(t *testing.T) []string {
	t.Helper()
	return suiteFiles(t, func(rel string) bool { return !strings.HasPrefix(rel, "optional/") })
}

// optionalFormatFiles lists the vendored optional format-assertion suite files.
func optionalFormatFiles(t *testing.T) []string {
	t.Helper()
	return suiteFiles(t, func(rel string) bool { return strings.HasPrefix(rel, optionalFormatDir) })
}

func suiteFiles(t *testing.T, keep func(string) bool) []string {
	t.Helper()

	root := filepath.Join(suiteRoot, "tests", "draft2020-12")
	var out []string
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || filepath.Ext(p) != ".json" {
			return nil
		}
		rel, err := filepath.Rel(root, p)
		if err != nil {
			return err
		}
		if slash := filepath.ToSlash(rel); keep(slash) {
			out = append(out, slash)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("WalkDir(%q) error = %v", root, err)
	}
	slices.Sort(out)
	return out
}

// runSuiteFile executes one required-vocabulary suite file and returns a
// human-readable failure for every case whose outcome disagrees with the suite.
func runSuiteFile(t *testing.T, rel string) []string {
	t.Helper()

	var failures []string
	forEachSuiteCase(t, rel, false, func(group, name string, ok bool, err error) {
		if !ok {
			failures = append(failures, fmt.Sprintf("%s / %s / %s: %v", rel, group, name, err))
		}
	})
	return failures
}

// runOptionalFormatFile executes one optional format file with format assertion
// enabled and returns the case count plus the sorted descriptions of the cases
// the validator does not implement.
func runOptionalFormatFile(t *testing.T, rel string) (int, []string) {
	t.Helper()

	count := 0
	var unsupported []string
	forEachSuiteCase(t, rel, true, func(group, name string, ok bool, _ error) {
		count++
		if !ok {
			unsupported = append(unsupported, group+" / "+name)
		}
	})
	slices.Sort(unsupported)
	return count, unsupported
}

// forEachSuiteCase compiles every group in a suite file and reports each case's
// agreement with the suite's expectation. Compilation failures are reported as
// a single failed pseudo-case so they can never be mistaken for a pass.
func forEachSuiteCase(t *testing.T, rel string, assertFormat bool, report func(group, name string, ok bool, err error)) {
	t.Helper()

	full := filepath.Join(suiteRoot, "tests", "draft2020-12", filepath.FromSlash(rel))
	raw, err := os.ReadFile(full) // #nosec G304 -- fixed, checked-in suite path
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", full, err)
	}
	var groups []suiteCase
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.UseNumber()
	if err := dec.Decode(&groups); err != nil {
		t.Fatalf("decode %q error = %v", full, err)
	}

	for _, group := range groups {
		compiler := jsonschema.NewCompiler()
		compiler.DefaultDraft(jsonschema.Draft2020)
		compiler.UseLoader(suiteRemoteLoader(suiteRoot))
		if assertFormat {
			compiler.AssertFormat()
		}

		const caseURL = "https://conformance.looprig.test/suite.json"
		if err := compiler.AddResource(caseURL, group.Schema); err != nil {
			report(group.Description, "<compile>", false, fmt.Errorf("AddResource: %w", err))
			continue
		}
		schema, err := compiler.Compile(caseURL)
		if err != nil {
			report(group.Description, "<compile>", false, fmt.Errorf("Compile: %w", err))
			continue
		}
		for _, tc := range group.Tests {
			err := schema.Validate(tc.Data)
			var verr *jsonschema.ValidationError
			if err != nil && !errors.As(err, &verr) {
				report(group.Description, tc.Description, false, fmt.Errorf("non-validation error: %w", err))
				continue
			}
			got := err == nil
			report(group.Description, tc.Description, got == tc.Valid,
				fmt.Errorf("valid = %v, want %v (%v)", got, tc.Valid, err))
		}
	}
}

func loadFormatSupport(t *testing.T) formatSupport {
	t.Helper()

	raw, err := os.ReadFile(formatSupportRecord)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", formatSupportRecord, err)
	}
	var out formatSupport
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal %q error = %v", formatSupportRecord, err)
	}
	return out
}
