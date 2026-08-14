package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"github.com/looprig/inference/codec/conformance"
)

// The conformance gate delegates every assertion to a third-party JSON Schema
// validator, so the package checks that validator against the JSON Schema
// organisation's own test suite on every run. The suite is vendored rather than
// fetched, which means it needs the same chain of custody as the API
// descriptions: this file produces its provenance entry.

// suiteProvenanceKey is the key the suite's record appears under.
const suiteProvenanceKey = "jsonschema-test-suite"

// vendoredSuite pins the checked-in copy of the suite. The commit is a
// deliberate pin: re-vendoring is a decision, not a side effect of running the
// generator, so the revision is updated here by hand at the same time as the
// files under testdata/jsonschema-suite.
var vendoredSuite = struct {
	url       string
	commit    string
	publisher string
	subtree   string
}{
	url:       "https://github.com/json-schema-org/JSON-Schema-Test-Suite",
	commit:    "6648e8194c69697b2e1a15fe76a06a480b183a51",
	publisher: "The JSON Schema organisation, from its own JSON-Schema-Test-Suite repository",
	subtree: "tests/draft2020-12/*.json, tests/draft2020-12/optional/format/*.json and " +
		"remotes/draft2020-12/** only; the rest of the suite covers drafts and optional " +
		"vocabularies the gate does not use.",
}

// suiteRecord hashes the vendored suite tree so a modification to any of its
// files is detectable. A per-tree manifest hash is used rather than a single
// file hash because the suite is a directory, and because it makes an
// accidental edit to one case file as visible as a wholesale replacement.
func suiteRecord(dir string, previous *conformance.Provenance, today string) (*conformance.Source, error) {
	info, err := os.Stat(dir)
	if err != nil {
		return nil, fmt.Errorf("vendored suite: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("vendored suite: %s is not a directory", dir)
	}

	var (
		manifest []string
		files    int
		bytes    int
	)
	err = filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		// #nosec G304,G122 -- the tree is a checked-in, repository-local fixture
		// directory named on the command line by the operator regenerating it.
		body, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return err
		}
		manifest = append(manifest, filepath.ToSlash(rel)+"\x00"+hashHex(body))
		files++
		bytes += len(body)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("vendored suite: %w", err)
	}
	if files == 0 {
		return nil, fmt.Errorf("vendored suite: %s is empty", dir)
	}
	sort.Strings(manifest)

	sum := sha256.New()
	for _, line := range manifest {
		_, _ = sum.Write([]byte(line))
		_, _ = sum.Write([]byte("\n"))
	}
	tree := hex.EncodeToString(sum.Sum(nil))

	rec := &conformance.Source{
		URL:       vendoredSuite.url,
		File:      "testdata/jsonschema-suite",
		Dialect:   "json-schema-test-suite",
		Publisher: vendoredSuite.publisher,
		Hosting: "Vendored into testdata rather than converted. It is not an API description: " +
			"it is the evidence that the third-party validator the gate depends on implements " +
			"draft 2020-12 correctly. suite_test.go runs it on every `go test`.",
		Bytes:  bytes,
		SHA256: tree,
		CanonicalNote: "sha256 is a manifest hash over the vendored tree: the SHA-256 of every file, " +
			"keyed by its path and sorted, so any edit to any case file changes it.",
		Fields: map[string]string{
			"commit":  vendoredSuite.commit,
			"files":   strconv.Itoa(files),
			"subtree": vendoredSuite.subtree,
		},
	}
	rec.Retrieved = retrievedDate(previous, suiteProvenanceKey, tree, today)
	return rec, nil
}
