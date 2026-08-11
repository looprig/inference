package examples_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type docsManifest struct {
	SchemaVersion int           `json:"schemaVersion"`
	Repository    string        `json:"repository"`
	ProofSources  []proofSource `json:"proofSources"`
	Examples      []example     `json:"examples"`
}

type proofSource struct {
	ID     string `json:"id"`
	Type   string `json:"type"`
	Path   string `json:"path"`
	Symbol string `json:"symbol"`
}

type example struct {
	ID             string            `json:"id"`
	Ecosystem      string            `json:"ecosystem"`
	Owner          string            `json:"owner"`
	SourcePath     string            `json:"sourcePath"`
	Availability   string            `json:"availability"`
	Versions       map[string]string `json:"versions"`
	OfflineCommand string            `json:"offlineCommand"`
	Assertion      string            `json:"assertion"`
	WorkflowPath   string            `json:"workflowPath"`
	JobID          string            `json:"jobId"`
	Cleanup        string            `json:"cleanup"`
	LiveGate       *string           `json:"liveGate"`
	ProofIDs       []string          `json:"proofIds"`
}

func TestDocsArtifacts(t *testing.T) {
	root := repositoryRoot(t)
	manifestPath := filepath.Join(root, "testdata/docs/examples.json")
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	var manifest docsManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	if manifest.SchemaVersion != 1 || manifest.Repository != "inference" {
		t.Fatalf("manifest identity = version %d repository %q", manifest.SchemaVersion, manifest.Repository)
	}

	proofs := make(map[string]proofSource, len(manifest.ProofSources))
	for _, proof := range manifest.ProofSources {
		if !strings.HasPrefix(proof.ID, "example-inference-") {
			t.Errorf("proof ID %q lacks repository prefix", proof.ID)
		}
		if proof.Type != "source" && proof.Type != "test" && proof.Type != "executable-fixture" {
			t.Errorf("proof %q has invalid type %q", proof.ID, proof.Type)
		}
		if _, err := os.Stat(filepath.Join(root, proof.Path)); err != nil {
			t.Errorf("proof %q path: %v", proof.ID, err)
		}
		if _, duplicate := proofs[proof.ID]; duplicate {
			t.Errorf("duplicate proof ID %q", proof.ID)
		}
		proofs[proof.ID] = proof
	}

	wantPaths := map[string]bool{
		"examples/invoke/main.go":  false,
		"examples/stream/main.go":  false,
		"examples/retry/main.go":   false,
		"examples/gateway/main.go": false,
	}
	workflow, err := os.ReadFile(filepath.Join(root, ".github/workflows/docs-examples.yml"))
	if err != nil {
		t.Fatalf("read workflow: %v", err)
	}
	for _, item := range manifest.Examples {
		if item.Owner != "inference" || item.Ecosystem != "go" || item.Availability != "source-workspace" {
			t.Errorf("example %q has invalid ownership or availability", item.ID)
		}
		if item.Versions["github.com/looprig/inference"] != "source-workspace" {
			t.Errorf("example %q lacks source-workspace module version", item.ID)
		}
		if _, ok := wantPaths[item.SourcePath]; !ok {
			t.Errorf("example %q has unexpected source path %q", item.ID, item.SourcePath)
		} else {
			wantPaths[item.SourcePath] = true
		}
		if item.Assertion == "" || item.Cleanup == "" || item.LiveGate != nil {
			t.Errorf("example %q lacks deterministic execution metadata", item.ID)
		}
		if item.WorkflowPath != ".github/workflows/docs-examples.yml" || item.JobID != "docs-examples" {
			t.Errorf("example %q has invalid workflow reference", item.ID)
		}
		if !strings.Contains(string(workflow), "run: "+item.OfflineCommand) {
			t.Errorf("workflow does not literally run %q", item.OfflineCommand)
		}
		for _, proofID := range item.ProofIDs {
			if _, ok := proofs[proofID]; !ok {
				t.Errorf("example %q references unknown proof %q", item.ID, proofID)
			}
		}
	}
	for path, found := range wantPaths {
		if !found {
			t.Errorf("manifest does not register %q", path)
		}
	}
	if !strings.Contains(string(workflow), "GOWORK=off GOCACHE=/tmp/looprig-inference-docs-gocache make test") {
		t.Error("workflow does not run the native test command")
	}
	if !strings.Contains(string(workflow), "GOWORK=off GOCACHE=/tmp/looprig-inference-docs-gocache go test -race ./...") {
		t.Error("workflow does not run the standalone race command")
	}
}
