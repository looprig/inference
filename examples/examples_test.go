package examples_test

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestRunnableExamples(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		path string
		want string
	}{
		{name: "invoke", path: "./examples/invoke", want: "model: demo-model\nreply: hello from inference"},
		{name: "stream", path: "./examples/stream", want: "chunks: hello stream\nfinish: stop"},
		{name: "retry", path: "./examples/retry", want: "attempts: 3\nreply: recovered"},
		{name: "gateway", path: "./examples/gateway", want: "requested: primary\nupstream: provider-model"},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			cmd := exec.Command("go", "run", tt.path)
			cmd.Dir = repositoryRoot(t)
			cmd.Env = append(cmd.Environ(), "GOWORK=off")
			output, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("go run %s: %v\n%s", tt.path, err, output)
			}
			if got := strings.TrimSpace(string(output)); got != tt.want {
				t.Fatalf("output = %q, want %q", got, tt.want)
			}
		})
	}
}

func repositoryRoot(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller did not report this test file")
	}
	return filepath.Dir(filepath.Dir(file))
}
