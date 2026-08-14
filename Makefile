.PHONY: test fmt fmt-check lint vuln verify secure fuzz conformance-schemas

# Module's own package dirs, excluding vendor/ and the nested .worktrees/ modules
# (go list ./... stops at nested module boundaries and skips vendor).
GO_DIRS := $(shell go list -f '{{.Dir}}' ./...)

# inference does not vendor, so there is no -mod=vendor export here.
# Verification runs GOWORK=off so the module proves it resolves through its own
# pinned dependency graph.

test:
	go test -race ./...

# Format the whole module in place.
fmt:
	gofmt -w $(GO_DIRS)

# Fail (non-zero exit) if any tracked Go file is not gofmt-clean. Wired into lint.
fmt-check:
	@unformatted=$$(gofmt -l $(GO_DIRS)); \
	if [ -n "$$unformatted" ]; then \
		echo "gofmt needed (run 'make fmt'):"; echo "$$unformatted"; exit 1; \
	fi

lint: fmt-check
	go vet ./...
	go tool staticcheck ./...
	# gosec is NOT module-aware: its ./... is a filesystem walk that descends into
	# the nested .worktrees/ checkouts (separate modules) and, under -mod=vendor,
	# reports modules.txt desyncs for those foreign trees. Scope it to THIS module's
	# package dirs via GO_DIRS (the same go-list idiom fmt/fmt-check use). go vet and
	# staticcheck are module-aware (go list stops at module boundaries), so they need
	# no scoping.
	go tool gosec $(GO_DIRS)

vuln:
	go mod verify
	go tool govulncheck ./...

secure: lint vuln

fuzz:
	@echo "Usage: go test -fuzz=FuzzXxx ./path/to/pkg -fuzztime=30s"

# Re-derive the provider conformance schemas from the upstream API descriptions.
# This is the ONLY supported way to change codec/conformance/schema: the tree,
# its provenance record and its unenforced-constraint report are all generated
# together, so a hand edit would desynchronise them. The command downloads each
# source from the URL recorded in provenance.json, so upstream drift lands as a
# reviewable git diff in the hashes and the schemas.
CONFORMANCE_SPECS ?= /tmp/looprig-inference-conformance-specs
conformance-schemas:
	go run ./codec/conformance/cmd/schemagen \
		-fetch \
		-specs $(CONFORMANCE_SPECS) \
		-out codec/conformance/schema
