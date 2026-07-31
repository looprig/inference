package gateway_test

// This file is Task 14's concurrency stress coverage: proving a single
// gateway.Handler (and the Mux it resolves through) is safe for concurrent
// use across independent requests, per the package's own documented
// contract ("Handler ... is safe for concurrent use by multiple goroutines,
// as any http.Handler must be" -- handler.go). Every test here reuses the
// dialect adapter table and fake-target machinery from
// matrix_fixtures_test.go; run with -race per this task's own verification
// step.
import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/gateway"
	"github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"
)

// TestConcurrent_DifferentTargets_NoCrossTalk proves concurrent requests to
// TWO DIFFERENT targets, through one shared Handler, never swap responses:
// every request addressed to alias-a gets target A's canned text and every
// request addressed to alias-b gets target B's, even when fired
// concurrently and in high volume.
func TestConcurrent_DifferentTargets_NoCrossTalk(t *testing.T) {
	t.Parallel()
	ingress := matrixDialects[model.APIFormatAnthropic]

	srvA, ftA := newFakeTarget(t, matrixDialects[model.APIFormatAnthropic].codec)
	ftA.setResponse(textResponse("response-from-target-A"))
	targetA := buildMatrixTarget(t, matrixDialects[model.APIFormatAnthropic], srvA, "model-a", broadCaps()...)

	srvB, ftB := newFakeTarget(t, matrixDialects[model.APIFormatGemini].codec)
	ftB.setResponse(textResponse("response-from-target-B"))
	targetB := buildMatrixTarget(t, matrixDialects[model.APIFormatGemini], srvB, "model-b", broadCaps()...)

	h := buildMultiRouteHandler(t, ingress, map[string]gateway.Target{
		"alias-a": targetA,
		"alias-b": targetB,
	})

	const n = 100
	var wg sync.WaitGroup
	errs := make(chan string, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			alias, want := "alias-a", "response-from-target-A"
			if i%2 == 1 {
				alias, want = "alias-b", "response-from-target-B"
			}
			rr, resp := sendIngressInvoke(t, h, ingress, alias, textRequest(fmt.Sprintf("q-%d", i)), false)
			if rr.Code != http.StatusOK {
				errs <- fmt.Sprintf("iter %d (alias %s): status = %d, body = %s", i, alias, rr.Code, rr.Body.String())
				return
			}
			got := allText(resp.Message.Blocks)
			if !strings.Contains(got, want) {
				errs <- fmt.Sprintf("iter %d (alias %s): response text = %q, want it to contain %q (cross-talk between targets?)", i, alias, got, want)
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}
	if got := ftA.callCount(); got != n/2 {
		t.Errorf("target A received %d requests, want %d", got, n/2)
	}
	if got := ftB.callCount(); got != n/2 {
		t.Errorf("target B received %d requests, want %d", got, n/2)
	}
}

// TestConcurrent_SameTarget_NoCrossTalk proves concurrent requests hitting
// the SAME target never observe each other's data: the fake target echoes
// each request's own unique text back, so a crossed response is directly
// detectable (a goroutine seeing another goroutine's echoed text).
func TestConcurrent_SameTarget_NoCrossTalk(t *testing.T) {
	t.Parallel()
	ingress := matrixDialects[model.APIFormatOpenAI]
	targetD := matrixDialects[model.APIFormatOpenAIResponses]

	srv, ft := newFakeTarget(t, targetD.codec)
	ft.setResponseFunc(func(decoded codec.DecodedRequest) (*inference.Response, error) {
		// Echo the caller's own text back, uniquely identifying which
		// request this response belongs to.
		var last string
		for _, m := range decoded.Request.Messages {
			if um, ok := m.(*content.UserMessage); ok {
				last = allText(um.Blocks)
			}
		}
		return textResponse("echo:" + last), nil
	})
	target := buildMatrixTarget(t, targetD, srv, "shared-model", broadCaps()...)
	h := buildMultiRouteHandler(t, ingress, map[string]gateway.Target{matrixAlias: target})

	const n = 200
	var wg sync.WaitGroup
	errs := make(chan string, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			unique := "unique-payload-" + strconv.Itoa(i)
			rr, resp := sendIngressInvoke(t, h, ingress, matrixAlias, textRequest(unique), false)
			if rr.Code != http.StatusOK {
				errs <- fmt.Sprintf("iter %d: status = %d, body = %s", i, rr.Code, rr.Body.String())
				return
			}
			got := allText(resp.Message.Blocks)
			want := "echo:" + unique
			if got != want {
				errs <- fmt.Sprintf("iter %d: response text = %q, want %q (cross-talk on shared target?)", i, got, want)
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}
	if got := ft.callCount(); got != n {
		t.Errorf("target received %d requests, want %d", got, n)
	}
}

// TestConcurrent_MixedStreamingAndNonStreaming proves a Handler correctly
// serves a mix of concurrent streaming and non-streaming requests to the
// same target without either kind corrupting the other.
func TestConcurrent_MixedStreamingAndNonStreaming(t *testing.T) {
	t.Parallel()
	ingress := matrixDialects[model.APIFormatAnthropic]
	targetD := matrixDialects[model.APIFormatAnthropic]

	srv, ft := newFakeTarget(t, targetD.codec)
	ft.setResponse(textResponse("non-streaming-reply"))
	ft.setStreamResponse(
		[]content.Chunk{&content.TextChunk{Text: "streamed-"}, &content.TextChunk{Text: "reply"}},
		stream.StreamResult{FinishReason: stream.FinishReasonStop},
	)
	target := buildMatrixTarget(t, targetD, srv, "mixed-model", broadCaps()...)
	h := buildMultiRouteHandler(t, ingress, map[string]gateway.Target{matrixAlias: target})

	const n = 100
	var wg sync.WaitGroup
	errs := make(chan string, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			streaming := i%2 == 0
			rr, resp := sendIngressInvoke(t, h, ingress, matrixAlias, textRequest(fmt.Sprintf("q-%d", i)), streaming)
			if rr.Code != http.StatusOK {
				errs <- fmt.Sprintf("iter %d (streaming=%v): status = %d, body = %s", i, streaming, rr.Code, rr.Body.String())
				return
			}
			if streaming {
				if !strings.Contains(rr.Body.String(), "streamed-") || !strings.Contains(rr.Body.String(), "reply") {
					errs <- fmt.Sprintf("iter %d (streaming): body = %q, want it to contain the streamed reply text", i, rr.Body.String())
				}
				return
			}
			got := allText(resp.Message.Blocks)
			if got != "non-streaming-reply" {
				errs <- fmt.Sprintf("iter %d (non-streaming): response text = %q, want %q", i, got, "non-streaming-reply")
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}
}

// TestConcurrent_ManyGoroutines_MuxResolutionSafe proves route resolution
// itself is safe under concurrent load: many goroutines resolve across
// SEVERAL distinct aliases (each mapped to a distinct target dialect) on one
// shared Handler/Mux, and each always reaches the target its alias names.
func TestConcurrent_ManyGoroutines_MuxResolutionSafe(t *testing.T) {
	t.Parallel()
	ingress := matrixDialects[model.APIFormatOpenAI]

	type route struct {
		alias  string
		target gateway.Target
		want   string
		ft     *fakeTarget
	}
	var routes []route
	for _, format := range matrixFormats {
		srv, ft := newFakeTarget(t, matrixDialects[format].codec)
		want := "reply-from-" + string(format)
		ft.setResponse(textResponse(want))
		target := buildMatrixTarget(t, matrixDialects[format], srv, "model-"+string(format), broadCaps()...)
		routes = append(routes, route{alias: "alias-" + string(format), target: target, want: want, ft: ft})
	}

	routeMap := make(map[string]gateway.Target, len(routes))
	for _, r := range routes {
		routeMap[r.alias] = r.target
	}
	h := buildMultiRouteHandler(t, ingress, routeMap)

	const perRoute = 50
	var wg sync.WaitGroup
	errs := make(chan string, perRoute*len(routes))
	for _, r := range routes {
		r := r
		for i := 0; i < perRoute; i++ {
			wg.Add(1)
			go func(i int) {
				defer wg.Done()
				rr, resp := sendIngressInvoke(t, h, ingress, r.alias, textRequest(fmt.Sprintf("q-%d", i)), false)
				if rr.Code != http.StatusOK {
					errs <- fmt.Sprintf("alias %s iter %d: status = %d, body = %s", r.alias, i, rr.Code, rr.Body.String())
					return
				}
				got := allText(resp.Message.Blocks)
				if got != r.want {
					errs <- fmt.Sprintf("alias %s iter %d: response text = %q, want %q (mis-resolved route under concurrent load?)", r.alias, i, got, r.want)
				}
			}(i)
		}
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}
	for _, r := range routes {
		if got := r.ft.callCount(); got != perRoute {
			t.Errorf("target for %s received %d requests, want %d", r.alias, got, perRoute)
		}
	}
}
