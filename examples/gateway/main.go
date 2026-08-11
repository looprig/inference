package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"

	"github.com/looprig/core/content"
	"github.com/looprig/inference"
	"github.com/looprig/inference/codec"
	"github.com/looprig/inference/codec/anthropicapi"
	"github.com/looprig/inference/gateway"
	"github.com/looprig/inference/model"
	"github.com/looprig/inference/stream"
)

type client struct{ upstreamModel string }

func (c *client) Invoke(_ context.Context, req inference.Request) (*inference.Response, error) {
	c.upstreamModel = req.Model.Name
	return &inference.Response{
		Message: &content.AIMessage{Message: content.Message{
			Role:   content.RoleAssistant,
			Blocks: []content.Block{&content.TextBlock{Text: "routed"}},
		}},
		Model:        req.Model.Name,
		FinishReason: stream.FinishReasonStop,
	}, nil
}

func (*client) Stream(context.Context, inference.Request) (*stream.StreamReader[content.Chunk], error) {
	return nil, errors.New("streaming is not used in this example")
}

func main() {
	upstream := &client{}
	providerModel := model.CustomModel("example", model.APIFormatAnthropic, "", "provider-model")
	resolver, err := gateway.NewMux(gateway.Mux{Routes: map[gateway.RouteKey]gateway.Target{
		{Ingress: model.APIFormatAnthropic, Model: "primary"}: {
			ID: "primary-target", Client: upstream, Model: providerModel,
		},
	}})
	if err != nil {
		panic(err)
	}
	handler, err := gateway.New(gateway.Config{
		Resolver: resolver,
		Codecs: map[model.APIFormat]codec.ServerCodec{
			model.APIFormatAnthropic: anthropicapi.Codec{},
		},
		Authenticate: gateway.StaticToken("local-token"),
	})
	if err != nil {
		panic(err)
	}

	body := `{"model":"primary","max_tokens":16,"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`
	request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer local-token")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		panic(recorder.Body.String())
	}
	var response struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		panic(err)
	}
	fmt.Printf("requested: %s\nupstream: %s\n", response.Model, upstream.upstreamModel)
}
