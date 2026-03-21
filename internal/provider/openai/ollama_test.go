package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"langdag.com/langdag/types"
)

func TestNewOllama_DefaultURL(t *testing.T) {
	p := NewOllama("")
	if p.baseURL != "http://localhost:11434" {
		t.Errorf("expected default URL, got %q", p.baseURL)
	}
}

func TestNewOllama_TrailingSlashStripped(t *testing.T) {
	p := NewOllama("http://localhost:11434/")
	if p.baseURL != "http://localhost:11434" {
		t.Errorf("expected trailing slash stripped, got %q", p.baseURL)
	}
}

func TestNewOllama_CustomURL(t *testing.T) {
	p := NewOllama("http://192.168.1.10:11434")
	if p.baseURL != "http://192.168.1.10:11434" {
		t.Errorf("expected custom URL preserved, got %q", p.baseURL)
	}
}

func TestOllamaName(t *testing.T) {
	p := NewOllama("")
	if p.Name() != "ollama" {
		t.Errorf("expected name 'ollama', got '%s'", p.Name())
	}
}

func TestOllamaModels_Empty(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models": []}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	models := p.Models()
	if models == nil {
		t.Fatal("expected empty slice, got nil")
	}
	if len(models) != 0 {
		t.Errorf("expected 0 models, got %d", len(models))
	}
}

func TestOllamaModels_TagsAPIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	if models := p.Models(); models != nil {
		t.Errorf("expected nil on /api/tags error, got %v", models)
	}
}

func TestOllamaModels_TagsInvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`not-json`))
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	if models := p.Models(); models != nil {
		t.Errorf("expected nil on invalid JSON from /api/tags, got %v", models)
	}
}

func TestOllamaModels_ModelIDAndNameSet(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models": [{"name": "llama3"}]}`))
			return
		}
		if r.URL.Path == "/api/show" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"model_info": {}}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	models := p.Models()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].ID != "llama3" {
		t.Errorf("ID = %q, want %q", models[0].ID, "llama3")
	}
	if models[0].Name != "llama3" {
		t.Errorf("Name = %q, want %q", models[0].Name, "llama3")
	}
}

func TestOllamaContextWindow_LlamaPrefix(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models": [{"name": "llama3"}]}`))
			return
		}
		if r.URL.Path == "/api/show" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"model_info": {"llama.context_length": 8192}}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	models := p.Models()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].ContextWindow != 8192 {
		t.Errorf("expected context window 8192, got %d", models[0].ContextWindow)
	}
}

func TestOllamaContextWindow_NonStandardPrefix(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models": [{"name": "qwen2.5:7b"}]}`))
			return
		}
		if r.URL.Path == "/api/show" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"model_info": {"qwen2.context_length": 32768}}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	models := p.Models()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].ContextWindow != 32768 {
		t.Errorf("expected context window 32768, got %d", models[0].ContextWindow)
	}
}

func TestOllamaContextWindow_MissingKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models": [{"name": "unknown-model"}]}`))
			return
		}
		if r.URL.Path == "/api/show" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"model_info": {}}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	models := p.Models()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].ContextWindow != 0 {
		t.Errorf("expected 0, got %d", models[0].ContextWindow)
	}
}

func TestOllamaContextWindow_WrongValueType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models": [{"name": "weird-model"}]}`))
			return
		}
		if r.URL.Path == "/api/show" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"model_info": {"llama.context_length": "not-a-number"}}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	models := p.Models()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].ContextWindow != 0 {
		t.Errorf("expected 0 for wrong value type, got %d", models[0].ContextWindow)
	}
}

func TestOllamaContextWindow_ShowError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models": [{"name": "llama3"}]}`))
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	models := p.Models()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].ContextWindow != 0 {
		t.Errorf("expected 0 on /api/show error, got %d", models[0].ContextWindow)
	}
}

func TestOllamaContextWindow_ShowInvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models": [{"name": "llama3"}]}`))
			return
		}
		if r.URL.Path == "/api/show" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`not-json`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	models := p.Models()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].ContextWindow != 0 {
		t.Errorf("expected 0 on invalid /api/show JSON, got %d", models[0].ContextWindow)
	}
}

func TestOllamaContextWindow_PerModelValues(t *testing.T) {
	modelCtx := map[string]int{"model-a": 8192, "model-b": 32768}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models": [{"name": "model-a"}, {"name": "model-b"}]}`))
			return
		}
		if r.URL.Path == "/api/show" {
			var req map[string]string
			json.NewDecoder(r.Body).Decode(&req)
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(fmt.Sprintf(`{"model_info": {"llama.context_length": %d}}`, modelCtx[req["name"]])))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	models := p.Models()
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
	got := map[string]int{}
	for _, m := range models {
		got[m.ID] = m.ContextWindow
	}
	for name, expected := range modelCtx {
		if got[name] != expected {
			t.Errorf("%s: expected context window %d, got %d", name, expected, got[name])
		}
	}
}

func TestOllamaContextWindow_CacheHit(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models": [{"name": "llama3"}]}`))
			return
		}
		if r.URL.Path == "/api/show" {
			callCount++
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"model_info": {"llama.context_length": 8192}}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	p.Models()
	p.Models()

	if callCount != 1 {
		t.Errorf("expected /api/show called once, got %d", callCount)
	}
}

func TestOllamaContextWindow_CacheReturnsSameValue(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models": [{"name": "llama3"}]}`))
			return
		}
		if r.URL.Path == "/api/show" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"model_info": {"llama.context_length": 16384}}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	first := p.Models()
	second := p.Models()

	if first[0].ContextWindow != 16384 || second[0].ContextWindow != 16384 {
		t.Errorf("cache returned wrong value: first=%d second=%d", first[0].ContextWindow, second[0].ContextWindow)
	}
}

func TestOllamaModels_ParallelFetch(t *testing.T) {
	modelCtx := map[string]int{
		"model-a": 4096,
		"model-b": 8192,
		"model-c": 32768,
		"model-d": 131072,
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models":[{"name":"model-a"},{"name":"model-b"},{"name":"model-c"},{"name":"model-d"}]}`))
			return
		}
		if r.URL.Path == "/api/show" {
			var req map[string]string
			json.NewDecoder(r.Body).Decode(&req)
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(fmt.Sprintf(`{"model_info":{"llama.context_length":%d}}`, modelCtx[req["name"]])))
			return
		}
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	models := p.Models()
	if len(models) != 4 {
		t.Fatalf("expected 4 models, got %d", len(models))
	}
	got := map[string]int{}
	for _, m := range models {
		got[m.ID] = m.ContextWindow
	}
	for name, expected := range modelCtx {
		if got[name] != expected {
			t.Errorf("%s: expected context window %d, got %d", name, expected, got[name])
		}
	}
}

func TestOllamaDoRequest_UsesCorrectEndpoint(t *testing.T) {
	var gotPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"r1","model":"llama3","choices":[]}`))
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	_, _ = p.Complete(context.Background(), &types.CompletionRequest{
		Model:    "llama3",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	})
	if gotPath != "/v1/chat/completions" {
		t.Errorf("expected /v1/chat/completions, got %s", gotPath)
	}
}

func TestOllamaDoRequest_NoAuthHeader(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"r1","model":"llama3","choices":[]}`))
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	_, _ = p.Complete(context.Background(), &types.CompletionRequest{
		Model:    "llama3",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	})
	if gotAuth != "" {
		t.Errorf("expected no Authorization header, got %q", gotAuth)
	}
}

func TestOllamaDoRequest_ContentTypeHeader(t *testing.T) {
	var gotCT string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotCT = r.Header.Get("Content-Type")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"r1","model":"llama3","choices":[]}`))
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	_, _ = p.Complete(context.Background(), &types.CompletionRequest{
		Model:    "llama3",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	})
	if gotCT != "application/json" {
		t.Errorf("expected Content-Type application/json, got %q", gotCT)
	}
}

func TestOllamaDoRequest_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte(`{"error":"model not loaded"}`))
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	_, err := p.Complete(context.Background(), &types.CompletionRequest{
		Model:    "llama3",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	})
	if err == nil {
		t.Error("expected error on non-200 response, got nil")
	}
}

func TestOllamaDoRequest_ErrorContainsStatusAndBody(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(`{"error":"invalid model"}`))
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	_, err := p.Complete(context.Background(), &types.CompletionRequest{
		Model:    "bad-model",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	})
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "400") {
		t.Errorf("error should contain status 400, got: %s", err.Error())
	}
	if !strings.Contains(err.Error(), "invalid model") {
		t.Errorf("error should contain response body, got: %s", err.Error())
	}
}

func TestOllamaDoRequest_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	p := NewOllama(server.URL)
	_, err := p.Complete(ctx, &types.CompletionRequest{
		Model:    "llama3",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	})
	if err == nil {
		t.Error("expected error on cancelled context, got nil")
	}
}

func TestOllamaComplete_RoundTrip(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-abc123","model":"llama3","choices":[{"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`))
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	resp, err := p.Complete(context.Background(), &types.CompletionRequest{
		Model:    "llama3",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"Hello"`)}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ID != "chatcmpl-abc123" {
		t.Errorf("ID = %q, want %q", resp.ID, "chatcmpl-abc123")
	}
	if resp.Model != "llama3" {
		t.Errorf("Model = %q, want %q", resp.Model, "llama3")
	}
	if resp.StopReason != "stop" {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, "stop")
	}
	if resp.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d, want 10", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 5 {
		t.Errorf("OutputTokens = %d, want 5", resp.Usage.OutputTokens)
	}
}

func TestOllamaComplete_InvalidJSONResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`not-json`))
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	_, err := p.Complete(context.Background(), &types.CompletionRequest{
		Model:    "llama3",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	})
	if err == nil {
		t.Error("expected error on invalid JSON response, got nil")
	}
}

func TestOllamaStream_ReceivesDeltaAndDone(t *testing.T) {
	sseData := `data: {"id":"1","model":"llama3","choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}

data: {"id":"1","model":"llama3","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte(sseData))
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	ch, err := p.Stream(context.Background(), &types.CompletionRequest{
		Model:    "llama3",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var collected []types.StreamEvent
	for e := range ch {
		collected = append(collected, e)
	}

	var text string
	for _, e := range collected {
		if e.Type == types.StreamEventDelta {
			text += e.Content
		}
	}
	if text != "Hi" {
		t.Errorf("expected delta content \"Hi\", got %q", text)
	}
}

func TestOllamaStream_DoRequestError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	ch, err := p.Stream(context.Background(), &types.CompletionRequest{
		Model:    "llama3",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	})
	if err == nil {
		t.Error("expected error when doRequest fails, got nil")
	}
	if ch != nil {
		t.Error("expected nil channel on error, got non-nil")
	}
}

func TestOllamaStream_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-r.Context().Done()
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	p := NewOllama(server.URL)
	_, err := p.Stream(ctx, &types.CompletionRequest{
		Model:    "llama3",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	})
	if err == nil {
		t.Error("expected error on cancelled context, got nil")
	}
}
