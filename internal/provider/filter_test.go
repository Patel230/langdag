package provider

import (
	"context"
	"encoding/json"
	"testing"

	"langdag.com/langdag/types"
)

// stubProvider is a minimal Provider for testing the filter wrapper.
type stubProvider struct {
	models []types.ModelInfo
	// lastReq captures the request passed to Complete for verification.
	lastReq *types.CompletionRequest
}

func (s *stubProvider) Name() string              { return "stub" }
func (s *stubProvider) Models() []types.ModelInfo  { return s.models }

func (s *stubProvider) Complete(_ context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	s.lastReq = req
	return &types.CompletionResponse{}, nil
}

func (s *stubProvider) Stream(_ context.Context, req *types.CompletionRequest) (<-chan types.StreamEvent, error) {
	s.lastReq = req
	ch := make(chan types.StreamEvent)
	close(ch)
	return ch, nil
}

func clientTool(name string) types.ToolDefinition {
	return types.ToolDefinition{
		Name:        name,
		Description: name,
		InputSchema: json.RawMessage(`{"type":"object"}`),
	}
}

func serverTool(name string) types.ToolDefinition {
	return types.ToolDefinition{
		Name:        name,
		Description: name,
	}
}

func TestFilterProvider_AllSupported(t *testing.T) {
	inner := &stubProvider{
		models: []types.ModelInfo{
			{ID: "model-a", ServerTools: []string{"web_search"}},
		},
	}
	fp := WithServerToolFilter(inner)

	req := &types.CompletionRequest{
		Model: "model-a",
		Tools: []types.ToolDefinition{serverTool("web_search"), clientTool("my_func")},
	}
	fp.Complete(context.Background(), req)

	if len(inner.lastReq.Tools) != 2 {
		t.Errorf("expected 2 tools, got %d", len(inner.lastReq.Tools))
	}
}

func TestFilterProvider_UnsupportedStripped(t *testing.T) {
	inner := &stubProvider{
		models: []types.ModelInfo{
			{ID: "model-a", ServerTools: []string{"web_search"}},
		},
	}
	fp := WithServerToolFilter(inner)

	req := &types.CompletionRequest{
		Model: "model-a",
		Tools: []types.ToolDefinition{
			serverTool("web_search"),
			serverTool("code_interpreter"), // unsupported
			clientTool("my_func"),
		},
	}
	fp.Complete(context.Background(), req)

	if len(inner.lastReq.Tools) != 2 {
		t.Errorf("expected 2 tools (web_search + my_func), got %d", len(inner.lastReq.Tools))
	}
	for _, tool := range inner.lastReq.Tools {
		if tool.Name == "code_interpreter" {
			t.Error("code_interpreter should have been stripped")
		}
	}
}

func TestFilterProvider_NoServerToolsSupported(t *testing.T) {
	inner := &stubProvider{
		models: []types.ModelInfo{
			{ID: "ollama-model"}, // no ServerTools
		},
	}
	fp := WithServerToolFilter(inner)

	req := &types.CompletionRequest{
		Model: "ollama-model",
		Tools: []types.ToolDefinition{
			serverTool("web_search"),
			clientTool("my_func"),
		},
	}
	fp.Complete(context.Background(), req)

	if len(inner.lastReq.Tools) != 1 {
		t.Errorf("expected 1 tool (my_func only), got %d", len(inner.lastReq.Tools))
	}
	if inner.lastReq.Tools[0].Name != "my_func" {
		t.Errorf("expected my_func, got %s", inner.lastReq.Tools[0].Name)
	}
}

func TestFilterProvider_NoToolsInRequest(t *testing.T) {
	inner := &stubProvider{
		models: []types.ModelInfo{
			{ID: "model-a", ServerTools: []string{"web_search"}},
		},
	}
	fp := WithServerToolFilter(inner)

	req := &types.CompletionRequest{Model: "model-a"}
	fp.Complete(context.Background(), req)

	if len(inner.lastReq.Tools) != 0 {
		t.Errorf("expected 0 tools, got %d", len(inner.lastReq.Tools))
	}
}

func TestFilterProvider_UnknownModel(t *testing.T) {
	inner := &stubProvider{
		models: []types.ModelInfo{
			{ID: "model-a", ServerTools: []string{"web_search"}},
		},
	}
	fp := WithServerToolFilter(inner)

	// Unknown model — all server tools should be stripped (safe default)
	req := &types.CompletionRequest{
		Model: "unknown-model",
		Tools: []types.ToolDefinition{
			serverTool("web_search"),
			clientTool("my_func"),
		},
	}
	fp.Complete(context.Background(), req)

	if len(inner.lastReq.Tools) != 1 {
		t.Errorf("expected 1 tool (my_func only), got %d", len(inner.lastReq.Tools))
	}
}

func TestFilterProvider_PerModelCapabilities(t *testing.T) {
	inner := &stubProvider{
		models: []types.ModelInfo{
			{ID: "model-with-search", ServerTools: []string{"web_search"}},
			{ID: "model-without-search"}, // no server tools
		},
	}
	fp := WithServerToolFilter(inner)

	tools := []types.ToolDefinition{serverTool("web_search"), clientTool("my_func")}

	// Model with search: web_search preserved
	req1 := &types.CompletionRequest{Model: "model-with-search", Tools: tools}
	fp.Complete(context.Background(), req1)
	if len(inner.lastReq.Tools) != 2 {
		t.Errorf("model-with-search: expected 2 tools, got %d", len(inner.lastReq.Tools))
	}

	// Model without search: web_search stripped
	req2 := &types.CompletionRequest{Model: "model-without-search", Tools: tools}
	fp.Complete(context.Background(), req2)
	if len(inner.lastReq.Tools) != 1 {
		t.Errorf("model-without-search: expected 1 tool, got %d", len(inner.lastReq.Tools))
	}
}

func TestFilterProvider_StreamFilters(t *testing.T) {
	inner := &stubProvider{
		models: []types.ModelInfo{
			{ID: "model-a"}, // no server tools
		},
	}
	fp := WithServerToolFilter(inner)

	req := &types.CompletionRequest{
		Model: "model-a",
		Tools: []types.ToolDefinition{serverTool("web_search")},
	}
	fp.Stream(context.Background(), req)

	if len(inner.lastReq.Tools) != 0 {
		t.Errorf("Stream: expected 0 tools, got %d", len(inner.lastReq.Tools))
	}
}

func TestFilterProvider_EmptyModelID(t *testing.T) {
	inner := &stubProvider{
		models: []types.ModelInfo{
			{ID: "model-a", ServerTools: []string{"web_search"}},
		},
	}
	fp := WithServerToolFilter(inner)

	// Empty model ID — not in catalog, all server tools stripped.
	req := &types.CompletionRequest{
		Model: "",
		Tools: []types.ToolDefinition{
			serverTool("web_search"),
			clientTool("my_func"),
		},
	}
	fp.Complete(context.Background(), req)

	if len(inner.lastReq.Tools) != 1 {
		t.Errorf("expected 1 tool (my_func only), got %d", len(inner.lastReq.Tools))
	}
	if inner.lastReq.Tools[0].Name != "my_func" {
		t.Errorf("expected my_func, got %s", inner.lastReq.Tools[0].Name)
	}
}

func TestFilterProvider_AllServerToolsUnknownModel(t *testing.T) {
	inner := &stubProvider{
		models: []types.ModelInfo{
			{ID: "model-a", ServerTools: []string{"web_search"}},
		},
	}
	fp := WithServerToolFilter(inner)

	// Unknown model + only server tools → all stripped, empty tools list.
	req := &types.CompletionRequest{
		Model: "unknown-model",
		Tools: []types.ToolDefinition{
			serverTool("web_search"),
			serverTool("code_interpreter"),
		},
	}
	fp.Complete(context.Background(), req)

	if len(inner.lastReq.Tools) != 0 {
		t.Errorf("expected 0 tools, got %d", len(inner.lastReq.Tools))
	}
}

func TestFilterProvider_MultipleModelsUnknownStripsAll(t *testing.T) {
	inner := &stubProvider{
		models: []types.ModelInfo{
			{ID: "model-a", ServerTools: []string{"web_search", "code_interpreter"}},
			{ID: "model-b", ServerTools: []string{"web_search"}},
		},
	}
	fp := WithServerToolFilter(inner)

	// Unknown model — doesn't inherit any model's capabilities.
	req := &types.CompletionRequest{
		Model: "model-c",
		Tools: []types.ToolDefinition{
			serverTool("web_search"),
			serverTool("code_interpreter"),
			clientTool("my_func"),
		},
	}
	fp.Complete(context.Background(), req)

	if len(inner.lastReq.Tools) != 1 {
		t.Errorf("expected 1 tool (my_func only), got %d", len(inner.lastReq.Tools))
	}
}

func TestFilterProvider_StreamUnknownModel(t *testing.T) {
	inner := &stubProvider{
		models: []types.ModelInfo{
			{ID: "model-a", ServerTools: []string{"web_search"}},
		},
	}
	fp := WithServerToolFilter(inner)

	// Stream path with unknown model — server tools stripped.
	req := &types.CompletionRequest{
		Model: "unknown-model",
		Tools: []types.ToolDefinition{
			serverTool("web_search"),
			clientTool("my_func"),
		},
	}
	fp.Stream(context.Background(), req)

	if len(inner.lastReq.Tools) != 1 {
		t.Errorf("Stream: expected 1 tool, got %d", len(inner.lastReq.Tools))
	}
	if inner.lastReq.Tools[0].Name != "my_func" {
		t.Errorf("Stream: expected my_func, got %s", inner.lastReq.Tools[0].Name)
	}
}

func TestFilterProvider_PassthroughMethods(t *testing.T) {
	inner := &stubProvider{
		models: []types.ModelInfo{
			{ID: "m1", ServerTools: []string{"web_search"}},
		},
	}
	fp := WithServerToolFilter(inner)

	if fp.Name() != "stub" {
		t.Errorf("Name() = %q, want %q", fp.Name(), "stub")
	}

	models := fp.Models()
	if len(models) != 1 || models[0].ID != "m1" {
		t.Errorf("Models() = %v, want [{ID:m1}]", models)
	}
}
