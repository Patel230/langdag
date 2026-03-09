package anthropic

import (
	"encoding/json"
	"testing"

	"langdag.com/langdag/types"
)

func TestConvertTools_FunctionOnly(t *testing.T) {
	tools := []types.ToolDefinition{
		{
			Name:        "get_weather",
			Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
		},
	}
	result, err := convertTools(tools)
	if err != nil {
		t.Fatalf("convertTools: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].OfTool == nil {
		t.Fatal("expected OfTool to be set for function tool")
	}
	if result[0].OfWebSearchTool20250305 != nil {
		t.Fatal("expected OfWebSearchTool20250305 to be nil for function tool")
	}
	if result[0].OfTool.Name != "get_weather" {
		t.Errorf("tool name = %q, want %q", result[0].OfTool.Name, "get_weather")
	}
}

func TestConvertTools_ServerToolWebSearch(t *testing.T) {
	tools := []types.ToolDefinition{
		{Name: types.ServerToolWebSearch},
	}
	result, err := convertTools(tools)
	if err != nil {
		t.Fatalf("convertTools: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].OfWebSearchTool20250305 == nil {
		t.Fatal("expected OfWebSearchTool20250305 to be set for web_search")
	}
	if result[0].OfTool != nil {
		t.Fatal("expected OfTool to be nil for server tool")
	}
}

func TestConvertTools_MixedFunctionAndServerTools(t *testing.T) {
	tools := []types.ToolDefinition{
		{
			Name:        "get_weather",
			Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
		},
		{Name: types.ServerToolWebSearch},
		{
			Name:        "calculator",
			Description: "Calculate",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"expr":{"type":"string"}}}`),
		},
	}
	result, err := convertTools(tools)
	if err != nil {
		t.Fatalf("convertTools: %v", err)
	}
	if len(result) != 3 {
		t.Fatalf("expected 3 tools, got %d", len(result))
	}

	// First: function tool
	if result[0].OfTool == nil || result[0].OfTool.Name != "get_weather" {
		t.Error("expected first tool to be get_weather function")
	}

	// Second: server tool
	if result[1].OfWebSearchTool20250305 == nil {
		t.Error("expected second tool to be web_search server tool")
	}

	// Third: function tool
	if result[2].OfTool == nil || result[2].OfTool.Name != "calculator" {
		t.Error("expected third tool to be calculator function")
	}
}

func TestConvertTools_UnknownServerToolErrors(t *testing.T) {
	// A tool without InputSchema and an unknown name should return an error
	tools := []types.ToolDefinition{
		{
			Name:        "custom_search",
			Description: "Custom search",
		},
	}
	_, err := convertTools(tools)
	if err == nil {
		t.Fatal("expected error for unknown server tool, got nil")
	}
	expected := `anthropic: unsupported server tool "custom_search"`
	if err.Error() != expected {
		t.Errorf("error = %q, want %q", err.Error(), expected)
	}
}

func TestConvertTools_WebSearchWithSchemaIsClientTool(t *testing.T) {
	// A tool named "web_search" but with InputSchema should be treated as a client (function) tool
	tools := []types.ToolDefinition{
		{
			Name:        "web_search",
			Description: "Custom web search override",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
		},
	}
	result, err := convertTools(tools)
	if err != nil {
		t.Fatalf("convertTools: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].OfTool == nil {
		t.Fatal("expected OfTool to be set for client tool with schema")
	}
	if result[0].OfWebSearchTool20250305 != nil {
		t.Fatal("expected OfWebSearchTool20250305 to be nil for client tool with schema")
	}
	if result[0].OfTool.Name != "web_search" {
		t.Errorf("tool name = %q, want %q", result[0].OfTool.Name, "web_search")
	}
}
