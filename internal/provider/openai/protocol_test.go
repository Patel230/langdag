package openai

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
	result := convertTools(tools, openAIServerTools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].Type != "function" {
		t.Errorf("type = %q, want %q", result[0].Type, "function")
	}
	if result[0].Function == nil {
		t.Fatal("expected Function to be set")
	}
	if result[0].Function.Name != "get_weather" {
		t.Errorf("function name = %q, want %q", result[0].Function.Name, "get_weather")
	}
}

func TestConvertTools_ServerToolWebSearch(t *testing.T) {
	tools := []types.ToolDefinition{
		{Name: types.ServerToolWebSearch},
	}
	result := convertTools(tools, openAIServerTools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].Type != "web_search_preview" {
		t.Errorf("type = %q, want %q", result[0].Type, "web_search_preview")
	}
	if result[0].Function != nil {
		t.Error("expected Function to be nil for server tool")
	}

	// Verify JSON serialization omits the function field
	b, err := json.Marshal(result[0])
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	var m map[string]interface{}
	json.Unmarshal(b, &m)
	if _, ok := m["function"]; ok {
		t.Errorf("expected 'function' key to be absent in JSON, got: %s", string(b))
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
	}
	result := convertTools(tools, openAIServerTools)
	if len(result) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(result))
	}
	if result[0].Type != "function" || result[0].Function == nil {
		t.Error("expected first tool to be a function tool")
	}
	if result[1].Type != "web_search_preview" || result[1].Function != nil {
		t.Error("expected second tool to be web_search_preview server tool")
	}
}

func TestConvertTools_UnknownServerToolPassedThrough(t *testing.T) {
	tools := []types.ToolDefinition{
		{Name: "code_execution"},
	}
	result := convertTools(tools, openAIServerTools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].Type != "code_execution" {
		t.Errorf("type = %q, want %q", result[0].Type, "code_execution")
	}
	if result[0].Function != nil {
		t.Error("expected Function to be nil for unknown server tool")
	}
}

func TestConvertTools_WebSearchWithSchemaIsClientTool(t *testing.T) {
	tools := []types.ToolDefinition{
		{
			Name:        "web_search",
			Description: "Custom web search",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
		},
	}
	result := convertTools(tools, openAIServerTools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].Type != "function" {
		t.Errorf("type = %q, want %q", result[0].Type, "function")
	}
	if result[0].Function == nil {
		t.Fatal("expected Function to be set for client tool")
	}
	if result[0].Function.Name != "web_search" {
		t.Errorf("function name = %q, want %q", result[0].Function.Name, "web_search")
	}
}

func TestConvertTools_GrokMapping(t *testing.T) {
	tools := []types.ToolDefinition{
		{Name: types.ServerToolWebSearch},
	}
	result := convertTools(tools, grokServerTools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	// Grok has no mapping for web_search, so it passes through as-is
	if result[0].Type != "web_search" {
		t.Errorf("type = %q, want %q", result[0].Type, "web_search")
	}
	if result[0].Function != nil {
		t.Error("expected Function to be nil for server tool")
	}
}
