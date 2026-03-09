package gemini

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
	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if len(result[0].FunctionDeclarations) != 1 {
		t.Fatalf("expected 1 function declaration, got %d", len(result[0].FunctionDeclarations))
	}
	if result[0].serverToolName != "" {
		t.Error("expected serverToolName to be empty for function-only tools")
	}
	if result[0].FunctionDeclarations[0].Name != "get_weather" {
		t.Errorf("name = %q, want %q", result[0].FunctionDeclarations[0].Name, "get_weather")
	}
}

func TestConvertTools_ServerToolWebSearch(t *testing.T) {
	tools := []types.ToolDefinition{
		{Name: types.ServerToolWebSearch},
	}
	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].serverToolName != "google_search" {
		t.Fatalf("expected serverToolName = %q, got %q", "google_search", result[0].serverToolName)
	}
	if len(result[0].FunctionDeclarations) != 0 {
		t.Error("expected no function declarations for server-only tools")
	}

	// Verify JSON serialization produces {"google_search":{}}
	b, err := json.Marshal(result[0])
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	var m map[string]interface{}
	json.Unmarshal(b, &m)
	if _, ok := m["google_search"]; !ok {
		t.Errorf("expected 'google_search' key in JSON, got: %s", string(b))
	}
	if _, ok := m["function_declarations"]; ok {
		t.Errorf("expected 'function_declarations' to be absent in JSON, got: %s", string(b))
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
	result := convertTools(tools)
	// Should produce 2 entries: one with function declarations, one server tool
	if len(result) != 2 {
		t.Fatalf("expected 2 tool entries, got %d", len(result))
	}

	// First entry: function declarations grouped together
	if len(result[0].FunctionDeclarations) != 2 {
		t.Fatalf("expected 2 function declarations, got %d", len(result[0].FunctionDeclarations))
	}
	if result[0].FunctionDeclarations[0].Name != "get_weather" {
		t.Errorf("first function = %q, want %q", result[0].FunctionDeclarations[0].Name, "get_weather")
	}
	if result[0].FunctionDeclarations[1].Name != "calculator" {
		t.Errorf("second function = %q, want %q", result[0].FunctionDeclarations[1].Name, "calculator")
	}

	// Second entry: server tool
	if result[1].serverToolName != "google_search" {
		t.Errorf("expected serverToolName = %q, got %q", "google_search", result[1].serverToolName)
	}
}

func TestConvertTools_UnknownServerToolPassedThrough(t *testing.T) {
	// An unknown name WITHOUT InputSchema should be passed as a server tool
	tools := []types.ToolDefinition{
		{Name: "code_execution"},
	}
	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].serverToolName != "code_execution" {
		t.Errorf("expected serverToolName = %q, got %q", "code_execution", result[0].serverToolName)
	}
	if len(result[0].FunctionDeclarations) != 0 {
		t.Errorf("expected no function declarations, got %d", len(result[0].FunctionDeclarations))
	}

	// Verify JSON serialization produces {"code_execution":{}}
	b, err := json.Marshal(result[0])
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	var m map[string]interface{}
	json.Unmarshal(b, &m)
	if _, ok := m["code_execution"]; !ok {
		t.Errorf("expected 'code_execution' key in JSON, got: %s", string(b))
	}
	if _, ok := m["function_declarations"]; ok {
		t.Errorf("expected 'function_declarations' to be absent in JSON, got: %s", string(b))
	}
}

func TestConvertTools_WebSearchWithSchemaIsClientTool(t *testing.T) {
	// A tool named "web_search" WITH InputSchema should be treated as a
	// client-side function declaration, not the built-in server tool.
	tools := []types.ToolDefinition{
		{
			Name:        types.ServerToolWebSearch,
			Description: "Custom web search",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
		},
	}
	result := convertTools(tools)
	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if len(result[0].FunctionDeclarations) != 1 {
		t.Fatalf("expected 1 function declaration, got %d", len(result[0].FunctionDeclarations))
	}
	if result[0].serverToolName != "" {
		t.Errorf("expected serverToolName to be empty, got %q", result[0].serverToolName)
	}
	if result[0].FunctionDeclarations[0].Name != types.ServerToolWebSearch {
		t.Errorf("name = %q, want %q", result[0].FunctionDeclarations[0].Name, types.ServerToolWebSearch)
	}

	// Verify JSON serialization includes function_declarations, not google_search
	b, err := json.Marshal(result[0])
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	var m map[string]interface{}
	json.Unmarshal(b, &m)
	if _, ok := m["function_declarations"]; !ok {
		t.Errorf("expected 'function_declarations' key in JSON, got: %s", string(b))
	}
	if _, ok := m["google_search"]; ok {
		t.Errorf("expected 'google_search' to be absent in JSON, got: %s", string(b))
	}
}
