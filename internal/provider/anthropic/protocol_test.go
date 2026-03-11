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

func TestConvertMessages_ToolUseBlocks(t *testing.T) {
	// Simulate an assistant message with tool_use blocks (e.g. stored from a non-Anthropic provider)
	blocks := []types.ContentBlock{
		{Type: "text", Text: "Let me check the weather."},
		{
			Type:  "tool_use",
			ID:    "toolu_abc123",
			Name:  "get_weather",
			Input: json.RawMessage(`{"location":"Paris"}`),
		},
	}
	blocksJSON, _ := json.Marshal(blocks)

	messages := []types.Message{
		{Role: "assistant", Content: blocksJSON},
	}

	result, err := convertMessages(messages)
	if err != nil {
		t.Fatalf("convertMessages: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if len(result[0].Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(result[0].Content))
	}

	// First block: text
	if result[0].Content[0].OfText == nil {
		t.Fatal("expected first block to be text")
	}
	if result[0].Content[0].OfText.Text != "Let me check the weather." {
		t.Errorf("text = %q, want %q", result[0].Content[0].OfText.Text, "Let me check the weather.")
	}

	// Second block: tool_use
	if result[0].Content[1].OfToolUse == nil {
		t.Fatal("expected second block to be tool_use")
	}
	if result[0].Content[1].OfToolUse.ID != "toolu_abc123" {
		t.Errorf("tool_use ID = %q, want %q", result[0].Content[1].OfToolUse.ID, "toolu_abc123")
	}
	if result[0].Content[1].OfToolUse.Name != "get_weather" {
		t.Errorf("tool_use Name = %q, want %q", result[0].Content[1].OfToolUse.Name, "get_weather")
	}
}

func TestConvertMessages_ToolUseFollowedByToolResult(t *testing.T) {
	// Full round-trip: assistant tool_use + user tool_result
	// This is the exact scenario that was failing: cross-provider conversation history
	toolUseBlocks := []types.ContentBlock{
		{
			Type:  "tool_use",
			ID:    "toolu_01Mg8jVsVMP5nUi6RPogGSkk",
			Name:  "get_weather",
			Input: json.RawMessage(`{"location":"San Francisco"}`),
		},
	}
	toolUseJSON, _ := json.Marshal(toolUseBlocks)

	toolResultBlocks := []types.ContentBlock{
		{
			Type:      "tool_result",
			ToolUseID: "toolu_01Mg8jVsVMP5nUi6RPogGSkk",
			Content:   "72°F, sunny",
		},
	}
	toolResultJSON, _ := json.Marshal(toolResultBlocks)

	messages := []types.Message{
		{Role: "assistant", Content: toolUseJSON},
		{Role: "user", Content: toolResultJSON},
	}

	result, err := convertMessages(messages)
	if err != nil {
		t.Fatalf("convertMessages: %v", err)
	}
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}

	// Assistant message should have tool_use block
	if len(result[0].Content) != 1 {
		t.Fatalf("expected 1 block in assistant message, got %d", len(result[0].Content))
	}
	if result[0].Content[0].OfToolUse == nil {
		t.Fatal("expected assistant block to be tool_use")
	}
	if result[0].Content[0].OfToolUse.ID != "toolu_01Mg8jVsVMP5nUi6RPogGSkk" {
		t.Errorf("tool_use ID = %q, want %q", result[0].Content[0].OfToolUse.ID, "toolu_01Mg8jVsVMP5nUi6RPogGSkk")
	}

	// User message should have tool_result block
	if len(result[1].Content) != 1 {
		t.Fatalf("expected 1 block in user message, got %d", len(result[1].Content))
	}
	if result[1].Content[0].OfToolResult == nil {
		t.Fatal("expected user block to be tool_result")
	}
	if result[1].Content[0].OfToolResult.ToolUseID != "toolu_01Mg8jVsVMP5nUi6RPogGSkk" {
		t.Errorf("tool_result ToolUseID = %q, want %q", result[1].Content[0].OfToolResult.ToolUseID, "toolu_01Mg8jVsVMP5nUi6RPogGSkk")
	}
}

func TestConvertMessages_ToolUseNilInput(t *testing.T) {
	// Edge case: tool_use block with nil input
	blocks := []types.ContentBlock{
		{
			Type: "tool_use",
			ID:   "toolu_nil",
			Name: "no_args_tool",
		},
	}
	blocksJSON, _ := json.Marshal(blocks)

	messages := []types.Message{
		{Role: "assistant", Content: blocksJSON},
	}

	result, err := convertMessages(messages)
	if err != nil {
		t.Fatalf("convertMessages: %v", err)
	}
	if len(result[0].Content) != 1 {
		t.Fatalf("expected 1 block, got %d", len(result[0].Content))
	}
	if result[0].Content[0].OfToolUse == nil {
		t.Fatal("expected tool_use block")
	}
	if result[0].Content[0].OfToolUse.Name != "no_args_tool" {
		t.Errorf("name = %q, want %q", result[0].Content[0].OfToolUse.Name, "no_args_tool")
	}
}
