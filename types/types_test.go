package types

import (
	"encoding/json"
	"testing"
)

func TestServerToolWebSearchConstant(t *testing.T) {
	if ServerToolWebSearch != "web_search" {
		t.Errorf("ServerToolWebSearch = %q, want %q", ServerToolWebSearch, "web_search")
	}
}

func TestIsClientTool(t *testing.T) {
	tests := []struct {
		name string
		tool ToolDefinition
		want bool
	}{
		{
			name: "with InputSchema is client tool",
			tool: ToolDefinition{
				Name:        "get_weather",
				Description: "Get weather",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
			},
			want: true,
		},
		{
			name: "without InputSchema is server tool",
			tool: ToolDefinition{Name: "web_search"},
			want: false,
		},
		{
			name: "nil InputSchema is server tool",
			tool: ToolDefinition{Name: "code_execution", InputSchema: nil},
			want: false,
		},
		{
			name: "empty InputSchema is server tool",
			tool: ToolDefinition{Name: "web_search", InputSchema: json.RawMessage{}},
			want: false,
		},
		{
			name: "web_search with schema overrides as client tool",
			tool: ToolDefinition{
				Name:        ServerToolWebSearch,
				Description: "My custom search",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}}}`),
			},
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.tool.IsClientTool(); got != tt.want {
				t.Errorf("IsClientTool() = %v, want %v", got, tt.want)
			}
		})
	}
}
