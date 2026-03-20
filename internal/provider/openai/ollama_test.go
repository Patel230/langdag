package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"langdag.com/langdag/types"
)

func TestOllamaProviderName(t *testing.T) {
	p := NewOllama("")
	if p.Name() != "ollama" {
		t.Errorf("expected name 'ollama', got '%s'", p.Name())
	}
}

func TestOllamaDefaultBaseURL(t *testing.T) {
	p := NewOllama("")
	if p.baseURL != "http://localhost:11434" {
		t.Errorf("expected default base URL 'http://localhost:11434', got '%s'", p.baseURL)
	}
}

func TestOllamaCustomBaseURL(t *testing.T) {
	p := NewOllama("http://192.168.1.1:11434")
	if p.baseURL != "http://192.168.1.1:11434" {
		t.Errorf("expected custom base URL, got '%s'", p.baseURL)
	}
}

func TestOllamaBaseURLTrimming(t *testing.T) {
	p := NewOllama("http://localhost:11434/")
	if strings.HasSuffix(p.baseURL, "/") {
		t.Error("expected trailing slash to be trimmed from base URL")
	}
}

func TestOllamaAPIKeyNotSet(t *testing.T) {
	p := NewOllama("http://localhost:11434")
	if p.apiKey != "" {
		t.Errorf("expected empty API key, got '%s'", p.apiKey)
	}
}

func TestOllamaWithAPIKey(t *testing.T) {
	p := NewOllama("http://localhost:11434")
	p.apiKey = "test-key"
	if p.apiKey != "test-key" {
		t.Errorf("expected API key 'test-key', got '%s'", p.apiKey)
	}
}

func TestOllamaBuildRequest(t *testing.T) {
	req := &types.CompletionRequest{
		Model:       "llama3",
		MaxTokens:   100,
		Temperature: 0.7,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
		},
	}

	body := buildRequest(req, false, nil)

	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if parsed["model"] != "llama3" {
		t.Errorf("expected model 'llama3', got %v", parsed["model"])
	}
	if parsed["max_tokens"] != float64(100) {
		t.Errorf("expected max_tokens 100, got %v", parsed["max_tokens"])
	}
}

func TestOllamaBuildRequestWithSystem(t *testing.T) {
	req := &types.CompletionRequest{
		Model:  "llama3",
		System: "You are helpful",
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Hi"`)},
		},
	}

	body := buildRequest(req, false, nil)

	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	messages := parsed["messages"].([]interface{})
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}

	first := messages[0].(map[string]interface{})
	if first["role"] != "system" || first["content"] != "You are helpful" {
		t.Errorf("unexpected first message: %+v", first)
	}
}

func TestOllamaBuildRequestWithTools(t *testing.T) {
	req := &types.CompletionRequest{
		Model: "llama3",
		Tools: []types.ToolDefinition{
			{
				Name:        "get_weather",
				Description: "Get weather",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}}}`),
			},
		},
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Weather?`)},
		},
	}

	body := buildRequest(req, false, nil)

	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	tools, ok := parsed["tools"].([]interface{})
	if !ok || len(tools) == 0 {
		t.Fatal("expected tools to be set")
	}

	tool := tools[0].(map[string]interface{})
	if tool["type"] != "function" {
		t.Errorf("expected type 'function', got %v", tool["type"])
	}
}

func TestOllamaBuildRequestWithStopSeqs(t *testing.T) {
	req := &types.CompletionRequest{
		Model:    "llama3",
		StopSeqs: []string{"END", "STOP"},
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Hi"`)},
		},
	}

	body := buildRequest(req, false, nil)

	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	stop := parsed["stop"].([]interface{})
	if len(stop) != 2 {
		t.Errorf("expected 2 stop sequences, got %d", len(stop))
	}
}

func TestOllamaBuildRequest_Streaming(t *testing.T) {
	req := &types.CompletionRequest{
		Model: "llama3",
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"Hi"`)},
		},
	}

	body := buildRequest(req, true, nil)

	var parsed map[string]interface{}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if parsed["stream"] != true {
		t.Errorf("expected stream=true, got %v", parsed["stream"])
	}

	streamOpts, ok := parsed["stream_options"].(map[string]interface{})
	if !ok {
		t.Fatal("expected stream_options to be set")
	}
	if streamOpts["include_usage"] != true {
		t.Errorf("expected include_usage=true, got %v", streamOpts["include_usage"])
	}
}

func TestOllamaConvertResponse(t *testing.T) {
	stop := "stop"
	content := "Hello world"
	resp := &chatCompletionResponse{
		ID:    "chatcmpl-ollama-1",
		Model: "llama3",
		Choices: []choice{
			{
				Message: responseMessage{
					Content: &content,
				},
				FinishReason: &stop,
			},
		},
		Usage: &usage{
			PromptTokens:     10,
			CompletionTokens: 5,
		},
	}

	result := convertResponse(resp)

	if result.ID != "chatcmpl-ollama-1" {
		t.Errorf("expected ID 'chatcmpl-ollama-1', got %s", result.ID)
	}
	if result.StopReason != "stop" {
		t.Errorf("expected stop reason 'stop', got %s", result.StopReason)
	}
	if len(result.Content) != 1 || result.Content[0].Text != "Hello world" {
		t.Errorf("unexpected content: %+v", result.Content)
	}
	if result.Usage.InputTokens != 10 {
		t.Errorf("expected InputTokens=10, got %d", result.Usage.InputTokens)
	}
	if result.Usage.OutputTokens != 5 {
		t.Errorf("expected OutputTokens=5, got %d", result.Usage.OutputTokens)
	}
}

func TestOllamaConvertResponse_ToolCalls(t *testing.T) {
	stop := "tool_calls"
	resp := &chatCompletionResponse{
		ID:    "chatcmpl-ollama-2",
		Model: "llama3",
		Choices: []choice{
			{
				Message: responseMessage{
					ToolCalls: []responseToolCall{
						{
							ID:   "call_1",
							Type: "function",
							Function: responseFunction{
								Name:      "get_weather",
								Arguments: `{"location":"NYC"}`,
							},
						},
					},
				},
				FinishReason: &stop,
			},
		},
		Usage: &usage{PromptTokens: 15, CompletionTokens: 10},
	}

	result := convertResponse(resp)

	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}
	if result.Content[0].Type != "tool_use" {
		t.Errorf("expected tool_use type, got %s", result.Content[0].Type)
	}
	if result.Content[0].Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %s", result.Content[0].Name)
	}
	if string(result.Content[0].Input) != `{"location":"NYC"}` {
		t.Errorf("expected input JSON, got %s", string(result.Content[0].Input))
	}
}

func TestOllamaParseSSEStream(t *testing.T) {
	sseData := `data: {"id":"chatcmpl-1","model":"llama3","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"llama3","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}

data: {"id":"chatcmpl-1","model":"llama3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-1","model":"llama3","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}

data: [DONE]

`

	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
	}()

	var collected []types.StreamEvent
	for e := range events {
		collected = append(collected, e)
	}

	if len(collected) < 3 {
		t.Fatalf("expected at least 3 events, got %d", len(collected))
	}

	if collected[0].Type != types.StreamEventStart {
		t.Errorf("expected start event, got %s", collected[0].Type)
	}

	var text string
	for _, e := range collected {
		if e.Type == types.StreamEventDelta {
			text += e.Content
		}
	}
	if text != "Hi" {
		t.Errorf("expected 'Hi', got '%s'", text)
	}

	last := collected[len(collected)-1]
	if last.Type != types.StreamEventDone {
		t.Errorf("expected done event, got %s", last.Type)
	}
	if last.Response.Usage.InputTokens != 5 {
		t.Errorf("expected InputTokens=5, got %d", last.Response.Usage.InputTokens)
	}
}

func TestOllamaParseSSEStream_ToolCalls(t *testing.T) {
	sseData := `data: {"id":"chatcmpl-2","model":"llama3","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":""}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-2","model":"llama3","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"q\":"}}]}},"finish_reason":null}]}

data: {"id":"chatcmpl-2","model":"llama3","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"test\""}}]}},"finish_reason":null}]}

data: {"id":"chatcmpl-2","model":"llama3","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"chatcmpl-2","model":"llama3","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":10}}

data: [DONE]

`

	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
	}()

	var collected []types.StreamEvent
	for e := range events {
		collected = append(collected, e)
	}

	var foundToolDone bool
	for _, e := range collected {
		if e.Type == types.StreamEventContentDone {
			foundToolDone = true
			if e.ContentBlock.Name != "search" {
				t.Errorf("expected tool name 'search', got %s", e.ContentBlock.Name)
			}
		}
	}
	if !foundToolDone {
		t.Error("expected content_done event for tool call")
	}
}

func TestOllamaModels_EmptyResponse(t *testing.T) {
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
		t.Fatal("expected empty models slice, got nil")
	}
	if len(models) != 0 {
		t.Errorf("expected 0 models, got %d", len(models))
	}
}

func TestOllamaModels_SingleModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"models": [{"name": "llama3"}]}`))
			return
		}
		if r.URL.Path == "/api/show" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"capabilities": {"num_ctx": 8192}}`))
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
		t.Errorf("expected ID 'llama3', got '%s'", models[0].ID)
	}
	if models[0].ContextWindow != 8192 {
		t.Errorf("expected context window 8192, got %d", models[0].ContextWindow)
	}
}

func TestOllamaConvertMessages_ToolUseAndResult(t *testing.T) {
	blocks := []types.ContentBlock{
		{Type: "text", Text: "I'll search for that"},
		{Type: "tool_use", ID: "call_1", Name: "search", Input: json.RawMessage(`{"q":"weather"}`)},
	}
	content, _ := json.Marshal(blocks)

	messages := []types.Message{
		{Role: "assistant", Content: content},
	}

	result := convertMessages(messages, "")

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}

	if result[0].Role != "assistant" || len(result[0].ToolCalls) != 1 {
		t.Errorf("unexpected message: role=%s, toolcalls=%d", result[0].Role, len(result[0].ToolCalls))
	}

	if result[0].ToolCalls[0].Function.Name != "search" {
		t.Errorf("expected tool name 'search', got %s", result[0].ToolCalls[0].Function.Name)
	}
}

func TestOllamaConvertMessages_ImageContent(t *testing.T) {
	blocks := []types.ContentBlock{
		{Type: "image", URL: "http://example.com/image.png"},
	}
	content, _ := json.Marshal(blocks)

	messages := []types.Message{
		{Role: "user", Content: content},
	}

	result := convertMessages(messages, "")

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}

	parts, ok := result[0].Content.([]contentPart)
	if !ok {
		t.Fatal("expected content to be []contentPart")
	}
	if len(parts) != 1 || parts[0].Type != "image_url" {
		t.Errorf("unexpected content parts: %+v", parts)
	}
}

func TestOllamaConvertMessages_MultipleToolResults(t *testing.T) {
	blocks1 := []types.ContentBlock{
		{Type: "tool_result", ToolUseID: "call_1", Content: "result 1"},
	}
	blocks2 := []types.ContentBlock{
		{Type: "tool_result", ToolUseID: "call_2", Content: "result 2"},
	}
	content1, _ := json.Marshal(blocks1)
	content2, _ := json.Marshal(blocks2)

	messages := []types.Message{
		{Role: "user", Content: content1},
		{Role: "user", Content: content2},
	}

	result := convertMessages(messages, "")

	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}

	if result[0].Role != "tool" || result[0].ToolCallID != "call_1" {
		t.Errorf("unexpected first message: %+v", result[0])
	}
	if result[1].Role != "tool" || result[1].ToolCallID != "call_2" {
		t.Errorf("unexpected second message: %+v", result[1])
	}
}

func TestOllamaConvertTools_EmptyTools(t *testing.T) {
	tools := []types.ToolDefinition{}
	result := convertTools(tools, nil)
	if len(result) != 0 {
		t.Errorf("expected 0 tools, got %d", len(result))
	}
}

func TestDefaultContextWindow(t *testing.T) {
	tests := []struct {
		model    string
		expected int
	}{
		{"llama4", 1000000},
		{"llama3.1", 128000},
		{"llama3.2-vision", 128000},
		{"llama3.2", 128000},
		{"llama3.3", 128000},
		{"llama3-gradient", 1000000},
		{"llama3", 8192},
		{"llama2", 4096},
		{"qwen3", 32768},
		{"qwen2.5", 32768},
		{"qwen2", 32768},
		{"qwen", 32768},
		{"mistral-large", 128000},
		{"mistral-small3", 128000},
		{"mistral", 32768},
		{"mixtral", 32768},
		{"codestral", 32768},
		{"gemma3", 32768},
		{"gemma2", 8192},
		{"gemma", 8192},
		{"deepseek-v3", 64000},
		{"deepseek-r1", 64000},
		{"deepseek-coder", 16384},
		{"deepseek", 4096},
		{"phi4", 16384},
		{"phi3", 128000},
		{"phi", 2048},
		{"codellama", 16384},
		{"starcoder", 16384},
		{"yi", 32768},
		{"kimi", 128000},
		{"llava", 4096},
		{"unknown-model", 4096},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			got := defaultContextWindow(tt.model)
			if got != tt.expected {
				t.Errorf("defaultContextWindow(%q) = %d, want %d", tt.model, got, tt.expected)
			}
		})
	}
}

func TestOllamaConvertResponse_EmptyChoices(t *testing.T) {
	resp := &chatCompletionResponse{
		ID:      "chatcmpl-empty",
		Model:   "llama3",
		Choices: []choice{},
	}

	result := convertResponse(resp)

	if result.ID != "chatcmpl-empty" {
		t.Errorf("expected ID 'chatcmpl-empty', got %s", result.ID)
	}
	if len(result.Content) != 0 {
		t.Errorf("expected 0 content blocks, got %d", len(result.Content))
	}
}

func TestOllamaConvertResponse_NilUsage(t *testing.T) {
	stop := "stop"
	content := "test"
	resp := &chatCompletionResponse{
		ID:    "chatcmpl-nilusage",
		Model: "llama3",
		Choices: []choice{
			{
				Message:      responseMessage{Content: &content},
				FinishReason: &stop,
			},
		},
		Usage: nil,
	}

	result := convertResponse(resp)

	if result.Usage.InputTokens != 0 {
		t.Errorf("expected InputTokens=0, got %d", result.Usage.InputTokens)
	}
}

func TestOllamaConvertResponse_TextAndToolCall(t *testing.T) {
	stop := "stop"
	resp := &chatCompletionResponse{
		ID:    "chatcmpl-mixed",
		Model: "llama3",
		Choices: []choice{
			{
				Message: responseMessage{
					Content: strPtr("Here is the weather:"),
					ToolCalls: []responseToolCall{
						{
							ID:       "call_1",
							Type:     "function",
							Function: responseFunction{Name: "get_weather", Arguments: `{"city":"NYC"}`},
						},
					},
				},
				FinishReason: &stop,
			},
		},
		Usage: &usage{PromptTokens: 10, CompletionTokens: 20},
	}

	result := convertResponse(resp)

	if len(result.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(result.Content))
	}
	if result.Content[0].Type != "text" || result.Content[0].Text != "Here is the weather:" {
		t.Errorf("unexpected text block: %+v", result.Content[0])
	}
	if result.Content[1].Type != "tool_use" || result.Content[1].Name != "get_weather" {
		t.Errorf("unexpected tool block: %+v", result.Content[1])
	}
}

func TestOllamaConvertMessages_SystemAndUser(t *testing.T) {
	messages := []types.Message{
		{Role: "user", Content: json.RawMessage(`"Hello"`)},
	}

	result := convertMessages(messages, "You are a helpful assistant")

	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}
	if result[0].Role != "system" {
		t.Errorf("expected first message to be system, got %s", result[0].Role)
	}
	if result[1].Role != "user" {
		t.Errorf("expected second message to be user, got %s", result[1].Role)
	}
}

func TestOllamaConvertMessages_AssistantMessage(t *testing.T) {
	messages := []types.Message{
		{Role: "assistant", Content: json.RawMessage(`"I am an assistant"`)},
	}

	result := convertMessages(messages, "")

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].Role != "assistant" {
		t.Errorf("expected assistant role, got %s", result[0].Role)
	}
	content, ok := result[0].Content.(string)
	if !ok || content != "I am an assistant" {
		t.Errorf("expected content 'I am an assistant', got %v", result[0].Content)
	}
}

func TestOllamaConvertMessages_MalformedContent(t *testing.T) {
	messages := []types.Message{
		{Role: "user", Content: json.RawMessage(`{invalid json}`)},
	}

	result := convertMessages(messages, "")

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	content, ok := result[0].Content.(string)
	if !ok || content != "{invalid json}" {
		t.Errorf("expected raw content, got %v", result[0].Content)
	}
}

func TestOllamaConvertMessages_DocumentContent(t *testing.T) {
	blocks := []types.ContentBlock{
		{Type: "document", Data: "document text content", MediaType: "text/plain"},
	}
	content, _ := json.Marshal(blocks)

	messages := []types.Message{
		{Role: "user", Content: content},
	}

	result := convertMessages(messages, "")

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	// Document text/plain is converted to string via extractText
	contentStr, ok := result[0].Content.(string)
	if !ok {
		t.Fatal("expected content to be string")
	}
	if contentStr != "document text content" {
		t.Errorf("unexpected content: %s", contentStr)
	}
}

func TestOllamaConvertMessages_NonTextDocumentIgnored(t *testing.T) {
	blocks := []types.ContentBlock{
		{Type: "document", Data: "pdf bytes", MediaType: "application/pdf"},
		{Type: "text", Text: "readable text"},
	}
	content, _ := json.Marshal(blocks)

	messages := []types.Message{
		{Role: "user", Content: content},
	}

	result := convertMessages(messages, "")

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	contentStr, ok := result[0].Content.(string)
	if !ok {
		t.Fatal("expected content to be string")
	}
	if contentStr != "readable text" {
		t.Errorf("expected only readable text, got: %s", contentStr)
	}
}

func TestOllamaConvertMessages_ImageWithBase64(t *testing.T) {
	blocks := []types.ContentBlock{
		{Type: "image", Data: "base64image123", MediaType: "image/png"},
	}
	content, _ := json.Marshal(blocks)

	messages := []types.Message{
		{Role: "user", Content: content},
	}

	result := convertMessages(messages, "")

	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	parts, ok := result[0].Content.([]contentPart)
	if !ok || len(parts) != 1 {
		t.Fatal("expected one content part")
	}
	if !strings.HasPrefix(parts[0].ImageURL.URL, "data:image/png;base64,") {
		t.Errorf("expected base64 data URL, got %s", parts[0].ImageURL.URL)
	}
}

func TestOllamaConvertTools_FunctionTools(t *testing.T) {
	tools := []types.ToolDefinition{
		{
			Name:        "tool1",
			Description: "First tool",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		},
		{
			Name:        "tool2",
			Description: "Second tool",
			InputSchema: json.RawMessage(`{"type":"object"}`),
		},
	}

	result := convertTools(tools, nil)

	if len(result) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(result))
	}
	if result[0].Function.Name != "tool1" || result[1].Function.Name != "tool2" {
		t.Errorf("unexpected tool names: %v", result)
	}
}

func TestOllamaBuildRequest_WithMaxTokens(t *testing.T) {
	req := &types.CompletionRequest{
		Model:     "llama3",
		MaxTokens: 500,
		Messages:  []types.Message{{Role: "user", Content: json.RawMessage(`"Hi"`)}},
	}

	body := buildRequest(req, false, nil)

	var parsed map[string]interface{}
	json.Unmarshal(body, &parsed)

	if parsed["max_tokens"] != float64(500) {
		t.Errorf("expected max_tokens=500, got %v", parsed["max_tokens"])
	}
}

func TestOllamaBuildRequest_WithTemperature(t *testing.T) {
	req := &types.CompletionRequest{
		Model:       "llama3",
		Temperature: 0.9,
		Messages:    []types.Message{{Role: "user", Content: json.RawMessage(`"Hi"`)}},
	}

	body := buildRequest(req, false, nil)

	var parsed map[string]interface{}
	json.Unmarshal(body, &parsed)

	if parsed["temperature"] != float64(0.9) {
		t.Errorf("expected temperature=0.9, got %v", parsed["temperature"])
	}
}

func TestOllamaBuildRequest_NilTemperature(t *testing.T) {
	req := &types.CompletionRequest{
		Model:       "llama3",
		Temperature: 0,
		Messages:    []types.Message{{Role: "user", Content: json.RawMessage(`"Hi"`)}},
	}

	body := buildRequest(req, false, nil)

	var parsed map[string]interface{}
	json.Unmarshal(body, &parsed)

	if _, exists := parsed["temperature"]; exists {
		t.Error("expected temperature to be omitted when 0")
	}
}

func TestOllamaParseSSEStream_MultipleChoices(t *testing.T) {
	sseData := `data: {"id":"chatcmpl-multi","model":"llama3","choices":[{"index":0,"delta":{"content":"First"},"finish_reason":null}]}

data: {"id":"chatcmpl-multi","model":"llama3","choices":[{"index":1,"delta":{"content":"Second"},"finish_reason":null}]}

data: {"id":"chatcmpl-multi","model":"llama3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`

	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
	}()

	var deltas []string
	for e := range events {
		if e.Type == types.StreamEventDelta {
			deltas = append(deltas, e.Content)
		}
	}

	if len(deltas) != 2 {
		t.Errorf("expected 2 deltas, got %d", len(deltas))
	}
}

func TestOllamaParseSSEStream_EmptyContentDelta(t *testing.T) {
	sseData := `data: {"id":"chatcmpl-empty","model":"llama3","choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-empty","model":"llama3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`

	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
	}()

	count := 0
	for e := range events {
		if e.Type == types.StreamEventDelta {
			count++
		}
	}

	if count != 0 {
		t.Errorf("expected 0 deltas for empty content, got %d", count)
	}
}

func TestOllamaParseSSEStream_UnknownField(t *testing.T) {
	sseData := `data: {"id":"chatcmpl-unk","model":"llama3","choices":[{"index":0,"delta":{"content":"Hi"}}],"unknown_field":"should be ignored"}

data: {"id":"chatcmpl-unk","model":"llama3","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

`

	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
	}()

	var text string
	for e := range events {
		if e.Type == types.StreamEventDelta {
			text += e.Content
		}
	}

	if text != "Hi" {
		t.Errorf("expected 'Hi', got '%s'", text)
	}
}

func TestOllamaDoRequest_NoAuthHeaderWhenNoKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Errorf("expected no Authorization header, got %s", auth)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"test","model":"llama3","choices":[]}`))
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	p.apiKey = ""
	_, _ = p.Complete(context.Background(), &types.CompletionRequest{
		Model:    "llama3",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	})
}

func TestOllamaDoRequest_WithAuthHeader(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "Bearer my-secret-key" {
			t.Errorf("expected 'Bearer my-secret-key', got %s", auth)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"test","model":"llama3","choices":[]}`))
	}))
	defer server.Close()

	p := NewOllama(server.URL)
	p.apiKey = "my-secret-key"
	_, _ = p.Complete(context.Background(), &types.CompletionRequest{
		Model:    "llama3",
		Messages: []types.Message{{Role: "user", Content: json.RawMessage(`"hi"`)}},
	})
}

func strPtr(s string) *string { return &s }
