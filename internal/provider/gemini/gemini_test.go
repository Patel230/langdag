package gemini

import (
	"encoding/json"
	"strings"
	"testing"

	"langdag.com/langdag/types"
)

func TestConvertMessages_PlainText(t *testing.T) {
	messages := []types.Message{
		{Role: "user", Content: json.RawMessage(`"Hello"`)},
		{Role: "assistant", Content: json.RawMessage(`"Hi there"`)},
	}

	result := convertMessages(messages)

	if len(result) != 2 {
		t.Fatalf("expected 2 contents, got %d", len(result))
	}
	if result[0].Role != "user" {
		t.Errorf("expected user role, got %s", result[0].Role)
	}
	if result[1].Role != "model" {
		t.Errorf("expected model role for assistant, got %s", result[1].Role)
	}
	if result[0].Parts[0].Text != "Hello" {
		t.Errorf("expected 'Hello', got '%s'", result[0].Parts[0].Text)
	}
}

func TestConvertMessages_ToolUse(t *testing.T) {
	blocks := []types.ContentBlock{
		{Type: "text", Text: "Let me search"},
		{Type: "tool_use", Name: "search", Input: json.RawMessage(`{"q":"test"}`)},
	}
	content, _ := json.Marshal(blocks)

	messages := []types.Message{
		{Role: "assistant", Content: content},
	}

	result := convertMessages(messages)

	if len(result) != 1 {
		t.Fatalf("expected 1 content, got %d", len(result))
	}
	if result[0].Role != "model" {
		t.Errorf("expected model role, got %s", result[0].Role)
	}
	if len(result[0].Parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(result[0].Parts))
	}
	if result[0].Parts[0].Text != "Let me search" {
		t.Errorf("expected text part, got %+v", result[0].Parts[0])
	}
	if result[0].Parts[1].FunctionCall == nil {
		t.Fatal("expected function call part")
	}
	if result[0].Parts[1].FunctionCall.Name != "search" {
		t.Errorf("expected function name 'search', got %s", result[0].Parts[1].FunctionCall.Name)
	}
}

func TestConvertMessages_ToolResult(t *testing.T) {
	blocks := []types.ContentBlock{
		{Type: "tool_result", ToolUseID: "search", Content: "result data"},
	}
	content, _ := json.Marshal(blocks)

	messages := []types.Message{
		{Role: "user", Content: content},
	}

	result := convertMessages(messages)

	if len(result) != 1 {
		t.Fatalf("expected 1 content, got %d", len(result))
	}
	if result[0].Parts[0].FunctionResponse == nil {
		t.Fatal("expected function response part")
	}
	if result[0].Parts[0].FunctionResponse.Name != "search" {
		t.Errorf("expected name 'search', got %s", result[0].Parts[0].FunctionResponse.Name)
	}
}

func TestMapUsage(t *testing.T) {
	u := &usageMetadata{
		PromptTokenCount:        100,
		CandidatesTokenCount:    50,
		CachedContentTokenCount: 30,
		ThoughtsTokenCount:      10,
	}

	result := mapUsage(u)

	if result.InputTokens != 100 {
		t.Errorf("expected InputTokens=100, got %d", result.InputTokens)
	}
	if result.OutputTokens != 50 {
		t.Errorf("expected OutputTokens=50, got %d", result.OutputTokens)
	}
	if result.CacheReadInputTokens != 30 {
		t.Errorf("expected CacheReadInputTokens=30, got %d", result.CacheReadInputTokens)
	}
	if result.ReasoningTokens != 10 {
		t.Errorf("expected ReasoningTokens=10, got %d", result.ReasoningTokens)
	}
}

func TestConvertResponse(t *testing.T) {
	resp := &geminiResponse{
		Candidates: []candidate{
			{
				Content: content{
					Parts: []part{
						{Text: "Hello world"},
					},
				},
				FinishReason: "STOP",
			},
		},
		UsageMetadata: &usageMetadata{
			PromptTokenCount:     10,
			CandidatesTokenCount: 5,
		},
	}

	result := convertResponse(resp)

	if result.StopReason != "stop" {
		t.Errorf("expected stop reason 'stop', got %s", result.StopReason)
	}
	if len(result.Content) != 1 || result.Content[0].Text != "Hello world" {
		t.Errorf("unexpected content: %+v", result.Content)
	}
	if result.Usage.InputTokens != 10 {
		t.Errorf("expected InputTokens=10, got %d", result.Usage.InputTokens)
	}
}

func TestConvertResponse_FunctionCall(t *testing.T) {
	resp := &geminiResponse{
		Candidates: []candidate{
			{
				Content: content{
					Parts: []part{
						{FunctionCall: &functionCall{
							Name: "search",
							Args: map[string]interface{}{"q": "test"},
						}},
					},
				},
			},
		},
	}

	result := convertResponse(resp)

	if len(result.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(result.Content))
	}
	if result.Content[0].Type != "tool_use" {
		t.Errorf("expected tool_use type, got %s", result.Content[0].Type)
	}
	if result.Content[0].Name != "search" {
		t.Errorf("expected name 'search', got %s", result.Content[0].Name)
	}
}

func TestParseSSEStream(t *testing.T) {
	// Gemini sends full response snapshots
	sseData := `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":1}}

data: {"candidates":[{"content":{"parts":[{"text":"Hello world"}]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":2}}

data: {"candidates":[{"content":{"parts":[{"text":"Hello world!"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3}}

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

	if collected[0].Type != types.StreamEventStart {
		t.Errorf("expected start event, got %s", collected[0].Type)
	}

	// Collect text deltas
	var text string
	for _, e := range collected {
		if e.Type == types.StreamEventDelta {
			text += e.Content
		}
	}
	if text != "Hello world!" {
		t.Errorf("expected 'Hello world!', got '%s'", text)
	}

	// Last should be done with usage
	last := collected[len(collected)-1]
	if last.Type != types.StreamEventDone {
		t.Errorf("expected done event, got %s", last.Type)
	}
	if last.Response == nil {
		t.Fatal("expected response in done event")
	}
	if last.Response.Usage.InputTokens != 10 {
		t.Errorf("expected InputTokens=10, got %d", last.Response.Usage.InputTokens)
	}
}

func TestParseSSEStream_CacheTokens(t *testing.T) {
	sseData := `data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"cachedContentTokenCount":80,"thoughtsTokenCount":3}}

`

	events := make(chan types.StreamEvent, 20)
	go func() {
		defer close(events)
		parseSSEStream(strings.NewReader(sseData), events)
	}()

	var doneResp *types.CompletionResponse
	for ev := range events {
		if ev.Type == types.StreamEventDone {
			doneResp = ev.Response
		}
	}

	if doneResp == nil {
		t.Fatal("expected done response")
	}
	if doneResp.Usage.InputTokens != 100 {
		t.Errorf("InputTokens = %d, want 100", doneResp.Usage.InputTokens)
	}
	if doneResp.Usage.CacheReadInputTokens != 80 {
		t.Errorf("CacheReadInputTokens = %d, want 80", doneResp.Usage.CacheReadInputTokens)
	}
	if doneResp.Usage.ReasoningTokens != 3 {
		t.Errorf("ReasoningTokens = %d, want 3", doneResp.Usage.ReasoningTokens)
	}
}

func TestParseSSEStream_DoneResponseContainsText(t *testing.T) {
	sseData := `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1}}

data: {"candidates":[{"content":{"parts":[{"text":"Hello world"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2}}

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

	// Find the done event
	var done *types.StreamEvent
	for i := range collected {
		if collected[i].Type == types.StreamEventDone {
			done = &collected[i]
			break
		}
	}
	if done == nil {
		t.Fatal("expected a done event")
	}
	if done.Response == nil {
		t.Fatal("expected response in done event")
	}
	if len(done.Response.Content) != 1 {
		t.Fatalf("expected 1 content block in done response, got %d", len(done.Response.Content))
	}
	if done.Response.Content[0].Type != "text" {
		t.Errorf("expected text block, got %s", done.Response.Content[0].Type)
	}
	if done.Response.Content[0].Text != "Hello world" {
		t.Errorf("expected full text 'Hello world', got '%s'", done.Response.Content[0].Text)
	}
}

func TestParseSSEStream_DoneResponseContainsFunctionCall(t *testing.T) {
	sseData := `data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{"q":"test"}}}]}}],"usageMetadata":{"promptTokenCount":8,"candidatesTokenCount":5}}

data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{"q":"test"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":8,"candidatesTokenCount":5}}

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

	// Find the done event
	var done *types.StreamEvent
	for i := range collected {
		if collected[i].Type == types.StreamEventDone {
			done = &collected[i]
			break
		}
	}
	if done == nil {
		t.Fatal("expected a done event")
	}
	if done.Response == nil {
		t.Fatal("expected response in done event")
	}
	// Should have at least one tool_use block (may be duplicated from multiple SSE chunks)
	hasToolUse := false
	for _, block := range done.Response.Content {
		if block.Type == "tool_use" && block.Name == "search" {
			hasToolUse = true
			var args map[string]interface{}
			if err := json.Unmarshal(block.Input, &args); err != nil {
				t.Fatalf("failed to unmarshal tool input: %v", err)
			}
			if args["q"] != "test" {
				t.Errorf("expected args.q='test', got %v", args["q"])
			}
			break
		}
	}
	if !hasToolUse {
		t.Errorf("expected at least one tool_use block in done response, got content: %+v", done.Response.Content)
	}
}

func TestConvertTools(t *testing.T) {
	tools := []types.ToolDefinition{
		{
			Name:        "search",
			Description: "Search the web",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`),
		},
	}

	result := convertTools(tools)

	if len(result) != 1 {
		t.Fatalf("expected 1 tool group, got %d", len(result))
	}
	if len(result[0].FunctionDeclarations) != 1 {
		t.Fatalf("expected 1 declaration, got %d", len(result[0].FunctionDeclarations))
	}
	if result[0].FunctionDeclarations[0].Name != "search" {
		t.Errorf("expected name 'search', got %s", result[0].FunctionDeclarations[0].Name)
	}
}

func TestVertexProviderName(t *testing.T) {
	p := &VertexProvider{}
	if p.Name() != "gemini-vertex" {
		t.Errorf("expected name 'gemini-vertex', got '%s'", p.Name())
	}
}

func TestVertexProviderModels(t *testing.T) {
	p := &VertexProvider{}
	models := p.Models()
	if len(models) == 0 {
		t.Fatal("expected at least one model")
	}
	for _, m := range models {
		if m.ID == "" || m.Name == "" {
			t.Errorf("model missing required fields: %+v", m)
		}
	}
}

func TestBuildRequest_SystemInstruction(t *testing.T) {
	req := &types.CompletionRequest{
		Model:     "gemini-2.0-flash",
		Messages:  []types.Message{{Role: "user", Content: json.RawMessage(`"Hi"`)}},
		System:    "You are helpful",
		MaxTokens: 1000,
	}

	body := buildRequest(req)

	var gr geminiRequest
	json.Unmarshal(body, &gr)

	if gr.SystemInstruction == nil {
		t.Fatal("expected system instruction")
	}
	if gr.SystemInstruction.Parts[0].Text != "You are helpful" {
		t.Errorf("expected system text, got %s", gr.SystemInstruction.Parts[0].Text)
	}
	if gr.GenerationConfig == nil || gr.GenerationConfig.MaxOutputTokens != 1000 {
		t.Errorf("expected max_output_tokens=1000")
	}
}
