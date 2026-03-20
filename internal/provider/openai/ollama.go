package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"langdag.com/langdag/types"
)

// ollamaProvider implements the provider interface for Ollama.
// Ollama is a local LLM server with an OpenAI-compatible API at /v1/chat/completions.
type ollamaProvider struct {
	baseURL string
	apiKey  string
	client  *http.Client
}

func NewOllama(baseURL string) *ollamaProvider {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	baseURL = strings.TrimRight(baseURL, "/")
	return &ollamaProvider{
		baseURL: baseURL,
		client:  &http.Client{},
	}
}

func (p *ollamaProvider) Name() string {
	return "ollama"
}

type ollamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

type ollamaShowResponse struct {
	Capabilities struct {
		NumCtx int `json:"num_ctx"`
	} `json:"capabilities"`
}

func (p *ollamaProvider) Models() []types.ModelInfo {
	ctx := context.Background()
	url := p.baseURL + "/api/tags"

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil
	}

	var tagsResp ollamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tagsResp); err != nil {
		return nil
	}

	models := make([]types.ModelInfo, 0, len(tagsResp.Models))
	for _, m := range tagsResp.Models {
		models = append(models, types.ModelInfo{
			ID:            m.Name,
			Name:          m.Name,
			ContextWindow: p.estimateContextWindow(m.Name),
		})
	}
	return models
}

func (p *ollamaProvider) estimateContextWindow(modelName string) int {
	ctx := context.Background()
	url := p.baseURL + "/api/show"

	body, _ := json.Marshal(map[string]string{"name": modelName})
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return defaultContextWindow(modelName)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return defaultContextWindow(modelName)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return defaultContextWindow(modelName)
	}

	var showResp ollamaShowResponse
	if err := json.NewDecoder(resp.Body).Decode(&showResp); err != nil {
		return defaultContextWindow(modelName)
	}

	if showResp.Capabilities.NumCtx > 0 {
		return showResp.Capabilities.NumCtx
	}
	return defaultContextWindow(modelName)
}

var contextWindowCache = sync.Map{}

func (p *ollamaProvider) getContextWindowWithCache(modelName string) int {
	if cached, ok := contextWindowCache.Load(modelName); ok {
		return cached.(int)
	}
	ctx := context.Background()
	url := p.baseURL + "/api/show"

	body, _ := json.Marshal(map[string]string{"name": modelName})
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return defaultContextWindow(modelName)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return defaultContextWindow(modelName)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return defaultContextWindow(modelName)
	}

	var showResp ollamaShowResponse
	if err := json.NewDecoder(resp.Body).Decode(&showResp); err != nil {
		return defaultContextWindow(modelName)
	}

	contextWindow := showResp.Capabilities.NumCtx
	if contextWindow == 0 {
		contextWindow = defaultContextWindow(modelName)
	}
	contextWindowCache.Store(modelName, contextWindow)
	return contextWindow
}

func (p *ollamaProvider) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	body := buildRequest(req, false, nil)

	respBody, err := p.doRequest(ctx, body)
	if err != nil {
		return nil, err
	}
	defer respBody.Close()

	var resp chatCompletionResponse
	if err := json.NewDecoder(respBody).Decode(&resp); err != nil {
		return nil, fmt.Errorf("ollama: failed to decode response: %w", err)
	}

	return convertResponse(&resp), nil
}

func (p *ollamaProvider) Stream(ctx context.Context, req *types.CompletionRequest) (<-chan types.StreamEvent, error) {
	body := buildRequest(req, true, nil)

	respBody, err := p.doRequest(ctx, body)
	if err != nil {
		return nil, err
	}

	events := make(chan types.StreamEvent, 100)
	go func() {
		defer close(events)
		defer respBody.Close()
		parseSSEStream(respBody, events)
	}()

	return events, nil
}

func (p *ollamaProvider) doRequest(ctx context.Context, body []byte) (io.ReadCloser, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama: failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama: API error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	return resp.Body, nil
}

func defaultContextWindow(modelName string) int {
	nameLower := strings.ToLower(modelName)

	switch {
	case strings.Contains(nameLower, "llama4"):
		return 1000000
	case strings.Contains(nameLower, "llama3.1"):
		return 128000
	case strings.Contains(nameLower, "llama3.2-vision"), strings.Contains(nameLower, "llama3.2"):
		return 128000
	case strings.Contains(nameLower, "llama3.3"):
		return 128000
	case strings.Contains(nameLower, "llama3-gradient"):
		return 1000000
	case strings.Contains(nameLower, "llama3"):
		return 8192
	case strings.Contains(nameLower, "llama2"):
		return 4096
	case strings.Contains(nameLower, "qwen3"):
		return 32768
	case strings.Contains(nameLower, "qwen2.5"), strings.Contains(nameLower, "qwen2"):
		return 32768
	case strings.Contains(nameLower, "qwen"):
		return 32768
	case strings.Contains(nameLower, "mistral-large"), strings.Contains(nameLower, "mistral-small3"), strings.Contains(nameLower, "mistral-nemo"):
		return 128000
	case strings.Contains(nameLower, "mistral"), strings.Contains(nameLower, "mixtral"), strings.Contains(nameLower, "codestral"), strings.Contains(nameLower, "mathstral"):
		return 32768
	case strings.Contains(nameLower, "gemma3"):
		return 32768
	case strings.Contains(nameLower, "gemma2"), strings.Contains(nameLower, "gemma"):
		return 8192
	case strings.Contains(nameLower, "deepseek-v3"), strings.Contains(nameLower, "deepseek-r1"):
		return 64000
	case strings.Contains(nameLower, "deepseek-coder"):
		return 16384
	case strings.Contains(nameLower, "deepseek"):
		return 4096
	case strings.Contains(nameLower, "phi4"):
		return 16384
	case strings.Contains(nameLower, "phi3"):
		return 128000
	case strings.Contains(nameLower, "phi"):
		return 2048
	case strings.Contains(nameLower, "lfm"), strings.Contains(nameLower, "glm"), strings.Contains(nameLower, "granite"):
		return 32768
	case strings.Contains(nameLower, "nemotron"):
		return 4096
	case strings.Contains(nameLower, "command-"):
		return 128000
	case strings.Contains(nameLower, "codellama"), strings.Contains(nameLower, "starcoder"), strings.Contains(nameLower, "yi-coder"):
		return 16384
	case strings.Contains(nameLower, "yi"), strings.Contains(nameLower, "zephyr"), strings.Contains(nameLower, "cogito"), strings.Contains(nameLower, "devstral"):
		return 32768
	case strings.Contains(nameLower, "kimi"):
		return 128000
	case strings.Contains(nameLower, "llava"), strings.Contains(nameLower, "bakllava"):
		return 4096
	case strings.Contains(nameLower, "minicpm-v"):
		return 8192
	case strings.Contains(nameLower, "embed"), strings.Contains(nameLower, "nomic"):
		return 8192
	default:
		return 4096
	}
}
