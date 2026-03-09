package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"langdag.com/langdag/types"
)

// AzureProvider implements the provider interface for Azure OpenAI.
// Azure uses a different URL scheme and auth header compared to direct OpenAI.
type AzureProvider struct {
	apiKey     string
	endpoint   string // e.g. "https://myresource.openai.azure.com"
	apiVersion string
	client     *http.Client
}

// NewAzure creates a new Azure OpenAI provider.
func NewAzure(apiKey, endpoint, apiVersion string) *AzureProvider {
	if apiVersion == "" {
		apiVersion = "2024-08-01-preview"
	}
	endpoint = strings.TrimRight(endpoint, "/")
	return &AzureProvider{
		apiKey:     apiKey,
		endpoint:   endpoint,
		apiVersion: apiVersion,
		client:     &http.Client{},
	}
}

// Name returns the provider name.
func (p *AzureProvider) Name() string {
	return "openai-azure"
}

// Models returns the available models.
func (p *AzureProvider) Models() []types.ModelInfo {
	return []types.ModelInfo{
		{ID: "gpt-4o", Name: "GPT-4o (Azure)", ContextWindow: 128000, MaxOutput: 16384},
		{ID: "gpt-4o-mini", Name: "GPT-4o Mini (Azure)", ContextWindow: 128000, MaxOutput: 16384},
	}
}

// Complete performs a synchronous completion request.
func (p *AzureProvider) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	body := buildRequest(req, false, openAIServerTools)

	respBody, err := p.doRequest(ctx, req.Model, body)
	if err != nil {
		return nil, err
	}
	defer respBody.Close()

	var resp chatCompletionResponse
	if err := json.NewDecoder(respBody).Decode(&resp); err != nil {
		return nil, fmt.Errorf("openai-azure: failed to decode response: %w", err)
	}

	return convertResponse(&resp), nil
}

// Stream performs a streaming completion request.
func (p *AzureProvider) Stream(ctx context.Context, req *types.CompletionRequest) (<-chan types.StreamEvent, error) {
	body := buildRequest(req, true, openAIServerTools)

	respBody, err := p.doRequest(ctx, req.Model, body)
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

func (p *AzureProvider) doRequest(ctx context.Context, model string, body []byte) (io.ReadCloser, error) {
	// Azure URL: {endpoint}/openai/deployments/{model}/chat/completions?api-version={version}
	url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s", p.endpoint, model, p.apiVersion)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai-azure: failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", p.apiKey)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai-azure: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai-azure: API error (status %d): %s", resp.StatusCode, string(bodyBytes))
	}

	return resp.Body, nil
}
