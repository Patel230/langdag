package gemini

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"langdag.com/langdag/types"
)

const defaultBaseURL = "https://generativelanguage.googleapis.com/v1beta"

// Provider implements the provider interface for the direct Gemini API.
type Provider struct {
	apiKey  string
	baseURL string
	client  *http.Client
}

// New creates a new Gemini provider.
func New(apiKey string) *Provider {
	return &Provider{
		apiKey:  apiKey,
		baseURL: defaultBaseURL,
		client:  &http.Client{},
	}
}

// Name returns the provider name.
func (p *Provider) Name() string {
	return "gemini"
}

// Models returns the available models.
func (p *Provider) Models() []types.ModelInfo {
	return []types.ModelInfo{
		{ID: "gemini-2.0-flash", Name: "Gemini 2.0 Flash", ContextWindow: 1048576, MaxOutput: 8192},
		{ID: "gemini-2.5-pro-preview-05-06", Name: "Gemini 2.5 Pro", ContextWindow: 1048576, MaxOutput: 65536},
	}
}

// Complete performs a synchronous completion request.
func (p *Provider) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	body := buildRequest(req)
	url := fmt.Sprintf("%s/models/%s:generateContent?key=%s", p.baseURL, req.Model, p.apiKey)

	respBody, err := doHTTPRequest(ctx, p.client, url, body, nil)
	if err != nil {
		return nil, err
	}
	defer respBody.Close()

	var resp geminiResponse
	if err := json.NewDecoder(respBody).Decode(&resp); err != nil {
		return nil, fmt.Errorf("gemini: failed to decode response: %w", err)
	}

	return convertResponse(&resp), nil
}

// Stream performs a streaming completion request.
func (p *Provider) Stream(ctx context.Context, req *types.CompletionRequest) (<-chan types.StreamEvent, error) {
	body := buildRequest(req)
	url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse&key=%s", p.baseURL, req.Model, p.apiKey)

	respBody, err := doHTTPRequest(ctx, p.client, url, body, nil)
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
