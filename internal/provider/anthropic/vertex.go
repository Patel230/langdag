package anthropic

import (
	"context"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/vertex"
	"github.com/langdag/langdag/pkg/types"
)

// VertexProvider implements the provider interface for Anthropic via Vertex AI.
type VertexProvider struct {
	client anthropic.Client
}

// NewVertex creates a new Anthropic Vertex AI provider.
// It uses Google Application Default Credentials for authentication.
// The region and projectID must be provided.
func NewVertex(ctx context.Context, region, projectID string) (*VertexProvider, error) {
	client := anthropic.NewClient(
		vertex.WithGoogleAuth(ctx, region, projectID),
	)
	return &VertexProvider{client: client}, nil
}

// Name returns the provider name.
func (p *VertexProvider) Name() string {
	return "anthropic-vertex"
}

// Models returns the available models.
func (p *VertexProvider) Models() []types.ModelInfo {
	return []types.ModelInfo{
		{ID: "claude-sonnet-4@20250514", Name: "Claude Sonnet 4 (Vertex)", ContextWindow: 200000, MaxOutput: 8192},
		{ID: "claude-opus-4@20250514", Name: "Claude Opus 4 (Vertex)", ContextWindow: 200000, MaxOutput: 8192},
		{ID: "claude-haiku-3-5@20241022", Name: "Claude Haiku 3.5 (Vertex)", ContextWindow: 200000, MaxOutput: 8192},
	}
}

// Complete performs a basic completion request.
func (p *VertexProvider) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	params, err := buildParams(req)
	if err != nil {
		return nil, err
	}

	resp, err := p.client.Messages.New(ctx, params)
	if err != nil {
		return nil, fmt.Errorf("anthropic-vertex completion failed: %w", err)
	}

	return convertResponse(resp), nil
}

// Stream performs a streaming completion request.
func (p *VertexProvider) Stream(ctx context.Context, req *types.CompletionRequest) (<-chan types.StreamEvent, error) {
	params, err := buildParams(req)
	if err != nil {
		return nil, err
	}

	stream := p.client.Messages.NewStreaming(ctx, params)

	events := make(chan types.StreamEvent, 100)

	go func() {
		defer close(events)
		processStreamEvents(stream, events)
	}()

	return events, nil
}
