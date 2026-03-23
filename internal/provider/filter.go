package provider

import (
	"context"

	"langdag.com/langdag/types"
)

// filterProvider wraps a Provider and silently removes server tools that
// the target model does not support, based on ModelInfo.ServerTools.
type filterProvider struct {
	inner Provider
	// modelTools maps model ID → set of supported server tool names.
	modelTools map[string]map[string]bool
}

// WithServerToolFilter wraps a Provider so that unsupported server tools are
// silently stripped from CompletionRequests before they reach the inner provider.
// Capability data comes from the inner provider's Models() method.
func WithServerToolFilter(p Provider) Provider {
	modelTools := map[string]map[string]bool{}
	for _, m := range p.Models() {
		if len(m.ServerTools) > 0 {
			set := make(map[string]bool, len(m.ServerTools))
			for _, st := range m.ServerTools {
				set[st] = true
			}
			modelTools[m.ID] = set
		}
	}
	return &filterProvider{inner: p, modelTools: modelTools}
}

func (f *filterProvider) Name() string              { return f.inner.Name() }
func (f *filterProvider) Models() []types.ModelInfo  { return f.inner.Models() }

func (f *filterProvider) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	return f.inner.Complete(ctx, f.filterTools(req))
}

func (f *filterProvider) Stream(ctx context.Context, req *types.CompletionRequest) (<-chan types.StreamEvent, error) {
	return f.inner.Stream(ctx, f.filterTools(req))
}

// filterTools returns a (possibly modified) copy of req with unsupported
// server tools removed. Client tools are always preserved.
func (f *filterProvider) filterTools(req *types.CompletionRequest) *types.CompletionRequest {
	if len(req.Tools) == 0 {
		return req
	}

	supported := f.modelTools[req.Model] // nil means no server tools supported

	// Fast path: check if any filtering is needed.
	needsFilter := false
	for _, t := range req.Tools {
		if !t.IsClientTool() && !supported[t.Name] {
			needsFilter = true
			break
		}
	}
	if !needsFilter {
		return req
	}

	// Shallow-copy the request and rebuild the tools slice.
	filtered := *req
	filtered.Tools = make([]types.ToolDefinition, 0, len(req.Tools))
	for _, t := range req.Tools {
		if t.IsClientTool() || supported[t.Name] {
			filtered.Tools = append(filtered.Tools, t)
		}
	}
	return &filtered
}
