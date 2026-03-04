package provider

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/langdag/langdag/pkg/types"
)

// failProvider fails N times then succeeds.
type failProvider struct {
	failCount  int
	callCount  int
	failErr    error
}

func (p *failProvider) Complete(ctx context.Context, req *types.CompletionRequest) (*types.CompletionResponse, error) {
	p.callCount++
	if p.callCount <= p.failCount {
		return nil, p.failErr
	}
	return &types.CompletionResponse{Content: []types.ContentBlock{{Type: "text", Text: "ok"}}}, nil
}

func (p *failProvider) Stream(ctx context.Context, req *types.CompletionRequest) (<-chan types.StreamEvent, error) {
	p.callCount++
	if p.callCount <= p.failCount {
		return nil, p.failErr
	}
	ch := make(chan types.StreamEvent, 1)
	ch <- types.StreamEvent{Type: types.StreamEventDone}
	close(ch)
	return ch, nil
}

func (p *failProvider) Name() string             { return "fail-provider" }
func (p *failProvider) Models() []types.ModelInfo { return nil }

func TestRetryComplete_TransientThenSuccess(t *testing.T) {
	inner := &failProvider{failCount: 2, failErr: fmt.Errorf("status 503: service unavailable")}
	prov := WithRetry(inner, RetryConfig{MaxRetries: 3, BaseDelay: 1 * time.Millisecond, MaxDelay: 10 * time.Millisecond})

	resp, err := prov.Complete(context.Background(), &types.CompletionRequest{})
	if err != nil {
		t.Fatalf("expected success, got: %v", err)
	}
	if resp == nil {
		t.Fatal("expected non-nil response")
	}
	if inner.callCount != 3 {
		t.Errorf("callCount = %d, want 3", inner.callCount)
	}
}

func TestRetryComplete_MaxRetriesExceeded(t *testing.T) {
	inner := &failProvider{failCount: 5, failErr: fmt.Errorf("status 500: internal server error")}
	prov := WithRetry(inner, RetryConfig{MaxRetries: 2, BaseDelay: 1 * time.Millisecond, MaxDelay: 10 * time.Millisecond})

	_, err := prov.Complete(context.Background(), &types.CompletionRequest{})
	if err == nil {
		t.Fatal("expected error after max retries")
	}
	if inner.callCount != 3 { // 1 initial + 2 retries
		t.Errorf("callCount = %d, want 3", inner.callCount)
	}
}

func TestRetryComplete_NonTransientError(t *testing.T) {
	inner := &failProvider{failCount: 5, failErr: fmt.Errorf("status 401: unauthorized")}
	prov := WithRetry(inner, RetryConfig{MaxRetries: 3, BaseDelay: 1 * time.Millisecond, MaxDelay: 10 * time.Millisecond})

	_, err := prov.Complete(context.Background(), &types.CompletionRequest{})
	if err == nil {
		t.Fatal("expected error for non-transient failure")
	}
	if inner.callCount != 1 {
		t.Errorf("callCount = %d, want 1 (no retries for non-transient)", inner.callCount)
	}
}

func TestRetryStream_TransientThenSuccess(t *testing.T) {
	inner := &failProvider{failCount: 1, failErr: fmt.Errorf("status 429: rate limited")}
	prov := WithRetry(inner, RetryConfig{MaxRetries: 2, BaseDelay: 1 * time.Millisecond, MaxDelay: 10 * time.Millisecond})

	ch, err := prov.Stream(context.Background(), &types.CompletionRequest{})
	if err != nil {
		t.Fatalf("expected success, got: %v", err)
	}
	if ch == nil {
		t.Fatal("expected non-nil channel")
	}
	if inner.callCount != 2 {
		t.Errorf("callCount = %d, want 2", inner.callCount)
	}
}

func TestRetryComplete_ContextCancelled(t *testing.T) {
	inner := &failProvider{failCount: 10, failErr: fmt.Errorf("status 503: unavailable")}
	prov := WithRetry(inner, RetryConfig{MaxRetries: 5, BaseDelay: 100 * time.Millisecond, MaxDelay: 1 * time.Second})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err := prov.Complete(ctx, &types.CompletionRequest{})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestRetryZeroRetries(t *testing.T) {
	inner := &failProvider{failCount: 0}
	prov := WithRetry(inner, RetryConfig{MaxRetries: 0})

	// Should return inner directly (no wrapping)
	if _, ok := prov.(*retryProvider); ok {
		t.Error("expected unwrapped provider when MaxRetries=0")
	}
}

func TestIsTransient(t *testing.T) {
	tests := []struct {
		err       error
		transient bool
	}{
		{fmt.Errorf("status 500: internal error"), true},
		{fmt.Errorf("status 502: bad gateway"), true},
		{fmt.Errorf("status 503: service unavailable"), true},
		{fmt.Errorf("status 429: rate limit exceeded"), true},
		{fmt.Errorf("connection refused"), true},
		{fmt.Errorf("timeout"), true},
		{fmt.Errorf("status 401: unauthorized"), false},
		{fmt.Errorf("status 400: bad request"), false},
		{fmt.Errorf("invalid model"), false},
		{nil, false},
	}

	for _, tt := range tests {
		got := isTransient(tt.err)
		if got != tt.transient {
			t.Errorf("isTransient(%v) = %v, want %v", tt.err, got, tt.transient)
		}
	}
}
