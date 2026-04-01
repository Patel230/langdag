package mock

import (
	"context"
	"errors"
	"testing"

	"langdag.com/langdag/types"
)

func collectEvents(ch <-chan types.StreamEvent) []types.StreamEvent {
	var events []types.StreamEvent
	for ev := range ch {
		events = append(events, ev)
	}
	return events
}

func TestErrorMode_Complete(t *testing.T) {
	errSim := errors.New("simulated provider failure")
	p := New(Config{Mode: "error", Error: errSim})

	_, err := p.Complete(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
	if !errors.Is(err, errSim) {
		t.Fatalf("expected simulated error, got %v", err)
	}
}

func TestErrorMode_Stream(t *testing.T) {
	errSim := errors.New("simulated provider failure")
	p := New(Config{Mode: "error", Error: errSim})

	_, err := p.Stream(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
	if !errors.Is(err, errSim) {
		t.Fatalf("expected simulated error, got %v", err)
	}
}

func TestStreamErrorMode(t *testing.T) {
	errSim := errors.New("mid-stream crash")
	p := New(Config{
		Mode:             "stream_error",
		FixedResponse:    "one two three four five",
		ErrorAfterChunks: 3,
		Error:            errSim,
	})

	ch, err := p.Stream(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
	if err != nil {
		t.Fatalf("Stream() returned error: %v", err)
	}

	events := collectEvents(ch)

	// Expect: start, 3 deltas, error
	if len(events) != 5 {
		t.Fatalf("expected 5 events (start + 3 deltas + error), got %d: %+v", len(events), events)
	}
	if events[0].Type != types.StreamEventStart {
		t.Errorf("events[0] type = %s, want start", events[0].Type)
	}
	for i := 1; i <= 3; i++ {
		if events[i].Type != types.StreamEventDelta {
			t.Errorf("events[%d] type = %s, want delta", i, events[i].Type)
		}
	}
	if events[4].Type != types.StreamEventError {
		t.Errorf("events[4] type = %s, want error", events[4].Type)
	}
	if !errors.Is(events[4].Error, errSim) {
		t.Errorf("events[4] error = %v, want %v", events[4].Error, errSim)
	}
}

func TestStreamErrorMode_ZeroChunks(t *testing.T) {
	errSim := errors.New("immediate stream error")
	p := New(Config{
		Mode:             "stream_error",
		ErrorAfterChunks: 0,
		Error:            errSim,
	})

	ch, err := p.Stream(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
	if err != nil {
		t.Fatalf("Stream() returned error: %v", err)
	}

	events := collectEvents(ch)
	// Expect: start, error (no deltas)
	if len(events) != 2 {
		t.Fatalf("expected 2 events (start + error), got %d", len(events))
	}
	if events[0].Type != types.StreamEventStart {
		t.Errorf("events[0] type = %s, want start", events[0].Type)
	}
	if events[1].Type != types.StreamEventError {
		t.Errorf("events[1] type = %s, want error", events[1].Type)
	}
}

func TestPartialMaxTokensMode_EmptyContent(t *testing.T) {
	p := New(Config{Mode: "partial_max_tokens"})

	resp, err := p.Complete(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
	if resp.StopReason != "max_tokens" {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, "max_tokens")
	}
}

func TestPartialMaxTokensMode_WithContent(t *testing.T) {
	p := New(Config{
		Mode:          "partial_max_tokens",
		FixedResponse: "partial content here",
	})

	resp, err := p.Complete(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
	if resp.StopReason != "max_tokens" {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, "max_tokens")
	}
	if len(resp.Content) == 0 || resp.Content[0].Text != "partial content here" {
		t.Errorf("expected content 'partial content here', got %+v", resp.Content)
	}
}

func TestPartialMaxTokensMode_Stream(t *testing.T) {
	p := New(Config{
		Mode:          "partial_max_tokens",
		FixedResponse: "partial output",
	})

	ch, err := p.Stream(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
	if err != nil {
		t.Fatalf("Stream() error: %v", err)
	}

	events := collectEvents(ch)
	last := events[len(events)-1]
	if last.Type != types.StreamEventDone {
		t.Fatalf("last event type = %s, want done", last.Type)
	}
	if last.Response.StopReason != "max_tokens" {
		t.Errorf("StopReason = %q, want %q", last.Response.StopReason, "max_tokens")
	}
}

func TestPartialMaxTokensMode_StopReasonOverride(t *testing.T) {
	p := New(Config{
		Mode:       "partial_max_tokens",
		StopReason: "custom_stop",
	})

	resp, err := p.Complete(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}
	// Explicit StopReason in config should take precedence
	if resp.StopReason != "custom_stop" {
		t.Errorf("StopReason = %q, want %q", resp.StopReason, "custom_stop")
	}
}

func TestFailUntilCall_Complete(t *testing.T) {
	errTransient := errors.New("transient failure")
	p := New(Config{
		Mode:          "fixed",
		FixedResponse: "success",
		FailUntilCall: 2,
		Error:         errTransient,
	})

	// Call 1: should fail
	_, err := p.Complete(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
	if !errors.Is(err, errTransient) {
		t.Fatalf("call 1: expected transient error, got %v", err)
	}

	// Call 2: should fail
	_, err = p.Complete(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
	if !errors.Is(err, errTransient) {
		t.Fatalf("call 2: expected transient error, got %v", err)
	}

	// Call 3: should succeed
	resp, err := p.Complete(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
	if err != nil {
		t.Fatalf("call 3: unexpected error: %v", err)
	}
	if resp.Content[0].Text != "success" {
		t.Errorf("call 3: content = %q, want %q", resp.Content[0].Text, "success")
	}

	if p.CallCount() != 3 {
		t.Errorf("CallCount() = %d, want 3", p.CallCount())
	}
}

func TestFailUntilCall_Stream(t *testing.T) {
	errTransient := errors.New("transient failure")
	p := New(Config{
		Mode:          "fixed",
		FixedResponse: "streamed success",
		FailUntilCall: 1,
		Error:         errTransient,
	})

	// Call 1: should fail
	_, err := p.Stream(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
	if !errors.Is(err, errTransient) {
		t.Fatalf("call 1: expected transient error, got %v", err)
	}

	// Call 2: should succeed
	ch, err := p.Stream(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
	if err != nil {
		t.Fatalf("call 2: unexpected error: %v", err)
	}

	events := collectEvents(ch)
	last := events[len(events)-1]
	if last.Type != types.StreamEventDone {
		t.Fatalf("last event type = %s, want done", last.Type)
	}
	if last.Response.Content[0].Text != "streamed success" {
		t.Errorf("content = %q, want %q", last.Response.Content[0].Text, "streamed success")
	}
}

func TestFailUntilCall_LastRequestCaptured(t *testing.T) {
	errTransient := errors.New("transient failure")
	p := New(Config{
		Mode:          "fixed",
		FixedResponse: "ok",
		FailUntilCall: 1,
		Error:         errTransient,
	})

	req := &types.CompletionRequest{Model: "mock-fast"}
	_, _ = p.Complete(context.Background(), req) // fails but should capture request

	if p.LastRequest != req {
		t.Error("LastRequest not captured during failing call")
	}
}

func TestExistingModes_Unaffected(t *testing.T) {
	// Verify existing modes still work with the new fields present (zero values)
	t.Run("random", func(t *testing.T) {
		p := New(Config{})
		resp, err := p.Complete(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.StopReason != "end_turn" {
			t.Errorf("StopReason = %q, want %q", resp.StopReason, "end_turn")
		}
	})

	t.Run("fixed", func(t *testing.T) {
		p := New(Config{Mode: "fixed", FixedResponse: "hello"})
		resp, err := p.Complete(context.Background(), &types.CompletionRequest{Model: "mock-fast"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Content[0].Text != "hello" {
			t.Errorf("content = %q, want %q", resp.Content[0].Text, "hello")
		}
	})

	t.Run("echo", func(t *testing.T) {
		p := New(Config{Mode: "echo"})
		msg := `"test message"`
		resp, err := p.Complete(context.Background(), &types.CompletionRequest{
			Model:    "mock-fast",
			Messages: []types.Message{{Role: "user", Content: []byte(msg)}},
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.Content[0].Text != "test message" {
			t.Errorf("content = %q, want %q", resp.Content[0].Text, "test message")
		}
	})
}
