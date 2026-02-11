package service

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

type recordingAdapter struct {
	mu         sync.Mutex
	batchSizes []int
}

func (a *recordingAdapter) Name() string { return "recording" }

func (a *recordingAdapter) PredictBatch(
	_ context.Context,
	reqs []PredictRequest,
) ([]PredictResponse, error) {
	a.mu.Lock()
	a.batchSizes = append(a.batchSizes, len(reqs))
	a.mu.Unlock()

	responses := make([]PredictResponse, len(reqs))
	for idx, req := range reqs {
		responses[idx] = PredictResponse{
			RequestID:     req.RequestID,
			BudgetProfile: req.BudgetProfile,
			Backend:       a.Name(),
		}
	}
	return responses, nil
}

func (a *recordingAdapter) Close() error { return nil }

func (a *recordingAdapter) Sizes() []int {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]int, len(a.batchSizes))
	copy(out, a.batchSizes)
	return out
}

func TestBatcherRespectsMaxBatchSize(t *testing.T) {
	adapter := &recordingAdapter{}
	batcher, err := NewBatcher(adapter, BatcherConfig{
		MaxBatchSize: 2,
		BatchWindow:  20 * time.Millisecond,
		QueueSize:    16,
	})
	if err != nil {
		t.Fatalf("NewBatcher() error = %v", err)
	}
	batcher.Start()
	defer batcher.Stop()

	var wg sync.WaitGroup
	for idx := range 5 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, submitErr := batcher.Submit(context.Background(), PredictRequest{
				RequestID:     "req",
				BudgetProfile: BudgetProfileBalanced,
				Input:         []float32{float32(i)},
			})
			if submitErr != nil {
				t.Errorf("Submit() error = %v", submitErr)
			}
		}(idx)
	}
	wg.Wait()

	for _, size := range adapter.Sizes() {
		if size > 2 {
			t.Fatalf("batch size %d exceeds max", size)
		}
	}
}

func TestBatcherBatchesNearbyRequests(t *testing.T) {
	adapter := &recordingAdapter{}
	batcher, err := NewBatcher(adapter, BatcherConfig{
		MaxBatchSize: 8,
		BatchWindow:  25 * time.Millisecond,
		QueueSize:    16,
	})
	if err != nil {
		t.Fatalf("NewBatcher() error = %v", err)
	}
	batcher.Start()
	defer batcher.Stop()

	var wg sync.WaitGroup
	for idx := range 4 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, submitErr := batcher.Submit(context.Background(), PredictRequest{
				RequestID:     "req",
				BudgetProfile: BudgetProfileBalanced,
				Input:         []float32{float32(i)},
			})
			if submitErr != nil {
				t.Errorf("Submit() error = %v", submitErr)
			}
		}(idx)
	}
	wg.Wait()

	sizes := adapter.Sizes()
	foundCombinedBatch := false
	for _, size := range sizes {
		if size >= 2 {
			foundCombinedBatch = true
			break
		}
	}
	if !foundCombinedBatch {
		t.Fatalf("expected at least one combined batch, got sizes=%v", sizes)
	}
}

func TestBatcherInjectsRuntimeTelemetry(t *testing.T) {
	adapter := &recordingAdapter{}
	batcher, err := NewBatcher(adapter, BatcherConfig{
		MaxBatchSize: 2,
		BatchWindow:  5 * time.Millisecond,
		QueueSize:    8,
	})
	if err != nil {
		t.Fatalf("NewBatcher() error = %v", err)
	}
	batcher.Start()
	defer batcher.Stop()

	response, submitErr := batcher.Submit(context.Background(), PredictRequest{
		RequestID:     "req-telemetry",
		BudgetProfile: BudgetProfileQuality,
		Input:         []float32{1.0, 2.0},
		Metadata: map[string]string{
			"requested_backend":         "tensorrt",
			"fallback_policy":           "permissive",
			"selection_fallback_reason": "triton_not_installed",
		},
	})
	if submitErr != nil {
		t.Fatalf("Submit() error = %v", submitErr)
	}

	if response.Runtime.RequestedBackend != "tensorrt" {
		t.Fatalf("unexpected requested backend: %q", response.Runtime.RequestedBackend)
	}
	if response.Runtime.SelectedBackend != adapter.Name() {
		t.Fatalf("unexpected selected backend: %q", response.Runtime.SelectedBackend)
	}
	if response.Runtime.ExecutionBackend != adapter.Name() {
		t.Fatalf("unexpected execution backend: %q", response.Runtime.ExecutionBackend)
	}
	if response.Runtime.FallbackPolicy != "permissive" {
		t.Fatalf("unexpected fallback policy: %q", response.Runtime.FallbackPolicy)
	}
	if response.Runtime.SelectionFallbackReason == nil ||
		*response.Runtime.SelectionFallbackReason != "triton_not_installed" {
		t.Fatalf("unexpected selection fallback reason: %+v", response.Runtime.SelectionFallbackReason)
	}
	if response.Runtime.ExecutionFallbackReason != nil {
		t.Fatalf("execution fallback reason should be nil, got %+v", response.Runtime.ExecutionFallbackReason)
	}
	if response.Runtime.PrecisionProfile != BudgetProfileQuality {
		t.Fatalf("unexpected precision profile: %q", response.Runtime.PrecisionProfile)
	}
	if response.Runtime.LatencyMS.Total < 0.0 ||
		response.Runtime.LatencyMS.BackendExecute < 0.0 ||
		response.Runtime.LatencyMS.BackendPreflight < 0.0 {
		t.Fatalf("latency fields must be non-negative: %+v", response.Runtime.LatencyMS)
	}
}

func TestBatcherSubmitReturnsQueueFullWhenQueueSaturated(t *testing.T) {
	adapter := &recordingAdapter{}
	batcher, err := NewBatcher(adapter, BatcherConfig{
		MaxBatchSize: 1,
		BatchWindow:  20 * time.Millisecond,
		QueueSize:    1,
	})
	if err != nil {
		t.Fatalf("NewBatcher() error = %v", err)
	}

	batcher.queue <- batchItem{
		req:      PredictRequest{RequestID: "occupied"},
		enqueued: time.Now(),
		result:   make(chan batchResult, 1),
	}

	_, submitErr := batcher.Submit(context.Background(), PredictRequest{
		RequestID:     "overflow",
		BudgetProfile: BudgetProfileBalanced,
	})
	if !errors.Is(submitErr, ErrQueueFull) {
		t.Fatalf("expected ErrQueueFull, got %v", submitErr)
	}
}
