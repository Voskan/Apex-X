package service

import (
	"context"
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
