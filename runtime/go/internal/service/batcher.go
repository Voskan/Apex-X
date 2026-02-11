package service

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"sync"
	"time"
)

var (
	ErrQueueFull      = errors.New("request queue is full")
	ErrBatcherStopped = errors.New("batcher is stopped")
)

type batchItem struct {
	ctx      context.Context
	req      PredictRequest
	enqueued time.Time
	result   chan batchResult
}

type batchResult struct {
	response PredictResponse
	err      error
}

type BatcherConfig struct {
	MaxBatchSize int
	BatchWindow  time.Duration
	QueueSize    int
	OnBatch      func(batchSize int, avgQueueWait time.Duration, inferenceTime time.Duration, err error)
	Logger       *slog.Logger
	Hooks        TelemetryHooks
}

type Batcher struct {
	adapter InferenceAdapter
	cfg     BatcherConfig

	queue  chan batchItem
	stop   chan struct{}
	wg     sync.WaitGroup
	logger *slog.Logger
	hooks  TelemetryHooks
}

func NewBatcher(adapter InferenceAdapter, cfg BatcherConfig) (*Batcher, error) {
	if adapter == nil {
		return nil, errors.New("adapter must not be nil")
	}
	if cfg.MaxBatchSize <= 0 {
		return nil, fmt.Errorf("max batch size must be > 0")
	}
	if cfg.BatchWindow <= 0 {
		return nil, fmt.Errorf("batch window must be > 0")
	}
	if cfg.QueueSize <= 0 {
		cfg.QueueSize = 256
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.New(slog.NewJSONHandler(io.Discard, nil))
	}
	hooks := cfg.Hooks
	if hooks == nil {
		hooks = NopTelemetryHooks{}
	}
	return &Batcher{
		adapter: adapter,
		cfg:     cfg,
		queue:   make(chan batchItem, cfg.QueueSize),
		stop:    make(chan struct{}),
		logger:  logger,
		hooks:   hooks,
	}, nil
}

func (b *Batcher) Start() {
	b.wg.Add(1)
	go func() {
		defer b.wg.Done()
		b.run()
	}()
}

func (b *Batcher) Stop() {
	close(b.stop)
	b.wg.Wait()
}

func (b *Batcher) Submit(ctx context.Context, req PredictRequest) (PredictResponse, error) {
	resultCh := make(chan batchResult, 1)
	item := batchItem{
		ctx:      ctx,
		req:      req,
		enqueued: time.Now(),
		result:   resultCh,
	}

	select {
	case <-ctx.Done():
		return PredictResponse{}, ctx.Err()
	case <-b.stop:
		return PredictResponse{}, ErrBatcherStopped
	default:
	}

	select {
	case b.queue <- item:
	default:
		return PredictResponse{}, ErrQueueFull
	}

	select {
	case result := <-resultCh:
		return result.response, result.err
	case <-ctx.Done():
		return PredictResponse{}, ctx.Err()
	case <-b.stop:
		return PredictResponse{}, ErrBatcherStopped
	}
}

func (b *Batcher) run() {
	for {
		select {
		case <-b.stop:
			return
		case first := <-b.queue:
			b.processBatch(first)
		}
	}
}

func (b *Batcher) processBatch(first batchItem) {
	batch := []batchItem{first}
	timer := time.NewTimer(b.cfg.BatchWindow)
	defer timer.Stop()

collectLoop:
	for len(batch) < b.cfg.MaxBatchSize {
		select {
		case <-b.stop:
			b.failAll(batch, ErrBatcherStopped)
			return
		case next := <-b.queue:
			batch = append(batch, next)
		case <-timer.C:
			break collectLoop
		}
	}

	requests := make([]PredictRequest, len(batch))
	for idx, item := range batch {
		requests[idx] = item.req
	}

	batchStart := time.Now()
	queueWaitTotal := time.Duration(0)
	for _, item := range batch {
		wait := batchStart.Sub(item.enqueued)
		if wait < 0 {
			wait = 0
		}
		queueWaitTotal += wait
	}
	avgQueueWait := time.Duration(0)
	if len(batch) > 0 {
		avgQueueWait = queueWaitTotal / time.Duration(len(batch))
	}

	inferenceStart := time.Now()
	responses, err := b.adapter.PredictBatch(context.Background(), requests)
	inferenceTime := time.Since(inferenceStart)
	if b.cfg.OnBatch != nil {
		b.cfg.OnBatch(len(batch), avgQueueWait, inferenceTime, err)
	}
	b.hooks.OnBatch(context.Background(), len(batch), avgQueueWait, inferenceTime, err)

	if err != nil {
		b.logger.Error(
			"batch_inference_failed",
			"adapter", b.adapter.Name(),
			"batch_size", len(batch),
			"queue_wait_ms", avgQueueWait.Seconds()*1000.0,
			"inference_ms", inferenceTime.Seconds()*1000.0,
			"error", err.Error(),
		)
		b.failAll(batch, err)
		return
	}
	if len(responses) != len(batch) {
		mismatchErr := fmt.Errorf(
			"adapter returned %d responses for %d requests",
			len(responses),
			len(batch),
		)
		b.logger.Error(
			"batch_inference_failed",
			"adapter", b.adapter.Name(),
			"batch_size", len(batch),
			"queue_wait_ms", avgQueueWait.Seconds()*1000.0,
			"inference_ms", inferenceTime.Seconds()*1000.0,
			"error", mismatchErr.Error(),
		)
		b.failAll(batch, mismatchErr)
		return
	}
	b.logger.Info(
		"batch_inference_done",
		"adapter", b.adapter.Name(),
		"batch_size", len(batch),
		"queue_wait_ms", avgQueueWait.Seconds()*1000.0,
		"inference_ms", inferenceTime.Seconds()*1000.0,
	)
	for idx := range batch {
		queueWait := batchStart.Sub(batch[idx].enqueued)
		if queueWait < 0 {
			queueWait = 0
		}
		response := responses[idx]
		req := batch[idx].req
		executionBackend := strings.TrimSpace(response.Backend)
		if executionBackend == "" {
			executionBackend = b.adapter.Name()
			response.Backend = executionBackend
		}
		if strings.TrimSpace(response.BudgetProfile) == "" {
			profile, profileErr := normalizeBudgetProfile(req.BudgetProfile, BudgetProfileBalanced)
			if profileErr != nil {
				profile = BudgetProfileBalanced
			}
			response.BudgetProfile = profile
		}
		response.Runtime = RuntimeMetadata{
			RequestedBackend:        requestedBackendFromMetadata(req, executionBackend),
			SelectedBackend:         executionBackend,
			ExecutionBackend:        executionBackend,
			FallbackPolicy:          fallbackPolicyFromMetadata(req),
			PrecisionProfile:        precisionProfileForBudget(response.BudgetProfile),
			SelectionFallbackReason: optionalMetadataValue(req, "selection_fallback_reason"),
			ExecutionFallbackReason: optionalMetadataValue(req, "execution_fallback_reason"),
			LatencyMS: RuntimeLatencyMillis{
				Total:            durationMillis(queueWait + inferenceTime),
				BackendExecute:   durationMillis(inferenceTime),
				BackendPreflight: durationMillis(queueWait),
			},
		}
		batch[idx].result <- batchResult{response: response}
	}
}

func (b *Batcher) failAll(batch []batchItem, err error) {
	for _, item := range batch {
		item.result <- batchResult{err: err}
	}
}

func durationMillis(value time.Duration) float64 {
	if value < 0 {
		return 0.0
	}
	return float64(value) / float64(time.Millisecond)
}

func metadataValue(req PredictRequest, key string) string {
	if req.Metadata == nil {
		return ""
	}
	return strings.TrimSpace(req.Metadata[key])
}

func optionalMetadataValue(req PredictRequest, key string) *string {
	value := metadataValue(req, key)
	if value == "" || strings.EqualFold(value, "none") {
		return nil
	}
	cleaned := value
	return &cleaned
}

func requestedBackendFromMetadata(req PredictRequest, executionBackend string) string {
	requested := metadataValue(req, "requested_backend")
	if requested == "" {
		requested = metadataValue(req, "backend")
	}
	if requested == "" {
		return strings.ToLower(strings.TrimSpace(executionBackend))
	}
	return strings.ToLower(requested)
}

func fallbackPolicyFromMetadata(req PredictRequest) string {
	policy := strings.ToLower(metadataValue(req, "fallback_policy"))
	if policy == "strict" || policy == "permissive" {
		return policy
	}
	return "strict"
}

func precisionProfileForBudget(profile string) string {
	normalized := strings.ToLower(strings.TrimSpace(profile))
	if _, ok := budgetProfiles[normalized]; ok {
		return normalized
	}
	return BudgetProfileBalanced
}
