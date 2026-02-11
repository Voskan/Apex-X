package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

type loadRecordingAdapter struct {
	mu         sync.Mutex
	batchSizes []int
	delay      time.Duration
}

func (a *loadRecordingAdapter) Name() string { return "integration-mock-adapter" }

func (a *loadRecordingAdapter) PredictBatch(
	_ context.Context,
	reqs []PredictRequest,
) ([]PredictResponse, error) {
	if a.delay > 0 {
		time.Sleep(a.delay)
	}
	a.mu.Lock()
	a.batchSizes = append(a.batchSizes, len(reqs))
	a.mu.Unlock()

	out := make([]PredictResponse, len(reqs))
	for idx, req := range reqs {
		out[idx] = PredictResponse{
			RequestID:     req.RequestID,
			BudgetProfile: req.BudgetProfile,
			SelectedTiles: selectedTilesForProfile(req.BudgetProfile),
			Scores:        []float32{mean(req.Input)},
			Backend:       a.Name(),
		}
	}
	return out, nil
}

func (a *loadRecordingAdapter) Close() error { return nil }

func (a *loadRecordingAdapter) SnapshotBatchSizes() []int {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]int, len(a.batchSizes))
	copy(out, a.batchSizes)
	return out
}

func TestHTTPServerIntegrationPredictAndBatching(t *testing.T) {
	adapter := &loadRecordingAdapter{delay: 8 * time.Millisecond}
	svc, err := NewHTTPService(adapter, HTTPServiceConfig{
		DefaultBudgetProfile: BudgetProfileBalanced,
		MaxBatchSize:         8,
		BatchWindow:          20 * time.Millisecond,
		QueueSize:            128,
		Logger:               slog.New(slog.NewJSONHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("NewHTTPService() error = %v", err)
	}
	t.Cleanup(func() {
		_ = svc.Close()
	})

	mux := http.NewServeMux()
	svc.RegisterRoutes(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	client := &http.Client{Timeout: 3 * time.Second}

	healthResp, healthErr := client.Get(server.URL + "/health")
	if healthErr != nil {
		t.Fatalf("GET /health error = %v", healthErr)
	}
	defer func() { _ = healthResp.Body.Close() }()
	if healthResp.StatusCode != http.StatusOK {
		t.Fatalf("unexpected /health status: %d", healthResp.StatusCode)
	}

	singleReq := PredictRequest{
		RequestID:     "single",
		BudgetProfile: BudgetProfileEdge,
		Input:         []float32{0.5, 1.5},
	}
	singleRaw, _ := json.Marshal(singleReq)
	singleResp, singleErr := client.Post(
		server.URL+"/predict",
		"application/json",
		bytes.NewReader(singleRaw),
	)
	if singleErr != nil {
		t.Fatalf("POST /predict single error = %v", singleErr)
	}
	defer func() { _ = singleResp.Body.Close() }()
	if singleResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(singleResp.Body)
		t.Fatalf("unexpected /predict single status: %d body=%s", singleResp.StatusCode, string(body))
	}
	var decoded PredictResponse
	if err := json.NewDecoder(singleResp.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode single response error = %v", err)
	}
	if decoded.RequestID == "" || decoded.Backend == "" || decoded.BudgetProfile == "" {
		t.Fatalf("predict response missing required structured fields: %+v", decoded)
	}
	if decoded.Runtime.ExecutionBackend != decoded.Backend {
		t.Fatalf("runtime backend mismatch: %+v", decoded.Runtime)
	}
	if decoded.Runtime.FallbackPolicy != "strict" {
		t.Fatalf("unexpected runtime fallback policy: %q", decoded.Runtime.FallbackPolicy)
	}
	if decoded.Runtime.LatencyMS.Total < 0.0 ||
		decoded.Runtime.LatencyMS.BackendExecute < 0.0 ||
		decoded.Runtime.LatencyMS.BackendPreflight < 0.0 {
		t.Fatalf("runtime latency must be non-negative: %+v", decoded.Runtime.LatencyMS)
	}

	const parallelRequests = 32
	start := make(chan struct{})
	errCh := make(chan error, parallelRequests)
	var wg sync.WaitGroup
	for idx := 0; idx < parallelRequests; idx++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			reqBody := PredictRequest{
				RequestID:     "load-" + strconv.Itoa(i),
				BudgetProfile: BudgetProfileBalanced,
				Input:         []float32{float32(i), float32(i + 1)},
			}
			raw, _ := json.Marshal(reqBody)
			resp, err := client.Post(server.URL+"/predict", "application/json", bytes.NewReader(raw))
			if err != nil {
				errCh <- err
				return
			}
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != http.StatusOK {
				body, _ := io.ReadAll(resp.Body)
				errCh <- fmt.Errorf("status=%d body=%s", resp.StatusCode, string(body))
				return
			}
			var out PredictResponse
			if decodeErr := json.NewDecoder(resp.Body).Decode(&out); decodeErr != nil {
				errCh <- decodeErr
				return
			}
			if out.RequestID == "" || out.Backend == "" {
				errCh <- fmt.Errorf("invalid predict response: %+v", out)
				return
			}
			if out.Runtime.ExecutionBackend != out.Backend {
				errCh <- fmt.Errorf("runtime backend mismatch: %+v", out.Runtime)
				return
			}
			if out.Runtime.FallbackPolicy == "" {
				errCh <- fmt.Errorf("runtime fallback policy missing: %+v", out.Runtime)
				return
			}
		}(idx)
	}
	close(start)
	wg.Wait()
	close(errCh)
	for reqErr := range errCh {
		t.Fatalf("load request failed: %v", reqErr)
	}

	sizes := adapter.SnapshotBatchSizes()
	foundCombinedBatch := false
	for _, size := range sizes {
		if size > 1 {
			foundCombinedBatch = true
			break
		}
	}
	if !foundCombinedBatch {
		t.Fatalf("expected combined batches under load; got sizes=%v", sizes)
	}

	metricsResp, metricsErr := client.Get(server.URL + "/metrics")
	if metricsErr != nil {
		t.Fatalf("GET /metrics error = %v", metricsErr)
	}
	defer func() { _ = metricsResp.Body.Close() }()
	metricsBody, _ := io.ReadAll(metricsResp.Body)
	metricsText := string(metricsBody)
	requiredKeys := []string{
		"apexx_batches_total",
		"apexx_batch_size_avg",
		"apexx_queue_latency_ms_avg",
		"apexx_inference_latency_ms_avg",
		"apexx_batch_errors_total",
		"apexx_canary_samples_total",
		"apexx_canary_compares_total",
		"apexx_canary_mismatches_total",
		"apexx_canary_errors_total",
	}
	for _, key := range requiredKeys {
		if !strings.Contains(metricsText, key) {
			t.Fatalf("missing metrics key %q in metrics output", key)
		}
	}
}
