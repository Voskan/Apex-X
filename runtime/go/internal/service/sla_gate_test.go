package service

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"sort"
	"strconv"
	"sync"
	"testing"
	"time"
)

type loadScenarioResult struct {
	statusCounts map[int]int
	latencyMS    []float64
}

func envFloatOrDefault(key string, fallback float64) float64 {
	raw := os.Getenv(key)
	if raw == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return fallback
	}
	return parsed
}

func envIntOrDefault(key string, fallback int) int {
	raw := os.Getenv(key)
	if raw == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return parsed
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0.0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	rank := int(math.Ceil((p / 100.0) * float64(len(sorted))))
	if rank <= 0 {
		rank = 1
	}
	if rank > len(sorted) {
		rank = len(sorted)
	}
	return sorted[rank-1]
}

func runPredictLoadScenario(
	t *testing.T,
	adapter InferenceAdapter,
	cfg HTTPServiceConfig,
	totalRequests int,
	concurrency int,
) loadScenarioResult {
	t.Helper()

	svc, err := NewHTTPService(adapter, cfg)
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
	sem := make(chan struct{}, concurrency)
	result := loadScenarioResult{
		statusCounts: make(map[int]int),
		latencyMS:    make([]float64, 0, totalRequests),
	}
	var mu sync.Mutex
	var wg sync.WaitGroup

	for idx := 0; idx < totalRequests; idx++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() {
				<-sem
			}()

			reqBody := PredictRequest{
				RequestID:     fmt.Sprintf("sla-gate-%d", i),
				BudgetProfile: BudgetProfileBalanced,
				Input:         []float32{float32(i), float32(i + 1)},
			}
			raw, _ := json.Marshal(reqBody)
			start := time.Now()
			resp, reqErr := client.Post(
				server.URL+"/predict",
				"application/json",
				bytes.NewReader(raw),
			)
			elapsedMS := float64(time.Since(start)) / float64(time.Millisecond)
			if reqErr != nil {
				mu.Lock()
				result.statusCounts[0]++
				mu.Unlock()
				return
			}
			_, _ = io.ReadAll(resp.Body)
			_ = resp.Body.Close()

			mu.Lock()
			result.statusCounts[resp.StatusCode]++
			if resp.StatusCode == http.StatusOK {
				result.latencyMS = append(result.latencyMS, elapsedMS)
			}
			mu.Unlock()
		}(idx)
	}
	wg.Wait()
	return result
}

func assertSLAStatusThresholds(
	t *testing.T,
	label string,
	result loadScenarioResult,
	totalRequests int,
	maxTimeoutRate float64,
	maxQueueOverflowRate float64,
) {
	t.Helper()
	transportErrors := result.statusCounts[0]
	if transportErrors > 0 {
		t.Fatalf("%s transport errors=%d", label, transportErrors)
	}
	successCount := result.statusCounts[http.StatusOK]
	timeoutCount := result.statusCounts[http.StatusGatewayTimeout] + result.statusCounts[http.StatusRequestTimeout]
	queueOverflowCount := result.statusCounts[http.StatusTooManyRequests]
	otherFailures := totalRequests - successCount - timeoutCount - queueOverflowCount
	if otherFailures > 0 {
		t.Fatalf("%s unexpected failure statuses: %+v", label, result.statusCounts)
	}
	timeoutRate := float64(timeoutCount) / float64(totalRequests)
	if timeoutRate > maxTimeoutRate {
		t.Fatalf(
			"%s timeout rate %.4f exceeded threshold %.4f (counts=%+v)",
			label,
			timeoutRate,
			maxTimeoutRate,
			result.statusCounts,
		)
	}
	queueOverflowRate := float64(queueOverflowCount) / float64(totalRequests)
	if queueOverflowRate > maxQueueOverflowRate {
		t.Fatalf(
			"%s queue overflow rate %.4f exceeded threshold %.4f (counts=%+v)",
			label,
			queueOverflowRate,
			maxQueueOverflowRate,
			result.statusCounts,
		)
	}
	if successCount == 0 {
		t.Fatalf("%s expected at least one successful response", label)
	}
}

func TestCanaryLoadGateThresholds(t *testing.T) {
	totalRequests := envIntOrDefault("APEXX_GO_CANARY_GATE_REQUESTS", 96)
	concurrency := envIntOrDefault("APEXX_GO_CANARY_GATE_CONCURRENCY", 12)
	primaryDelayMS := envIntOrDefault("APEXX_GO_CANARY_GATE_PRIMARY_DELAY_MS", 3)
	canaryDelayMS := envIntOrDefault("APEXX_GO_CANARY_GATE_CANARY_DELAY_MS", 3)
	maxOverheadRatio := envFloatOrDefault("APEXX_GO_CANARY_GATE_MAX_OVERHEAD_RATIO", 0.75)
	maxOverheadAbsMS := envFloatOrDefault("APEXX_GO_CANARY_GATE_MAX_OVERHEAD_ABS_MS", 5.0)
	maxTimeoutRate := envFloatOrDefault("APEXX_GO_CANARY_GATE_MAX_TIMEOUT_RATE", 0.0)
	maxQueueOverflowRate := envFloatOrDefault("APEXX_GO_CANARY_GATE_MAX_QUEUE_OVERFLOW_RATE", 0.0)

	if totalRequests <= 0 {
		t.Fatalf("APEXX_GO_CANARY_GATE_REQUESTS must be > 0")
	}
	if concurrency <= 0 {
		t.Fatalf("APEXX_GO_CANARY_GATE_CONCURRENCY must be > 0")
	}
	if maxOverheadRatio < 0.0 {
		t.Fatalf("APEXX_GO_CANARY_GATE_MAX_OVERHEAD_RATIO must be >= 0")
	}
	if maxOverheadAbsMS < 0.0 {
		t.Fatalf("APEXX_GO_CANARY_GATE_MAX_OVERHEAD_ABS_MS must be >= 0")
	}
	if maxTimeoutRate < 0.0 || maxTimeoutRate > 1.0 {
		t.Fatalf("APEXX_GO_CANARY_GATE_MAX_TIMEOUT_RATE must be in [0,1]")
	}
	if maxQueueOverflowRate < 0.0 || maxQueueOverflowRate > 1.0 {
		t.Fatalf("APEXX_GO_CANARY_GATE_MAX_QUEUE_OVERFLOW_RATE must be in [0,1]")
	}

	baseConfig := HTTPServiceConfig{
		DefaultBudgetProfile: BudgetProfileBalanced,
		MaxBatchSize:         8,
		BatchWindow:          4 * time.Millisecond,
		QueueSize:            max(totalRequests, 128),
		Logger:               nil,
	}

	baseline := runPredictLoadScenario(
		t,
		&loadRecordingAdapter{delay: time.Duration(primaryDelayMS) * time.Millisecond},
		baseConfig,
		totalRequests,
		concurrency,
	)
	assertSLAStatusThresholds(
		t,
		"baseline",
		baseline,
		totalRequests,
		maxTimeoutRate,
		maxQueueOverflowRate,
	)

	canaryConfig := baseConfig
	canaryConfig.CanaryAdapter = &loadRecordingAdapter{delay: time.Duration(canaryDelayMS) * time.Millisecond}
	canaryConfig.CanarySampleRate = 1.0
	canaryConfig.CanaryScoreAbsTol = 1e-4
	canaryConfig.CanaryTimeout = 200 * time.Millisecond
	canary := runPredictLoadScenario(
		t,
		&loadRecordingAdapter{delay: time.Duration(primaryDelayMS) * time.Millisecond},
		canaryConfig,
		totalRequests,
		concurrency,
	)
	assertSLAStatusThresholds(
		t,
		"canary",
		canary,
		totalRequests,
		maxTimeoutRate,
		maxQueueOverflowRate,
	)

	p95Baseline := percentile(baseline.latencyMS, 95.0)
	p95Canary := percentile(canary.latencyMS, 95.0)
	if p95Baseline <= 0.0 {
		t.Fatalf("baseline p95 latency is non-positive: %.4f", p95Baseline)
	}
	allowedP95 := (p95Baseline * (1.0 + maxOverheadRatio)) + maxOverheadAbsMS
	if p95Canary > allowedP95 {
		t.Fatalf(
			"canary p95 %.4fms exceeded gate %.4fms (baseline=%.4fms ratio=%.3f abs=%.3f)",
			p95Canary,
			allowedP95,
			p95Baseline,
			maxOverheadRatio,
			maxOverheadAbsMS,
		)
	}

	t.Logf(
		"canary_gate baseline_p95_ms=%.4f canary_p95_ms=%.4f allowed_p95_ms=%.4f",
		p95Baseline,
		p95Canary,
		allowedP95,
	)
}

func max(lhs int, rhs int) int {
	if lhs >= rhs {
		return lhs
	}
	return rhs
}
