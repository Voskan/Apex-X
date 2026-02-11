package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type staticAdapter struct{}

func (a *staticAdapter) Name() string { return "static-adapter" }

func (a *staticAdapter) PredictBatch(
	_ context.Context,
	reqs []PredictRequest,
) ([]PredictResponse, error) {
	responses := make([]PredictResponse, len(reqs))
	for idx, req := range reqs {
		responses[idx] = PredictResponse{
			RequestID:     req.RequestID,
			BudgetProfile: req.BudgetProfile,
			SelectedTiles: selectedTilesForProfile(req.BudgetProfile),
			Scores:        []float32{1.0},
			Backend:       a.Name(),
		}
	}
	return responses, nil
}

func (a *staticAdapter) Close() error { return nil }

type delayedAdapter struct {
	delay time.Duration
}

func (a *delayedAdapter) Name() string { return "delayed-adapter" }

func (a *delayedAdapter) PredictBatch(
	_ context.Context,
	reqs []PredictRequest,
) ([]PredictResponse, error) {
	time.Sleep(a.delay)
	responses := make([]PredictResponse, len(reqs))
	for idx, req := range reqs {
		responses[idx] = PredictResponse{
			RequestID:     req.RequestID,
			BudgetProfile: req.BudgetProfile,
			SelectedTiles: selectedTilesForProfile(req.BudgetProfile),
			Scores:        []float32{1.0},
			Backend:       a.Name(),
		}
	}
	return responses, nil
}

func (a *delayedAdapter) Close() error { return nil }

type mismatchCanaryAdapter struct{}

func (a *mismatchCanaryAdapter) Name() string { return "canary-mismatch-adapter" }

func (a *mismatchCanaryAdapter) PredictBatch(
	_ context.Context,
	reqs []PredictRequest,
) ([]PredictResponse, error) {
	responses := make([]PredictResponse, len(reqs))
	for idx, req := range reqs {
		responses[idx] = PredictResponse{
			RequestID:     req.RequestID,
			BudgetProfile: req.BudgetProfile,
			SelectedTiles: selectedTilesForProfile(req.BudgetProfile) + 1,
			Scores:        []float32{9.0},
			Backend:       a.Name(),
		}
	}
	return responses, nil
}

func (a *mismatchCanaryAdapter) Close() error { return nil }

type failingAdapter struct {
	err error
}

func (a *failingAdapter) Name() string { return "failing-adapter" }

func (a *failingAdapter) PredictBatch(
	_ context.Context,
	_ []PredictRequest,
) ([]PredictResponse, error) {
	return nil, a.err
}

func (a *failingAdapter) Close() error { return nil }

func waitForFileContent(path string, timeout time.Duration) (string, error) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		raw, err := os.ReadFile(path)
		if err == nil && len(raw) > 0 {
			return string(raw), nil
		}
		time.Sleep(10 * time.Millisecond)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(raw), nil
}

func TestHTTPHandlers(t *testing.T) {
	svc, err := NewHTTPService(&staticAdapter{}, HTTPServiceConfig{
		DefaultBudgetProfile: BudgetProfileBalanced,
		MaxBatchSize:         8,
		BatchWindow:          10 * time.Millisecond,
		QueueSize:            32,
	})
	if err != nil {
		t.Fatalf("NewHTTPService() error = %v", err)
	}
	defer func() {
		_ = svc.Close()
	}()

	mux := http.NewServeMux()
	svc.RegisterRoutes(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	t.Run("health", func(t *testing.T) {
		resp, getErr := http.Get(server.URL + "/health")
		if getErr != nil {
			t.Fatalf("GET /health error = %v", getErr)
		}
		defer func() {
			_ = resp.Body.Close()
		}()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("unexpected /health status: %d", resp.StatusCode)
		}
	})

	t.Run("predict", func(t *testing.T) {
		payload := PredictRequest{
			RequestID: "abc",
			Input:     []float32{1.0, 3.0},
		}
		raw, _ := json.Marshal(payload)
		resp, postErr := http.Post(
			server.URL+"/predict",
			"application/json",
			bytes.NewReader(raw),
		)
		if postErr != nil {
			t.Fatalf("POST /predict error = %v", postErr)
		}
		defer func() {
			_ = resp.Body.Close()
		}()
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			t.Fatalf("unexpected /predict status: %d body=%s", resp.StatusCode, string(body))
		}
		var decoded PredictResponse
		if decodeErr := json.NewDecoder(resp.Body).Decode(&decoded); decodeErr != nil {
			t.Fatalf("decode response error = %v", decodeErr)
		}
		if decoded.BudgetProfile != BudgetProfileBalanced {
			t.Fatalf("expected default budget profile, got %q", decoded.BudgetProfile)
		}
		if decoded.Runtime.ExecutionBackend != decoded.Backend {
			t.Fatalf(
				"expected runtime execution backend %q, got %q",
				decoded.Backend,
				decoded.Runtime.ExecutionBackend,
			)
		}
		if decoded.Runtime.FallbackPolicy != "strict" {
			t.Fatalf("expected strict fallback policy, got %q", decoded.Runtime.FallbackPolicy)
		}
		if decoded.Runtime.LatencyMS.Total < 0.0 ||
			decoded.Runtime.LatencyMS.BackendExecute < 0.0 ||
			decoded.Runtime.LatencyMS.BackendPreflight < 0.0 {
			t.Fatalf("latency fields must be non-negative: %+v", decoded.Runtime.LatencyMS)
		}
	})

	t.Run("metrics", func(t *testing.T) {
		resp, getErr := http.Get(server.URL + "/metrics")
		if getErr != nil {
			t.Fatalf("GET /metrics error = %v", getErr)
		}
		defer func() {
			_ = resp.Body.Close()
		}()
		body, _ := io.ReadAll(resp.Body)
		content := string(body)
		if !strings.Contains(content, "apexx_requests_total") {
			t.Fatalf("metrics missing requests counter: %s", content)
		}
		if !strings.Contains(content, "apexx_batches_total") {
			t.Fatalf("metrics missing batches counter: %s", content)
		}
	})
}

func TestHTTPPredictTimeoutReturnsGatewayTimeout(t *testing.T) {
	svc, err := NewHTTPService(&delayedAdapter{delay: 50 * time.Millisecond}, HTTPServiceConfig{
		DefaultBudgetProfile: BudgetProfileBalanced,
		MaxBatchSize:         1,
		BatchWindow:          1 * time.Millisecond,
		QueueSize:            8,
		PredictTimeout:       5 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("NewHTTPService() error = %v", err)
	}
	defer func() {
		_ = svc.Close()
	}()

	mux := http.NewServeMux()
	svc.RegisterRoutes(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	payload := PredictRequest{
		RequestID: "timeout-1",
		Input:     []float32{1.0, 2.0},
	}
	raw, _ := json.Marshal(payload)
	resp, postErr := http.Post(
		server.URL+"/predict",
		"application/json",
		bytes.NewReader(raw),
	)
	if postErr != nil {
		t.Fatalf("POST /predict error = %v", postErr)
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusGatewayTimeout {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("unexpected /predict status: %d body=%s", resp.StatusCode, string(body))
	}
}

func TestHTTPPredictBackendUnavailableReturnsServiceUnavailable(t *testing.T) {
	svc, err := NewHTTPService(&failingAdapter{err: ErrBackendUnavailable}, HTTPServiceConfig{
		DefaultBudgetProfile: BudgetProfileBalanced,
		MaxBatchSize:         1,
		BatchWindow:          1 * time.Millisecond,
		QueueSize:            8,
	})
	if err != nil {
		t.Fatalf("NewHTTPService() error = %v", err)
	}
	defer func() {
		_ = svc.Close()
	}()

	mux := http.NewServeMux()
	svc.RegisterRoutes(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	payload := PredictRequest{RequestID: "backend-unavailable", Input: []float32{1.0}}
	raw, _ := json.Marshal(payload)
	resp, postErr := http.Post(
		server.URL+"/predict",
		"application/json",
		bytes.NewReader(raw),
	)
	if postErr != nil {
		t.Fatalf("POST /predict error = %v", postErr)
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusServiceUnavailable {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("unexpected /predict status: %d body=%s", resp.StatusCode, string(body))
	}
}

func TestHTTPPredictBackendInferenceFailureReturnsBadGateway(t *testing.T) {
	svc, err := NewHTTPService(&failingAdapter{err: ErrBackendInference}, HTTPServiceConfig{
		DefaultBudgetProfile: BudgetProfileBalanced,
		MaxBatchSize:         1,
		BatchWindow:          1 * time.Millisecond,
		QueueSize:            8,
	})
	if err != nil {
		t.Fatalf("NewHTTPService() error = %v", err)
	}
	defer func() {
		_ = svc.Close()
	}()

	mux := http.NewServeMux()
	svc.RegisterRoutes(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	payload := PredictRequest{RequestID: "backend-inference", Input: []float32{1.0}}
	raw, _ := json.Marshal(payload)
	resp, postErr := http.Post(
		server.URL+"/predict",
		"application/json",
		bytes.NewReader(raw),
	)
	if postErr != nil {
		t.Fatalf("POST /predict error = %v", postErr)
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusBadGateway {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("unexpected /predict status: %d body=%s", resp.StatusCode, string(body))
	}
}

func TestPredictErrorStatusCodeClassification(t *testing.T) {
	if code := predictErrorStatusCode(ErrQueueFull); code != http.StatusTooManyRequests {
		t.Fatalf("ErrQueueFull expected %d, got %d", http.StatusTooManyRequests, code)
	}
	if code := predictErrorStatusCode(ErrBackendUnavailable); code != http.StatusServiceUnavailable {
		t.Fatalf("ErrBackendUnavailable expected %d, got %d", http.StatusServiceUnavailable, code)
	}
	if code := predictErrorStatusCode(ErrBackendInference); code != http.StatusBadGateway {
		t.Fatalf("ErrBackendInference expected %d, got %d", http.StatusBadGateway, code)
	}
	if code := predictErrorStatusCode(ErrBackendProtocol); code != http.StatusBadGateway {
		t.Fatalf("ErrBackendProtocol expected %d, got %d", http.StatusBadGateway, code)
	}
	if code := predictErrorStatusCode(context.DeadlineExceeded); code != http.StatusGatewayTimeout {
		t.Fatalf("DeadlineExceeded expected %d, got %d", http.StatusGatewayTimeout, code)
	}
	if code := predictErrorStatusCode(context.Canceled); code != http.StatusRequestTimeout {
		t.Fatalf("Canceled expected %d, got %d", http.StatusRequestTimeout, code)
	}
	if code := predictErrorStatusCode(io.EOF); code != http.StatusInternalServerError {
		t.Fatalf("default expected %d, got %d", http.StatusInternalServerError, code)
	}
}

func TestHTTPCanaryMismatchTelemetryDoesNotBreakPrimaryResponse(t *testing.T) {
	svc, err := NewHTTPService(&staticAdapter{}, HTTPServiceConfig{
		DefaultBudgetProfile: BudgetProfileBalanced,
		MaxBatchSize:         8,
		BatchWindow:          2 * time.Millisecond,
		QueueSize:            16,
		CanaryAdapter:        &mismatchCanaryAdapter{},
		CanarySampleRate:     1.0,
		CanaryScoreAbsTol:    1e-6,
		CanaryTimeout:        50 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("NewHTTPService() error = %v", err)
	}
	defer func() {
		_ = svc.Close()
	}()

	mux := http.NewServeMux()
	svc.RegisterRoutes(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	payload := PredictRequest{
		RequestID: "canary-1",
		Input:     []float32{0.1, 0.2},
	}
	raw, _ := json.Marshal(payload)
	resp, postErr := http.Post(
		server.URL+"/predict",
		"application/json",
		bytes.NewReader(raw),
	)
	if postErr != nil {
		t.Fatalf("POST /predict error = %v", postErr)
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("unexpected /predict status: %d body=%s", resp.StatusCode, string(body))
	}

	foundMismatch := false
	for range 20 {
		metricsResp, metricsErr := http.Get(server.URL + "/metrics")
		if metricsErr != nil {
			t.Fatalf("GET /metrics error = %v", metricsErr)
		}
		body, _ := io.ReadAll(metricsResp.Body)
		_ = metricsResp.Body.Close()
		text := string(body)
		if strings.Contains(text, "apexx_canary_mismatches_total 1") {
			foundMismatch = true
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !foundMismatch {
		t.Fatalf("expected canary mismatch metric to be incremented")
	}
}

func TestParseCanaryCapturePolicy(t *testing.T) {
	cases := []struct {
		raw  string
		want canaryCapturePolicy
	}{
		{raw: "", want: canaryCapturePolicyOff},
		{raw: "off", want: canaryCapturePolicyOff},
		{raw: "mismatch", want: canaryCapturePolicyMismatch},
		{raw: "error", want: canaryCapturePolicyError},
		{raw: "all", want: canaryCapturePolicyAll},
	}
	for _, tc := range cases {
		got, err := parseCanaryCapturePolicy(tc.raw)
		if err != nil {
			t.Fatalf("parseCanaryCapturePolicy(%q) error = %v", tc.raw, err)
		}
		if got != tc.want {
			t.Fatalf("parseCanaryCapturePolicy(%q) = %v, want %v", tc.raw, got, tc.want)
		}
	}
	if _, err := parseCanaryCapturePolicy("unsupported"); err == nil {
		t.Fatalf("expected parse error for unsupported policy")
	}
}

func TestNewHTTPServiceCanaryCapturePolicyRequiresPath(t *testing.T) {
	_, err := NewHTTPService(&staticAdapter{}, HTTPServiceConfig{
		DefaultBudgetProfile: BudgetProfileBalanced,
		MaxBatchSize:         8,
		BatchWindow:          2 * time.Millisecond,
		QueueSize:            16,
		CanaryCapturePolicy:  "mismatch",
	})
	if err == nil {
		t.Fatalf("expected configuration error when capture policy is enabled without path")
	}
}

func TestHTTPCanaryCaptureWritesMismatchRecord(t *testing.T) {
	capturePath := filepath.Join(t.TempDir(), "captures", "canary.jsonl")
	svc, err := NewHTTPService(&staticAdapter{}, HTTPServiceConfig{
		DefaultBudgetProfile:  BudgetProfileBalanced,
		MaxBatchSize:          8,
		BatchWindow:           2 * time.Millisecond,
		QueueSize:             16,
		CanaryAdapter:         &mismatchCanaryAdapter{},
		CanarySampleRate:      1.0,
		CanaryScoreAbsTol:     1e-6,
		CanaryTimeout:         50 * time.Millisecond,
		CanaryCapturePolicy:   "mismatch",
		CanaryCapturePath:     capturePath,
		CanaryCaptureMaxBytes: 1024 * 1024,
	})
	if err != nil {
		t.Fatalf("NewHTTPService() error = %v", err)
	}
	defer func() {
		_ = svc.Close()
	}()

	mux := http.NewServeMux()
	svc.RegisterRoutes(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	payload := PredictRequest{
		RequestID: "canary-capture-mismatch-1",
		Input:     []float32{0.1, 0.2},
	}
	raw, _ := json.Marshal(payload)
	resp, postErr := http.Post(
		server.URL+"/predict",
		"application/json",
		bytes.NewReader(raw),
	)
	if postErr != nil {
		t.Fatalf("POST /predict error = %v", postErr)
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("unexpected /predict status: %d body=%s", resp.StatusCode, string(body))
	}

	content, readErr := waitForFileContent(capturePath, 500*time.Millisecond)
	if readErr != nil {
		t.Fatalf("failed to read canary capture file: %v", readErr)
	}
	lines := strings.Split(strings.TrimSpace(content), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) == "" {
		t.Fatalf("expected non-empty canary capture file")
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &decoded); err != nil {
		t.Fatalf("failed to decode capture line: %v", err)
	}
	if decoded["event_type"] != "mismatch" {
		t.Fatalf("expected mismatch event, got %v", decoded["event_type"])
	}
}

func TestHTTPCanaryCaptureWritesErrorRecord(t *testing.T) {
	capturePath := filepath.Join(t.TempDir(), "captures", "canary_error.jsonl")
	svc, err := NewHTTPService(&staticAdapter{}, HTTPServiceConfig{
		DefaultBudgetProfile:  BudgetProfileBalanced,
		MaxBatchSize:          8,
		BatchWindow:           2 * time.Millisecond,
		QueueSize:             16,
		CanaryAdapter:         &failingAdapter{err: errors.New("canary failed")},
		CanarySampleRate:      1.0,
		CanaryScoreAbsTol:     1e-6,
		CanaryTimeout:         50 * time.Millisecond,
		CanaryCapturePolicy:   "error",
		CanaryCapturePath:     capturePath,
		CanaryCaptureMaxBytes: 1024 * 1024,
	})
	if err != nil {
		t.Fatalf("NewHTTPService() error = %v", err)
	}
	defer func() {
		_ = svc.Close()
	}()

	mux := http.NewServeMux()
	svc.RegisterRoutes(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	payload := PredictRequest{
		RequestID: "canary-capture-error-1",
		Input:     []float32{0.1, 0.2},
	}
	raw, _ := json.Marshal(payload)
	resp, postErr := http.Post(
		server.URL+"/predict",
		"application/json",
		bytes.NewReader(raw),
	)
	if postErr != nil {
		t.Fatalf("POST /predict error = %v", postErr)
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("unexpected /predict status: %d body=%s", resp.StatusCode, string(body))
	}

	content, readErr := waitForFileContent(capturePath, 500*time.Millisecond)
	if readErr != nil {
		t.Fatalf("failed to read canary capture file: %v", readErr)
	}
	lines := strings.Split(strings.TrimSpace(content), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) == "" {
		t.Fatalf("expected non-empty canary capture file")
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(lines[0]), &decoded); err != nil {
		t.Fatalf("failed to decode capture line: %v", err)
	}
	if decoded["event_type"] != "error" {
		t.Fatalf("expected error event, got %v", decoded["event_type"])
	}
}

func TestHTTPCanaryCaptureRespectsMaxBytes(t *testing.T) {
	capturePath := filepath.Join(t.TempDir(), "captures", "limited.jsonl")
	svc, err := NewHTTPService(&staticAdapter{}, HTTPServiceConfig{
		DefaultBudgetProfile:  BudgetProfileBalanced,
		MaxBatchSize:          8,
		BatchWindow:           2 * time.Millisecond,
		QueueSize:             16,
		CanaryAdapter:         &mismatchCanaryAdapter{},
		CanarySampleRate:      1.0,
		CanaryScoreAbsTol:     1e-6,
		CanaryTimeout:         50 * time.Millisecond,
		CanaryCapturePolicy:   "mismatch",
		CanaryCapturePath:     capturePath,
		CanaryCaptureMaxBytes: 1,
	})
	if err != nil {
		t.Fatalf("NewHTTPService() error = %v", err)
	}
	defer func() {
		_ = svc.Close()
	}()

	mux := http.NewServeMux()
	svc.RegisterRoutes(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	payload := PredictRequest{
		RequestID: "canary-capture-limit-1",
		Input:     []float32{0.1, 0.2},
	}
	raw, _ := json.Marshal(payload)
	resp, postErr := http.Post(
		server.URL+"/predict",
		"application/json",
		bytes.NewReader(raw),
	)
	if postErr != nil {
		t.Fatalf("POST /predict error = %v", postErr)
	}
	defer func() {
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("unexpected /predict status: %d body=%s", resp.StatusCode, string(body))
	}
	time.Sleep(120 * time.Millisecond)
	info, statErr := os.Stat(capturePath)
	if statErr == nil && info.Size() > 0 {
		t.Fatalf("expected capture file to remain empty under max-bytes constraint")
	}
	if statErr != nil && !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("unexpected capture stat error: %v", statErr)
	}
}

func TestComparePredictResponsesDetectsScoreMismatch(t *testing.T) {
	mismatch, reason := comparePredictResponses(
		PredictResponse{
			SelectedTiles: 4,
			Scores:        []float32{0.1, 0.2},
		},
		PredictResponse{
			SelectedTiles: 4,
			Scores:        []float32{0.1, 0.5},
		},
		1e-3,
	)
	if !mismatch {
		t.Fatalf("expected mismatch for score divergence")
	}
	if reason == "" {
		t.Fatalf("expected non-empty mismatch reason")
	}
}
