package service

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
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
