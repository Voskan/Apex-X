package service

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"
)

type HTTPServiceConfig struct {
	DefaultBudgetProfile string
	MaxBatchSize         int
	BatchWindow          time.Duration
	QueueSize            int
	Logger               *slog.Logger
	Hooks                TelemetryHooks
}

type HTTPService struct {
	adapter InferenceAdapter
	batcher *Batcher
	metrics *Metrics

	defaultBudgetProfile string
	reqCounter           atomic.Uint64
	logger               *slog.Logger
	hooks                TelemetryHooks
}

func NewHTTPService(adapter InferenceAdapter, cfg HTTPServiceConfig) (*HTTPService, error) {
	defaultProfile, err := normalizeBudgetProfile(cfg.DefaultBudgetProfile, BudgetProfileBalanced)
	if err != nil {
		return nil, fmt.Errorf("invalid default budget profile: %w", err)
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.New(slog.NewJSONHandler(io.Discard, nil))
	}
	hooks := cfg.Hooks
	if hooks == nil {
		hooks = NopTelemetryHooks{}
	}
	metrics := &Metrics{}
	batcher, err := NewBatcher(adapter, BatcherConfig{
		MaxBatchSize: cfg.MaxBatchSize,
		BatchWindow:  cfg.BatchWindow,
		QueueSize:    cfg.QueueSize,
		Logger:       logger,
		Hooks:        hooks,
		OnBatch: func(
			batchSize int,
			avgQueueWait time.Duration,
			inferenceTime time.Duration,
			batchErr error,
		) {
			metrics.RecordBatchStats(batchSize, avgQueueWait, inferenceTime, batchErr == nil)
		},
	})
	if err != nil {
		return nil, err
	}
	batcher.Start()
	return &HTTPService{
		adapter:              adapter,
		batcher:              batcher,
		metrics:              metrics,
		defaultBudgetProfile: defaultProfile,
		logger:               logger,
		hooks:                hooks,
	}, nil
}

func (s *HTTPService) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/predict", s.handlePredict)
}

func (s *HTTPService) Close() error {
	s.batcher.Stop()
	return s.adapter.Close()
}

func (s *HTTPService) handleHealth(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		http.Error(writer, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	response := map[string]string{
		"status":  "ok",
		"adapter": s.adapter.Name(),
	}
	writeJSON(writer, http.StatusOK, response)
}

func (s *HTTPService) handleMetrics(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		http.Error(writer, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writer.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = writer.Write([]byte(s.metrics.Snapshot().PrometheusText()))
}

func (s *HTTPService) handlePredict(writer http.ResponseWriter, request *http.Request) {
	start := time.Now()
	if request.Method != http.MethodPost {
		http.Error(writer, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var body PredictRequest
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&body); err != nil {
		http.Error(writer, fmt.Sprintf("invalid request payload: %v", err), http.StatusBadRequest)
		return
	}

	if strings.TrimSpace(body.RequestID) == "" {
		body.RequestID = fmt.Sprintf("req-%d", s.reqCounter.Add(1))
	}
	s.hooks.OnHTTPRequestStart(request.Context(), "/predict", body.RequestID)
	profile, err := normalizeBudgetProfile(body.BudgetProfile, s.defaultBudgetProfile)
	if err != nil {
		s.hooks.OnHTTPRequestDone(
			request.Context(),
			"/predict",
			body.RequestID,
			http.StatusBadRequest,
			time.Since(start),
			err,
		)
		http.Error(writer, err.Error(), http.StatusBadRequest)
		return
	}
	body.BudgetProfile = profile
	s.logger.Info(
		"predict_request_received",
		"request_id", body.RequestID,
		"budget_profile", body.BudgetProfile,
		"input_len", len(body.Input),
	)

	response, predictErr := s.Predict(request.Context(), body)
	if predictErr != nil {
		s.hooks.OnHTTPRequestDone(
			request.Context(),
			"/predict",
			body.RequestID,
			http.StatusInternalServerError,
			time.Since(start),
			predictErr,
		)
		s.logger.Error(
			"predict_request_failed",
			"request_id", body.RequestID,
			"budget_profile", body.BudgetProfile,
			"error", predictErr.Error(),
		)
		http.Error(writer, predictErr.Error(), http.StatusInternalServerError)
		return
	}
	s.hooks.OnHTTPRequestDone(
		request.Context(),
		"/predict",
		body.RequestID,
		http.StatusOK,
		time.Since(start),
		nil,
	)
	s.logger.Info(
		"predict_request_done",
		"request_id", response.RequestID,
		"budget_profile", response.BudgetProfile,
		"selected_tiles", response.SelectedTiles,
		"backend", response.Backend,
	)
	writeJSON(writer, http.StatusOK, response)
}

func (s *HTTPService) Predict(ctx context.Context, req PredictRequest) (PredictResponse, error) {
	s.metrics.RecordRequestStart()
	start := time.Now()
	resp, err := s.batcher.Submit(ctx, req)
	s.metrics.RecordRequestDone(time.Since(start), err == nil)
	return resp, err
}

func writeJSON(writer http.ResponseWriter, statusCode int, payload any) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(statusCode)
	_ = json.NewEncoder(writer).Encode(payload)
}
