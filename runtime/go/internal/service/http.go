package service

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type canaryCapturePolicy uint8

const (
	canaryCapturePolicyOff canaryCapturePolicy = iota
	canaryCapturePolicyMismatch
	canaryCapturePolicyError
	canaryCapturePolicyAll
)

func parseCanaryCapturePolicy(raw string) (canaryCapturePolicy, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "off", "none":
		return canaryCapturePolicyOff, nil
	case "mismatch":
		return canaryCapturePolicyMismatch, nil
	case "error":
		return canaryCapturePolicyError, nil
	case "all":
		return canaryCapturePolicyAll, nil
	default:
		return canaryCapturePolicyOff, fmt.Errorf("invalid canary capture policy %q", raw)
	}
}

func shouldCaptureCanaryEvent(policy canaryCapturePolicy, eventType string) bool {
	switch policy {
	case canaryCapturePolicyAll:
		return true
	case canaryCapturePolicyMismatch:
		return eventType == "mismatch"
	case canaryCapturePolicyError:
		return eventType == "error"
	default:
		return false
	}
}

type canaryCaptureRecord struct {
	Timestamp      string           `json:"timestamp"`
	EventType      string           `json:"event_type"`
	Request        PredictRequest   `json:"request"`
	Primary        PredictResponse  `json:"primary"`
	Shadow         *PredictResponse `json:"shadow,omitempty"`
	Reason         string           `json:"reason,omitempty"`
	CanaryError    string           `json:"canary_error,omitempty"`
	PrimaryBackend string           `json:"primary_backend"`
	ShadowBackend  string           `json:"shadow_backend,omitempty"`
}

type HTTPServiceConfig struct {
	DefaultBudgetProfile  string
	MaxBatchSize          int
	BatchWindow           time.Duration
	QueueSize             int
	PredictTimeout        time.Duration
	CanaryAdapter         InferenceAdapter
	CanarySampleRate      float64
	CanaryScoreAbsTol     float64
	CanaryTimeout         time.Duration
	CanaryCapturePolicy   string
	CanaryCapturePath     string
	CanaryCaptureMaxBytes int64
	Logger                *slog.Logger
	Hooks                 TelemetryHooks
}

type HTTPService struct {
	adapter InferenceAdapter
	batcher *Batcher
	metrics *Metrics

	defaultBudgetProfile  string
	predictTimeout        time.Duration
	canaryAdapter         InferenceAdapter
	canarySampleRate      float64
	canaryScoreAbsTol     float64
	canaryTimeout         time.Duration
	canaryCapturePolicy   canaryCapturePolicy
	canaryCapturePath     string
	canaryCaptureMaxBytes int64
	canaryCaptureMu       sync.Mutex
	canarySeq             atomic.Uint64
	canaryWG              sync.WaitGroup
	reqCounter            atomic.Uint64
	logger                *slog.Logger
	hooks                 TelemetryHooks
}

func NewHTTPService(adapter InferenceAdapter, cfg HTTPServiceConfig) (*HTTPService, error) {
	defaultProfile, err := normalizeBudgetProfile(cfg.DefaultBudgetProfile, BudgetProfileBalanced)
	if err != nil {
		return nil, fmt.Errorf("invalid default budget profile: %w", err)
	}
	if cfg.PredictTimeout < 0 {
		return nil, fmt.Errorf("predict timeout must be >= 0")
	}
	if cfg.CanarySampleRate < 0.0 || cfg.CanarySampleRate > 1.0 {
		return nil, fmt.Errorf("canary sample rate must be in [0,1]")
	}
	if cfg.CanaryScoreAbsTol < 0.0 {
		return nil, fmt.Errorf("canary score tolerance must be >= 0")
	}
	if cfg.CanaryTimeout < 0 {
		return nil, fmt.Errorf("canary timeout must be >= 0")
	}
	if cfg.CanaryCaptureMaxBytes < 0 {
		return nil, fmt.Errorf("canary capture max bytes must be >= 0")
	}
	capturePolicy, err := parseCanaryCapturePolicy(cfg.CanaryCapturePolicy)
	if err != nil {
		return nil, err
	}
	capturePath := strings.TrimSpace(cfg.CanaryCapturePath)
	if capturePolicy != canaryCapturePolicyOff && capturePath == "" {
		return nil, fmt.Errorf("canary capture path is required when capture policy is enabled")
	}
	if capturePath != "" {
		capturePath, err = filepath.Abs(filepath.Clean(capturePath))
		if err != nil {
			return nil, fmt.Errorf("failed to resolve canary capture path %q: %w", cfg.CanaryCapturePath, err)
		}
	}

	canarySampleRate := cfg.CanarySampleRate
	canaryScoreTol := cfg.CanaryScoreAbsTol
	canaryTimeout := cfg.CanaryTimeout
	if cfg.CanaryAdapter != nil {
		if canarySampleRate == 0.0 {
			canarySampleRate = 1.0
		}
		if canaryScoreTol == 0.0 {
			canaryScoreTol = 1e-4
		}
		if canaryTimeout == 0 {
			canaryTimeout = 150 * time.Millisecond
		}
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
		adapter:               adapter,
		batcher:               batcher,
		metrics:               metrics,
		defaultBudgetProfile:  defaultProfile,
		predictTimeout:        cfg.PredictTimeout,
		canaryAdapter:         cfg.CanaryAdapter,
		canarySampleRate:      canarySampleRate,
		canaryScoreAbsTol:     canaryScoreTol,
		canaryTimeout:         canaryTimeout,
		canaryCapturePolicy:   capturePolicy,
		canaryCapturePath:     capturePath,
		canaryCaptureMaxBytes: cfg.CanaryCaptureMaxBytes,
		logger:                logger,
		hooks:                 hooks,
	}, nil
}

func (s *HTTPService) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/predict", s.handlePredict)
}

func (s *HTTPService) Close() error {
	s.batcher.Stop()
	s.canaryWG.Wait()
	primaryErr := s.adapter.Close()
	var canaryErr error
	if s.canaryAdapter != nil {
		canaryErr = s.canaryAdapter.Close()
	}
	if primaryErr != nil {
		if canaryErr != nil {
			return fmt.Errorf(
				"primary adapter close failed: %v; canary adapter close failed: %v",
				primaryErr,
				canaryErr,
			)
		}
		return primaryErr
	}
	return canaryErr
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

	predictCtx := request.Context()
	cancel := func() {}
	if s.predictTimeout > 0 {
		predictCtx, cancel = context.WithTimeout(request.Context(), s.predictTimeout)
	}
	defer cancel()

	response, predictErr := s.Predict(predictCtx, body)
	if predictErr != nil {
		statusCode := predictErrorStatusCode(predictErr)
		s.hooks.OnHTTPRequestDone(
			request.Context(),
			"/predict",
			body.RequestID,
			statusCode,
			time.Since(start),
			predictErr,
		)
		s.logger.Error(
			"predict_request_failed",
			"request_id", body.RequestID,
			"budget_profile", body.BudgetProfile,
			"status_code", statusCode,
			"error", predictErr.Error(),
		)
		http.Error(writer, predictErr.Error(), statusCode)
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
	if s.shouldSampleCanary() {
		s.metrics.RecordCanarySample()
		s.dispatchCanaryCompare(body, response)
	}
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

func predictErrorStatusCode(err error) int {
	switch {
	case errors.Is(err, ErrQueueFull):
		return http.StatusTooManyRequests
	case errors.Is(err, ErrBackendUnavailable):
		return http.StatusServiceUnavailable
	case errors.Is(err, ErrBackendInference):
		return http.StatusBadGateway
	case errors.Is(err, ErrBackendProtocol):
		return http.StatusBadGateway
	case errors.Is(err, context.DeadlineExceeded):
		return http.StatusGatewayTimeout
	case errors.Is(err, context.Canceled):
		return http.StatusRequestTimeout
	default:
		return http.StatusInternalServerError
	}
}

func (s *HTTPService) shouldSampleCanary() bool {
	if s.canaryAdapter == nil || s.canarySampleRate <= 0.0 {
		return false
	}
	if s.canarySampleRate >= 1.0 {
		return true
	}
	sequence := s.canarySeq.Add(1)
	threshold := uint64(s.canarySampleRate * 10000.0)
	return ((sequence - 1) % 10000) < threshold
}

func (s *HTTPService) dispatchCanaryCompare(
	req PredictRequest,
	primary PredictResponse,
) {
	s.canaryWG.Add(1)
	go func() {
		defer s.canaryWG.Done()
		ctx := context.Background()
		cancel := func() {}
		if s.canaryTimeout > 0 {
			ctx, cancel = context.WithTimeout(ctx, s.canaryTimeout)
		}
		defer cancel()

		shadowBatch, err := s.canaryAdapter.PredictBatch(ctx, []PredictRequest{req})
		if err != nil {
			s.metrics.RecordCanaryError()
			s.logger.Warn(
				"canary_compare_error",
				"request_id", primary.RequestID,
				"primary_backend", primary.Backend,
				"error", err.Error(),
			)
			s.captureCanaryRecord(canaryCaptureRecord{
				Timestamp:      time.Now().UTC().Format(time.RFC3339Nano),
				EventType:      "error",
				Request:        req,
				Primary:        primary,
				CanaryError:    err.Error(),
				PrimaryBackend: primary.Backend,
			})
			return
		}
		if len(shadowBatch) != 1 {
			s.metrics.RecordCanaryError()
			compareErr := fmt.Sprintf("canary adapter returned %d responses", len(shadowBatch))
			s.logger.Warn(
				"canary_compare_error",
				"request_id", primary.RequestID,
				"primary_backend", primary.Backend,
				"error", compareErr,
			)
			s.captureCanaryRecord(canaryCaptureRecord{
				Timestamp:      time.Now().UTC().Format(time.RFC3339Nano),
				EventType:      "error",
				Request:        req,
				Primary:        primary,
				CanaryError:    compareErr,
				PrimaryBackend: primary.Backend,
			})
			return
		}
		shadow := shadowBatch[0]
		mismatch, reason := comparePredictResponses(primary, shadow, s.canaryScoreAbsTol)
		s.metrics.RecordCanaryCompared(mismatch)
		if mismatch {
			s.logger.Warn(
				"canary_mismatch",
				"request_id", primary.RequestID,
				"primary_backend", primary.Backend,
				"shadow_backend", shadow.Backend,
				"reason", reason,
			)
			s.captureCanaryRecord(canaryCaptureRecord{
				Timestamp:      time.Now().UTC().Format(time.RFC3339Nano),
				EventType:      "mismatch",
				Request:        req,
				Primary:        primary,
				Shadow:         &shadow,
				Reason:         reason,
				PrimaryBackend: primary.Backend,
				ShadowBackend:  shadow.Backend,
			})
			return
		}
		s.captureCanaryRecord(canaryCaptureRecord{
			Timestamp:      time.Now().UTC().Format(time.RFC3339Nano),
			EventType:      "match",
			Request:        req,
			Primary:        primary,
			Shadow:         &shadow,
			PrimaryBackend: primary.Backend,
			ShadowBackend:  shadow.Backend,
		})
	}()
}

func (s *HTTPService) captureCanaryRecord(record canaryCaptureRecord) {
	if !shouldCaptureCanaryEvent(s.canaryCapturePolicy, record.EventType) {
		return
	}
	if s.canaryCapturePath == "" {
		return
	}

	payload, err := json.Marshal(record)
	if err != nil {
		s.logger.Warn("canary_capture_encode_failed", "error", err.Error())
		return
	}
	line := append(payload, '\n')

	s.canaryCaptureMu.Lock()
	defer s.canaryCaptureMu.Unlock()

	if err := os.MkdirAll(filepath.Dir(s.canaryCapturePath), 0o755); err != nil {
		s.logger.Warn(
			"canary_capture_dir_failed",
			"path", filepath.Dir(s.canaryCapturePath),
			"error", err.Error(),
		)
		return
	}
	if s.canaryCaptureMaxBytes > 0 {
		info, statErr := os.Stat(s.canaryCapturePath)
		if statErr == nil && info.Size()+int64(len(line)) > s.canaryCaptureMaxBytes {
			s.logger.Warn(
				"canary_capture_dropped_max_size",
				"path", s.canaryCapturePath,
				"max_bytes", s.canaryCaptureMaxBytes,
			)
			return
		}
		if errors.Is(statErr, os.ErrNotExist) && int64(len(line)) > s.canaryCaptureMaxBytes {
			s.logger.Warn(
				"canary_capture_dropped_max_size",
				"path", s.canaryCapturePath,
				"max_bytes", s.canaryCaptureMaxBytes,
			)
			return
		}
		if statErr != nil && !errors.Is(statErr, os.ErrNotExist) {
			s.logger.Warn(
				"canary_capture_stat_failed",
				"path", s.canaryCapturePath,
				"error", statErr.Error(),
			)
			return
		}
	}
	file, openErr := os.OpenFile(s.canaryCapturePath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if openErr != nil {
		s.logger.Warn(
			"canary_capture_open_failed",
			"path", s.canaryCapturePath,
			"error", openErr.Error(),
		)
		return
	}
	defer func() {
		_ = file.Close()
	}()
	if _, writeErr := file.Write(line); writeErr != nil {
		s.logger.Warn(
			"canary_capture_write_failed",
			"path", s.canaryCapturePath,
			"error", writeErr.Error(),
		)
	}
}

func comparePredictResponses(
	primary PredictResponse,
	shadow PredictResponse,
	scoreAbsTolerance float64,
) (bool, string) {
	if primary.SelectedTiles != shadow.SelectedTiles {
		return true, fmt.Sprintf(
			"selected_tiles_mismatch primary=%d shadow=%d",
			primary.SelectedTiles,
			shadow.SelectedTiles,
		)
	}
	if len(primary.Scores) != len(shadow.Scores) {
		return true, fmt.Sprintf(
			"scores_length_mismatch primary=%d shadow=%d",
			len(primary.Scores),
			len(shadow.Scores),
		)
	}
	for idx := range primary.Scores {
		diff := float64(primary.Scores[idx]) - float64(shadow.Scores[idx])
		if diff < 0.0 {
			diff = -diff
		}
		if diff > scoreAbsTolerance {
			return true, fmt.Sprintf(
				"score_mismatch idx=%d primary=%.6f shadow=%.6f tol=%.6f",
				idx,
				primary.Scores[idx],
				shadow.Scores[idx],
				scoreAbsTolerance,
			)
		}
	}
	return false, ""
}
