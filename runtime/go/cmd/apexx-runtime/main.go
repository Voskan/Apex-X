package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/apex-x/apex-x/runtime/go/internal/service"
)

func main() {
	var (
		addr             = flag.String("addr", ":8080", "HTTP listen address")
		adapterName      = flag.String("adapter", "onnxruntime", "inference adapter: onnxruntime|tensorrt")
		modelPath        = flag.String("model-path", "models/apex-x.onnx", "ONNX model path (ORT adapter)")
		enginePath       = flag.String("engine-path", "models/apex-x.plan", "TensorRT engine path (TRT adapter)")
		batchWindowMS    = flag.Int("batch-window-ms", 8, "batch queue window in milliseconds")
		maxBatchSize     = flag.Int("max-batch-size", 8, "maximum batch size")
		queueSize        = flag.Int("queue-size", 256, "request queue size")
		predictTimeoutMS = flag.Int(
			"predict-timeout-ms",
			envInt("APEXX_PREDICT_TIMEOUT_MS", 0),
			"optional per-request predict timeout in milliseconds (0 disables timeout)",
		)
		canaryAdapterName = flag.String(
			"canary-adapter",
			envOr("APEXX_CANARY_ADAPTER", ""),
			"optional canary adapter: onnxruntime|tensorrt",
		)
		canaryModelPath = flag.String(
			"canary-model-path",
			envOr("APEXX_CANARY_MODEL_PATH", ""),
			"optional ONNX model path for canary adapter",
		)
		canaryEnginePath = flag.String(
			"canary-engine-path",
			envOr("APEXX_CANARY_ENGINE_PATH", ""),
			"optional TensorRT engine path for canary adapter",
		)
		canarySampleRate = flag.Float64(
			"canary-sample-rate",
			envFloat("APEXX_CANARY_SAMPLE_RATE", 0.0),
			"canary sampling ratio in [0,1]; defaults to 1 when canary adapter is set",
		)
		canaryScoreAbsTol = flag.Float64(
			"canary-score-abs-tol",
			envFloat("APEXX_CANARY_SCORE_ABS_TOL", 1e-4),
			"absolute tolerance for canary score parity checks",
		)
		canaryTimeoutMS = flag.Int(
			"canary-timeout-ms",
			envInt("APEXX_CANARY_TIMEOUT_MS", 150),
			"timeout for background canary comparisons in milliseconds",
		)
		canaryCapturePolicy = flag.String(
			"canary-capture-policy",
			envOr("APEXX_CANARY_CAPTURE_POLICY", "off"),
			"canary capture policy: off|mismatch|error|all",
		)
		canaryCapturePath = flag.String(
			"canary-capture-path",
			envOr("APEXX_CANARY_CAPTURE_PATH", ""),
			"optional JSONL file path for canary capture payloads",
		)
		canaryCaptureMaxBytes = flag.Int64(
			"canary-capture-max-bytes",
			envInt64("APEXX_CANARY_CAPTURE_MAX_BYTES", 10*1024*1024),
			"maximum canary capture file size in bytes (0 disables size limit)",
		)
		profile    = flag.String("default-budget-profile", service.BudgetProfileBalanced, "default budget profile")
		logFormat  = flag.String("log-format", envOr("APEXX_LOG_FORMAT", "json"), "log format: json|text")
		logLevel   = flag.String("log-level", envOr("APEXX_LOG_LEVEL", "info"), "log level: debug|info|warn|error")
		enableOTel = flag.Bool("enable-otel-hooks", envBool("APEXX_ENABLE_OTEL_HOOKS", false), "enable no-op OpenTelemetry hooks extension points")
	)
	flag.Parse()

	logger, err := buildLogger(*logFormat, *logLevel)
	if err != nil {
		log.Fatalf("failed to configure logger: %v", err)
	}

	adapter, err := buildAdapter(*adapterName, *modelPath, *enginePath)
	if err != nil {
		log.Fatalf("failed to build adapter: %v", err)
	}
	var canaryAdapter service.InferenceAdapter
	if strings.TrimSpace(*canaryAdapterName) != "" {
		canaryAdapter, err = buildAdapter(
			strings.TrimSpace(*canaryAdapterName),
			*canaryModelPath,
			*canaryEnginePath,
		)
		if err != nil {
			log.Fatalf("failed to build canary adapter: %v", err)
		}
	}
	hooks := service.TelemetryHooks(service.NopTelemetryHooks{})
	if *enableOTel {
		hooks = service.NopTelemetryHooks{}
		logger.Info("otel_hooks_enabled", "status", "configured_noop_hooks")
	}

	httpService, err := service.NewHTTPService(adapter, service.HTTPServiceConfig{
		DefaultBudgetProfile:  *profile,
		MaxBatchSize:          *maxBatchSize,
		BatchWindow:           time.Duration(*batchWindowMS) * time.Millisecond,
		QueueSize:             *queueSize,
		PredictTimeout:        time.Duration(*predictTimeoutMS) * time.Millisecond,
		CanaryAdapter:         canaryAdapter,
		CanarySampleRate:      *canarySampleRate,
		CanaryScoreAbsTol:     *canaryScoreAbsTol,
		CanaryTimeout:         time.Duration(*canaryTimeoutMS) * time.Millisecond,
		CanaryCapturePolicy:   *canaryCapturePolicy,
		CanaryCapturePath:     *canaryCapturePath,
		CanaryCaptureMaxBytes: *canaryCaptureMaxBytes,
		Logger:                logger,
		Hooks:                 hooks,
	})
	if err != nil {
		log.Fatalf("failed to create http service: %v", err)
	}
	defer func() {
		if closeErr := httpService.Close(); closeErr != nil {
			log.Printf("service shutdown error: %v", closeErr)
		}
	}()

	mux := http.NewServeMux()
	httpService.RegisterRoutes(mux)

	// Apply middleware chain
	handler := service.Chain(mux,
		service.RequestIDMiddleware,
		service.RecoveryMiddleware(logger),
		service.LoggingMiddleware(logger),
	)

	server := &http.Server{
		Addr:              *addr,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		logger.Info(
			"runtime_server_start",
			"addr", *addr,
			"adapter", adapter.Name(),
			"default_budget_profile", *profile,
			"max_batch_size", *maxBatchSize,
			"batch_window_ms", *batchWindowMS,
			"queue_size", *queueSize,
			"predict_timeout_ms", *predictTimeoutMS,
			"canary_adapter", strings.TrimSpace(*canaryAdapterName),
			"canary_sample_rate", *canarySampleRate,
			"canary_score_abs_tol", *canaryScoreAbsTol,
			"canary_timeout_ms", *canaryTimeoutMS,
			"canary_capture_policy", strings.TrimSpace(*canaryCapturePolicy),
			"canary_capture_path", strings.TrimSpace(*canaryCapturePath),
			"canary_capture_max_bytes", *canaryCaptureMaxBytes,
		)
		if serveErr := server.ListenAndServe(); serveErr != nil && serveErr != http.ErrServerClosed {
			log.Fatalf("http serve failed: %v", serveErr)
		}
	}()

	waitForSignal()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if shutdownErr := server.Shutdown(ctx); shutdownErr != nil {
		log.Printf("http shutdown error: %v", shutdownErr)
	}
}

func buildAdapter(adapterName string, modelPath string, enginePath string) (service.InferenceAdapter, error) {
	switch adapterName {
	case "onnxruntime":
		return service.NewORTAdapter(modelPath)
	case "tensorrt":
		return service.NewTensorRTAdapter(enginePath)
	default:
		return nil, fmt.Errorf("unsupported adapter %q", adapterName)
	}
}

func buildLogger(format string, level string) (*slog.Logger, error) {
	normalizedFormat := strings.ToLower(strings.TrimSpace(format))
	normalizedLevel := strings.ToLower(strings.TrimSpace(level))

	var slogLevel slog.Level
	switch normalizedLevel {
	case "debug":
		slogLevel = slog.LevelDebug
	case "info", "":
		slogLevel = slog.LevelInfo
	case "warn", "warning":
		slogLevel = slog.LevelWarn
	case "error":
		slogLevel = slog.LevelError
	default:
		return nil, fmt.Errorf("unsupported log level %q", level)
	}

	opts := &slog.HandlerOptions{Level: slogLevel}
	switch normalizedFormat {
	case "json", "":
		return slog.New(slog.NewJSONHandler(os.Stdout, opts)), nil
	case "text":
		return slog.New(slog.NewTextHandler(os.Stdout, opts)), nil
	case "discard":
		return slog.New(slog.NewJSONHandler(io.Discard, opts)), nil
	default:
		return nil, fmt.Errorf("unsupported log format %q", format)
	}
}

func envOr(key string, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}

func envBool(key string, fallback bool) bool {
	value := strings.ToLower(strings.TrimSpace(os.Getenv(key)))
	if value == "" {
		return fallback
	}
	switch value {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}

func envInt(key string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func envFloat(key string, fallback float64) float64 {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return fallback
	}
	return parsed
}

func envInt64(key string, fallback int64) int64 {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil {
		return fallback
	}
	return parsed
}

func waitForSignal() {
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGTERM, syscall.SIGINT)
	<-signals
}
