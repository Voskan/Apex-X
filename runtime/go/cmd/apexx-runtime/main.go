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
	"strings"
	"syscall"
	"time"

	"github.com/apex-x/apex-x/runtime/go/internal/service"
)

func main() {
	var (
		addr          = flag.String("addr", ":8080", "HTTP listen address")
		adapterName   = flag.String("adapter", "onnxruntime", "inference adapter: onnxruntime|tensorrt")
		modelPath     = flag.String("model-path", "models/apex-x.onnx", "ONNX model path (ORT adapter)")
		enginePath    = flag.String("engine-path", "models/apex-x.plan", "TensorRT engine path (TRT adapter)")
		batchWindowMS = flag.Int("batch-window-ms", 8, "batch queue window in milliseconds")
		maxBatchSize  = flag.Int("max-batch-size", 8, "maximum batch size")
		queueSize     = flag.Int("queue-size", 256, "request queue size")
		profile       = flag.String("default-budget-profile", service.BudgetProfileBalanced, "default budget profile")
		logFormat     = flag.String("log-format", envOr("APEXX_LOG_FORMAT", "json"), "log format: json|text")
		logLevel      = flag.String("log-level", envOr("APEXX_LOG_LEVEL", "info"), "log level: debug|info|warn|error")
		enableOTel    = flag.Bool("enable-otel-hooks", envBool("APEXX_ENABLE_OTEL_HOOKS", false), "enable no-op OpenTelemetry hooks extension points")
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
	hooks := service.TelemetryHooks(service.NopTelemetryHooks{})
	if *enableOTel {
		hooks = service.NopTelemetryHooks{}
		logger.Info("otel_hooks_enabled", "status", "configured_noop_hooks")
	}

	httpService, err := service.NewHTTPService(adapter, service.HTTPServiceConfig{
		DefaultBudgetProfile: *profile,
		MaxBatchSize:         *maxBatchSize,
		BatchWindow:          time.Duration(*batchWindowMS) * time.Millisecond,
		QueueSize:            *queueSize,
		Logger:               logger,
		Hooks:                hooks,
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

	server := &http.Server{
		Addr:              *addr,
		Handler:           mux,
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

func waitForSignal() {
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGTERM, syscall.SIGINT)
	<-signals
}
