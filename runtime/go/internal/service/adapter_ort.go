package service

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

type ORTAdapter struct {
	modelPath     string
	modelSize     int64
	bridgeCommand []string
}

func NewORTAdapter(modelPath string) (InferenceAdapter, error) {
	resolvedPath, err := resolveModelPath(
		modelPath,
		"APEXX_ORT_MODEL_PATH",
		"onnx model",
	)
	if err != nil {
		return nil, err
	}
	info, statErr := os.Stat(resolvedPath)
	if statErr != nil {
		return nil, fmt.Errorf("failed to stat onnx model %q: %w", resolvedPath, statErr)
	}
	if info.IsDir() {
		return nil, fmt.Errorf("onnx model path %q is a directory", resolvedPath)
	}
	if info.Size() <= 0 {
		return nil, fmt.Errorf("onnx model path %q is empty", resolvedPath)
	}
	bridgeCommand, err := parseBridgeCommand(os.Getenv("APEXX_ORT_BRIDGE_CMD"))
	if err != nil {
		return nil, fmt.Errorf("invalid APEXX_ORT_BRIDGE_CMD: %w", err)
	}
	return &ORTAdapter{
		modelPath:     resolvedPath,
		modelSize:     info.Size(),
		bridgeCommand: bridgeCommand,
	}, nil
}

func (a *ORTAdapter) Name() string {
	return "onnxruntime-cpu-baseline"
}

func (a *ORTAdapter) PredictBatch(
	ctx context.Context,
	reqs []PredictRequest,
) ([]PredictResponse, error) {
	if len(a.bridgeCommand) == 0 {
		return nil, fmt.Errorf(
			"%w: onnxruntime bridge command is not configured; set APEXX_ORT_BRIDGE_CMD",
			ErrBackendUnavailable,
		)
	}
	bridgeResponses, bridgeErr := runBridgePredictBatch(
		ctx,
		a.bridgeCommand,
		bridgePredictRequest{
			Backend:      "onnxruntime",
			ArtifactPath: a.modelPath,
			Requests:     reqs,
		},
		a.Name(),
	)
	if bridgeErr != nil {
		return nil, fmt.Errorf("onnxruntime bridge predict failed: %w", bridgeErr)
	}
	return bridgeResponses, nil
}

func (a *ORTAdapter) Close() error {
	if a == nil {
		return errors.New("adapter is nil")
	}
	return nil
}

func selectedTilesForProfile(profile string) int {
	switch profile {
	case BudgetProfileQuality:
		return 64
	case BudgetProfileEdge:
		return 16
	default:
		return 32
	}
}

func mean(values []float32) float32 {
	if len(values) == 0 {
		return 0.0
	}
	var total float32
	for _, value := range values {
		total += value
	}
	return total / float32(len(values))
}

func resolveModelPath(value string, envVar string, label string) (string, error) {
	candidate := value
	if candidate == "" {
		candidate = os.Getenv(envVar)
	}
	candidate = filepath.Clean(candidate)
	if candidate == "" || candidate == "." {
		return "", fmt.Errorf("%s path is required (flag or %s)", label, envVar)
	}
	absPath, err := filepath.Abs(candidate)
	if err != nil {
		return "", fmt.Errorf("failed to resolve %s path %q: %w", label, candidate, err)
	}
	return absPath, nil
}
