package service

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

type ORTAdapter struct {
	modelPath string
	modelSize int64
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
	return &ORTAdapter{
		modelPath: resolvedPath,
		modelSize: info.Size(),
	}, nil
}

func (a *ORTAdapter) Name() string {
	return "onnxruntime-cpu-baseline"
}

func (a *ORTAdapter) PredictBatch(
	ctx context.Context,
	reqs []PredictRequest,
) ([]PredictResponse, error) {
	out := make([]PredictResponse, len(reqs))
	for idx, req := range reqs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		profile, err := normalizeBudgetProfile(req.BudgetProfile, BudgetProfileBalanced)
		if err != nil {
			return nil, err
		}
		score := mean(req.Input)
		out[idx] = PredictResponse{
			RequestID:     req.RequestID,
			BudgetProfile: profile,
			SelectedTiles: selectedTilesForProfile(profile),
			Scores:        []float32{score},
			Backend:       a.Name(),
		}
	}
	return out, nil
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
