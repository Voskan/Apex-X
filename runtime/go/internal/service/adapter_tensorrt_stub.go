//go:build !tensorrt || !cgo

package service

import (
	"context"
	"fmt"
	"os"
)

type tensorRTBridgeAdapter struct {
	enginePath    string
	bridgeCommand []string
}

func NewTensorRTAdapter(enginePath string) (InferenceAdapter, error) {
	resolvedPath, err := resolveModelPath(
		enginePath,
		"APEXX_TRT_ENGINE_PATH",
		"tensorrt engine",
	)
	if err != nil {
		return nil, err
	}
	info, statErr := os.Stat(resolvedPath)
	if statErr != nil {
		return nil, fmt.Errorf("failed to stat TensorRT engine %q: %w", resolvedPath, statErr)
	}
	if info.IsDir() {
		return nil, fmt.Errorf("TensorRT engine path %q is a directory", resolvedPath)
	}
	if info.Size() <= 0 {
		return nil, fmt.Errorf("TensorRT engine path %q is empty", resolvedPath)
	}
	bridgeCommand, bridgeErr := parseBridgeCommand(os.Getenv("APEXX_TRT_BRIDGE_CMD"))
	if bridgeErr != nil {
		return nil, fmt.Errorf("invalid APEXX_TRT_BRIDGE_CMD: %w", bridgeErr)
	}
	if len(bridgeCommand) == 0 {
		return nil, fmt.Errorf(
			"tensorrt adapter unavailable: build with -tags tensorrt and enable CGO, " +
				"or configure APEXX_TRT_BRIDGE_CMD",
		)
	}
	return &tensorRTBridgeAdapter{
		enginePath:    resolvedPath,
		bridgeCommand: bridgeCommand,
	}, nil
}

func (a *tensorRTBridgeAdapter) Name() string {
	return "tensorrt-python-bridge"
}

func (a *tensorRTBridgeAdapter) PredictBatch(
	ctx context.Context,
	reqs []PredictRequest,
) ([]PredictResponse, error) {
	bridgeResponses, bridgeErr := runBridgePredictBatch(
		ctx,
		a.bridgeCommand,
		bridgePredictRequest{
			Backend:      "tensorrt",
			ArtifactPath: a.enginePath,
			Requests:     reqs,
		},
		a.Name(),
	)
	if bridgeErr != nil {
		return nil, fmt.Errorf("tensorrt bridge predict failed: %w", bridgeErr)
	}
	return bridgeResponses, nil
}

func (a *tensorRTBridgeAdapter) Close() error {
	_ = a
	return nil
}
