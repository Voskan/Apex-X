//go:build !tensorrt || !cgo

package service

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestTensorRTStubAdapterRequiresBridgeCommand(t *testing.T) {
	tmpDir := t.TempDir()
	enginePath := filepath.Join(tmpDir, "engine.plan")
	if err := os.WriteFile(enginePath, []byte("engine"), 0o644); err != nil {
		t.Fatalf("failed to write engine file: %v", err)
	}
	t.Setenv("APEXX_TRT_BRIDGE_CMD", "")

	_, err := NewTensorRTAdapter(enginePath)
	if err == nil {
		t.Fatalf("expected adapter initialization error")
	}
}

func TestTensorRTStubAdapterUsesBridgeWhenConfigured(t *testing.T) {
	tmpDir := t.TempDir()
	enginePath := filepath.Join(tmpDir, "engine.plan")
	if err := os.WriteFile(enginePath, []byte("engine"), 0o644); err != nil {
		t.Fatalf("failed to write engine file: %v", err)
	}
	t.Setenv("APEXX_TRT_BRIDGE_CMD", "python -m apex_x.runtime.service_bridge")

	original := runBridgePredictBatch
	runBridgePredictBatch = func(
		_ context.Context,
		command []string,
		request bridgePredictRequest,
		defaultBackend string,
	) ([]PredictResponse, error) {
		if len(command) == 0 {
			t.Fatalf("bridge command should be configured")
		}
		if request.Backend != "tensorrt" {
			t.Fatalf("unexpected backend: %q", request.Backend)
		}
		if request.ArtifactPath != enginePath {
			t.Fatalf("unexpected artifact path: %q", request.ArtifactPath)
		}
		out := make([]PredictResponse, len(request.Requests))
		for idx, req := range request.Requests {
			out[idx] = PredictResponse{
				RequestID:     req.RequestID,
				BudgetProfile: req.BudgetProfile,
				SelectedTiles: 41,
				Scores:        []float32{0.93},
				Backend:       defaultBackend,
			}
		}
		return out, nil
	}
	t.Cleanup(func() {
		runBridgePredictBatch = original
	})

	adapter, err := NewTensorRTAdapter(enginePath)
	if err != nil {
		t.Fatalf("NewTensorRTAdapter() error = %v", err)
	}
	t.Cleanup(func() {
		_ = adapter.Close()
	})

	responses, predictErr := adapter.PredictBatch(context.Background(), []PredictRequest{
		{
			RequestID:     "r1",
			BudgetProfile: BudgetProfileBalanced,
			Input:         []float32{1.0, 3.0},
		},
	})
	if predictErr != nil {
		t.Fatalf("PredictBatch() error = %v", predictErr)
	}
	if len(responses) != 1 {
		t.Fatalf("unexpected response length: %d", len(responses))
	}
	if responses[0].Scores[0] != float32(0.93) {
		t.Fatalf("expected bridge score 0.93, got %.2f", responses[0].Scores[0])
	}
	if responses[0].SelectedTiles != 41 {
		t.Fatalf("expected bridge selected_tiles=41, got %d", responses[0].SelectedTiles)
	}
}

func TestTensorRTStubAdapterBridgeFailurePropagatesError(t *testing.T) {
	tmpDir := t.TempDir()
	enginePath := filepath.Join(tmpDir, "engine.plan")
	if err := os.WriteFile(enginePath, []byte("engine"), 0o644); err != nil {
		t.Fatalf("failed to write engine file: %v", err)
	}
	t.Setenv("APEXX_TRT_BRIDGE_CMD", "python -m apex_x.runtime.service_bridge")

	original := runBridgePredictBatch
	runBridgePredictBatch = func(
		_ context.Context,
		_ []string,
		_ bridgePredictRequest,
		_ string,
	) ([]PredictResponse, error) {
		return nil, fmt.Errorf("%w: bridge unavailable", ErrBackendUnavailable)
	}
	t.Cleanup(func() {
		runBridgePredictBatch = original
	})

	adapter, err := NewTensorRTAdapter(enginePath)
	if err != nil {
		t.Fatalf("NewTensorRTAdapter() error = %v", err)
	}
	t.Cleanup(func() {
		_ = adapter.Close()
	})

	_, predictErr := adapter.PredictBatch(context.Background(), []PredictRequest{
		{
			RequestID:     "r1",
			BudgetProfile: BudgetProfileBalanced,
			Input:         []float32{1.0, 3.0},
		},
	})
	if predictErr == nil {
		t.Fatalf("expected bridge error")
	}
	if !errors.Is(predictErr, ErrBackendUnavailable) {
		t.Fatalf("expected ErrBackendUnavailable, got %v", predictErr)
	}
}
