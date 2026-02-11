package service

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestNewORTAdapterLoadsModelFile(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := filepath.Join(tmpDir, "model.onnx")
	if err := os.WriteFile(modelPath, []byte("onnx"), 0o644); err != nil {
		t.Fatalf("failed to write model file: %v", err)
	}
	t.Setenv("APEXX_ORT_BRIDGE_CMD", "python -m apex_x.runtime.service_bridge")
	original := runBridgePredictBatch
	runBridgePredictBatch = func(
		_ context.Context,
		_ []string,
		request bridgePredictRequest,
		defaultBackend string,
	) ([]PredictResponse, error) {
		out := make([]PredictResponse, len(request.Requests))
		for idx, req := range request.Requests {
			out[idx] = PredictResponse{
				RequestID:     req.RequestID,
				BudgetProfile: req.BudgetProfile,
				SelectedTiles: 32,
				Scores:        []float32{0.5},
				Backend:       defaultBackend,
			}
		}
		return out, nil
	}
	t.Cleanup(func() {
		runBridgePredictBatch = original
	})

	adapter, err := NewORTAdapter(modelPath)
	if err != nil {
		t.Fatalf("NewORTAdapter() error = %v", err)
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
	if responses[0].Backend != "onnxruntime-cpu-baseline" {
		t.Fatalf("unexpected backend: %q", responses[0].Backend)
	}
}

func TestORTAdapterPredictRequiresBridgeCommand(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := filepath.Join(tmpDir, "model.onnx")
	if err := os.WriteFile(modelPath, []byte("onnx"), 0o644); err != nil {
		t.Fatalf("failed to write model file: %v", err)
	}
	t.Setenv("APEXX_ORT_BRIDGE_CMD", "")
	adapter, err := NewORTAdapter(modelPath)
	if err != nil {
		t.Fatalf("NewORTAdapter() error = %v", err)
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
		t.Fatalf("expected unavailable error when bridge command is missing")
	}
	if !errors.Is(predictErr, ErrBackendUnavailable) {
		t.Fatalf("expected ErrBackendUnavailable, got %v", predictErr)
	}
}

func TestNewORTAdapterMissingFileFails(t *testing.T) {
	_, err := NewORTAdapter(filepath.Join(t.TempDir(), "missing.onnx"))
	if err == nil {
		t.Fatalf("expected error for missing model path")
	}
}

func TestNewORTAdapterUsesEnvPath(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := filepath.Join(tmpDir, "env_model.onnx")
	if err := os.WriteFile(modelPath, []byte("onnx"), 0o644); err != nil {
		t.Fatalf("failed to write model file: %v", err)
	}

	t.Setenv("APEXX_ORT_MODEL_PATH", modelPath)
	adapter, err := NewORTAdapter("")
	if err != nil {
		t.Fatalf("NewORTAdapter() env error = %v", err)
	}
	t.Cleanup(func() {
		_ = adapter.Close()
	})
}

func TestORTAdapterUsesBridgeWhenConfigured(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := filepath.Join(tmpDir, "model.onnx")
	if err := os.WriteFile(modelPath, []byte("onnx"), 0o644); err != nil {
		t.Fatalf("failed to write model file: %v", err)
	}

	t.Setenv("APEXX_ORT_BRIDGE_CMD", "python -m apex_x.runtime.service_bridge")
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
		if request.Backend != "onnxruntime" {
			t.Fatalf("unexpected bridge backend: %q", request.Backend)
		}
		if request.ArtifactPath != modelPath {
			t.Fatalf("unexpected bridge artifact path: %q", request.ArtifactPath)
		}
		out := make([]PredictResponse, len(request.Requests))
		for idx, req := range request.Requests {
			out[idx] = PredictResponse{
				RequestID:     req.RequestID,
				BudgetProfile: req.BudgetProfile,
				SelectedTiles: 99,
				Scores:        []float32{0.77},
				Backend:       defaultBackend,
			}
		}
		return out, nil
	}
	t.Cleanup(func() {
		runBridgePredictBatch = original
	})

	adapter, err := NewORTAdapter(modelPath)
	if err != nil {
		t.Fatalf("NewORTAdapter() error = %v", err)
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
	if responses[0].Scores[0] != float32(0.77) {
		t.Fatalf("expected bridge score 0.77, got %.2f", responses[0].Scores[0])
	}
	if responses[0].SelectedTiles != 99 {
		t.Fatalf("expected bridge selected_tiles=99, got %d", responses[0].SelectedTiles)
	}
}

func TestORTAdapterBridgeFailurePropagatesError(t *testing.T) {
	tmpDir := t.TempDir()
	modelPath := filepath.Join(tmpDir, "model.onnx")
	if err := os.WriteFile(modelPath, []byte("onnx"), 0o644); err != nil {
		t.Fatalf("failed to write model file: %v", err)
	}

	t.Setenv("APEXX_ORT_BRIDGE_CMD", "python -m apex_x.runtime.service_bridge")
	original := runBridgePredictBatch
	runBridgePredictBatch = func(
		_ context.Context,
		_ []string,
		_ bridgePredictRequest,
		_ string,
	) ([]PredictResponse, error) {
		return nil, fmt.Errorf("bridge failed")
	}
	t.Cleanup(func() {
		runBridgePredictBatch = original
	})

	adapter, err := NewORTAdapter(modelPath)
	if err != nil {
		t.Fatalf("NewORTAdapter() error = %v", err)
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
}
