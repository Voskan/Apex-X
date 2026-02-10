package service

import (
	"context"
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
