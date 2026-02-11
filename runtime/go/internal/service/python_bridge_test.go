package service

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func TestParseBridgeCommand(t *testing.T) {
	t.Run("empty", func(t *testing.T) {
		parts, err := parseBridgeCommand("   ")
		if err != nil {
			t.Fatalf("parseBridgeCommand() error = %v", err)
		}
		if len(parts) != 0 {
			t.Fatalf("expected empty command, got %v", parts)
		}
	})

	t.Run("split", func(t *testing.T) {
		parts, err := parseBridgeCommand("python -m apex_x.runtime.service_bridge")
		if err != nil {
			t.Fatalf("parseBridgeCommand() error = %v", err)
		}
		want := []string{"python", "-m", "apex_x.runtime.service_bridge"}
		if !reflect.DeepEqual(parts, want) {
			t.Fatalf("unexpected command parts: got %v want %v", parts, want)
		}
	})
}

func TestDefaultRunBridgePredictBatchNoCommandIsUnavailable(t *testing.T) {
	_, err := defaultRunBridgePredictBatch(
		context.Background(),
		nil,
		bridgePredictRequest{},
		"onnxruntime-cpu-baseline",
	)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !errors.Is(err, ErrBackendUnavailable) {
		t.Fatalf("expected ErrBackendUnavailable, got %v", err)
	}
}
