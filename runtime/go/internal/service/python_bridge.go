package service

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
)

type bridgePredictResult struct {
	RequestID     string    `json:"request_id"`
	SelectedTiles *int      `json:"selected_tiles,omitempty"`
	Scores        []float32 `json:"scores"`
	Backend       string    `json:"backend,omitempty"`
}

type bridgePredictResponse struct {
	Results []bridgePredictResult `json:"results"`
	Error   string                `json:"error,omitempty"`
}

type bridgePredictRequest struct {
	Backend      string           `json:"backend"`
	ArtifactPath string           `json:"artifact_path"`
	Requests     []PredictRequest `json:"requests"`
}

type bridgePredictBatchFn func(
	ctx context.Context,
	command []string,
	request bridgePredictRequest,
	defaultBackend string,
) ([]PredictResponse, error)

var runBridgePredictBatch bridgePredictBatchFn = defaultRunBridgePredictBatch

func parseBridgeCommand(raw string) ([]string, error) {
	clean := strings.TrimSpace(raw)
	if clean == "" {
		return nil, nil
	}
	parts := strings.Fields(clean)
	if len(parts) == 0 {
		return nil, fmt.Errorf("bridge command is empty")
	}
	return parts, nil
}

func defaultRunBridgePredictBatch(
	ctx context.Context,
	command []string,
	request bridgePredictRequest,
	defaultBackend string,
) ([]PredictResponse, error) {
	if len(command) == 0 {
		return nil, fmt.Errorf("%w: bridge command is not configured", ErrBackendUnavailable)
	}
	payload, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("%w: failed to encode bridge request: %w", ErrBackendProtocol, err)
	}
	cmd := exec.CommandContext(ctx, command[0], command[1:]...)
	cmd.Stdin = bytes.NewReader(payload)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if runErr := cmd.Run(); runErr != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		errText := strings.TrimSpace(stderr.String())
		var execErr *exec.Error
		var pathErr *os.PathError
		if errors.As(runErr, &execErr) || errors.As(runErr, &pathErr) {
			if errText == "" {
				return nil, fmt.Errorf("%w: bridge command failed: %w", ErrBackendUnavailable, runErr)
			}
			return nil, fmt.Errorf(
				"%w: bridge command failed: %w: %s",
				ErrBackendUnavailable,
				runErr,
				errText,
			)
		}
		if errText == "" {
			return nil, fmt.Errorf("%w: bridge command failed: %w", ErrBackendInference, runErr)
		}
		return nil, fmt.Errorf("%w: bridge command failed: %w: %s", ErrBackendInference, runErr, errText)
	}
	var decoded bridgePredictResponse
	if err := json.Unmarshal(stdout.Bytes(), &decoded); err != nil {
		return nil, fmt.Errorf("%w: failed to decode bridge response: %w", ErrBackendProtocol, err)
	}
	if strings.TrimSpace(decoded.Error) != "" {
		return nil, fmt.Errorf(
			"%w: bridge runtime error: %s",
			ErrBackendInference,
			strings.TrimSpace(decoded.Error),
		)
	}

	byID := make(map[string]bridgePredictResult, len(decoded.Results))
	for _, result := range decoded.Results {
		byID[result.RequestID] = result
	}
	if len(byID) == 0 && len(request.Requests) != 0 {
		return nil, fmt.Errorf("%w: bridge returned no results", ErrBackendProtocol)
	}

	out := make([]PredictResponse, len(request.Requests))
	for idx, req := range request.Requests {
		profile, profileErr := normalizeBudgetProfile(req.BudgetProfile, BudgetProfileBalanced)
		if profileErr != nil {
			return nil, profileErr
		}
		bridgeResult, ok := byID[req.RequestID]
		if !ok {
			return nil, fmt.Errorf(
				"%w: bridge response missing request_id=%q",
				ErrBackendProtocol,
				req.RequestID,
			)
		}
		backend := strings.TrimSpace(bridgeResult.Backend)
		if backend == "" {
			backend = defaultBackend
		}
		selectedTiles := selectedTilesForProfile(profile)
		if bridgeResult.SelectedTiles != nil {
			selectedTiles = *bridgeResult.SelectedTiles
		}
		scores := bridgeResult.Scores
		if len(scores) == 0 {
			scores = []float32{0.0}
		}
		out[idx] = PredictResponse{
			RequestID:     req.RequestID,
			BudgetProfile: profile,
			SelectedTiles: selectedTiles,
			Scores:        scores,
			Backend:       backend,
		}
	}
	return out, nil
}
