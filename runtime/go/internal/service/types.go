package service

import (
	"fmt"
	"strings"
)

const (
	BudgetProfileQuality  = "quality"
	BudgetProfileBalanced = "balanced"
	BudgetProfileEdge     = "edge"
)

var budgetProfiles = map[string]struct{}{
	BudgetProfileQuality:  {},
	BudgetProfileBalanced: {},
	BudgetProfileEdge:     {},
}

type PredictRequest struct {
	RequestID     string            `json:"request_id,omitempty"`
	Input         []float32         `json:"input,omitempty"`
	BudgetProfile string            `json:"budget_profile,omitempty"`
	Metadata      map[string]string `json:"metadata,omitempty"`
}

type RuntimeLatencyMillis struct {
	Total            float64 `json:"total"`
	BackendExecute   float64 `json:"backend_execute"`
	BackendPreflight float64 `json:"backend_preflight"`
}

type RuntimeMetadata struct {
	RequestedBackend        string               `json:"requested_backend"`
	SelectedBackend         string               `json:"selected_backend"`
	ExecutionBackend        string               `json:"execution_backend"`
	FallbackPolicy          string               `json:"fallback_policy"`
	PrecisionProfile        string               `json:"precision_profile"`
	SelectionFallbackReason *string              `json:"selection_fallback_reason"`
	ExecutionFallbackReason *string              `json:"execution_fallback_reason"`
	LatencyMS               RuntimeLatencyMillis `json:"latency_ms"`
}

type PredictResponse struct {
	RequestID     string          `json:"request_id"`
	BudgetProfile string          `json:"budget_profile"`
	SelectedTiles int             `json:"selected_tiles"`
	Scores        []float32       `json:"scores"`
	Backend       string          `json:"backend"`
	Runtime       RuntimeMetadata `json:"runtime"`
}

func normalizeBudgetProfile(profile string, fallback string) (string, error) {
	cleanProfile := strings.ToLower(strings.TrimSpace(profile))
	if cleanProfile == "" {
		cleanProfile = strings.ToLower(strings.TrimSpace(fallback))
	}
	if _, ok := budgetProfiles[cleanProfile]; !ok {
		return "", fmt.Errorf("unsupported budget profile %q", profile)
	}
	return cleanProfile, nil
}
