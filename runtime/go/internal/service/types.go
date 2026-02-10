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

type PredictResponse struct {
	RequestID     string    `json:"request_id"`
	BudgetProfile string    `json:"budget_profile"`
	SelectedTiles int       `json:"selected_tiles"`
	Scores        []float32 `json:"scores"`
	Backend       string    `json:"backend"`
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
