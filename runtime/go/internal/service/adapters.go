package service

import "context"

type InferenceAdapter interface {
	Name() string
	PredictBatch(ctx context.Context, reqs []PredictRequest) ([]PredictResponse, error)
	Close() error
}
