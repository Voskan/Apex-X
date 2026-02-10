package service

import (
	"context"
	"time"
)

type TelemetryHooks interface {
	OnHTTPRequestStart(ctx context.Context, route string, requestID string)
	OnHTTPRequestDone(
		ctx context.Context,
		route string,
		requestID string,
		statusCode int,
		duration time.Duration,
		err error,
	)
	OnBatch(
		ctx context.Context,
		batchSize int,
		avgQueueWait time.Duration,
		inferenceTime time.Duration,
		err error,
	)
}

type NopTelemetryHooks struct{}

func (NopTelemetryHooks) OnHTTPRequestStart(
	_ context.Context,
	_ string,
	_ string,
) {
}

func (NopTelemetryHooks) OnHTTPRequestDone(
	_ context.Context,
	_ string,
	_ string,
	_ int,
	_ time.Duration,
	_ error,
) {
}

func (NopTelemetryHooks) OnBatch(
	_ context.Context,
	_ int,
	_ time.Duration,
	_ time.Duration,
	_ error,
) {
}
