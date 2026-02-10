package service

import (
	"fmt"
	"sync/atomic"
	"time"
)

type Metrics struct {
	requestsTotal     atomic.Int64
	requestsFailed    atomic.Int64
	batchesTotal      atomic.Int64
	batchErrorsTotal  atomic.Int64
	batchItemsTotal   atomic.Int64
	batchSizeMax      atomic.Int64
	inflight          atomic.Int64
	latencyNanos      atomic.Int64
	queueLatencyNanos atomic.Int64
	queueLatencyMax   atomic.Int64
	inferenceNanos    atomic.Int64
	inferenceNanosMax atomic.Int64
}

type MetricsSnapshot struct {
	RequestsTotal      int64
	RequestsFailed     int64
	BatchesTotal       int64
	BatchErrorsTotal   int64
	BatchItemsTotal    int64
	BatchSizeMax       int64
	InFlight           int64
	AvgLatencyMillis   float64
	AvgQueueMillis     float64
	MaxQueueMillis     float64
	AvgInferenceMillis float64
	MaxInferenceMillis float64
}

func (m *Metrics) RecordRequestStart() {
	m.requestsTotal.Add(1)
	m.inflight.Add(1)
}

func (m *Metrics) RecordRequestDone(latency time.Duration, success bool) {
	m.inflight.Add(-1)
	m.latencyNanos.Add(latency.Nanoseconds())
	if !success {
		m.requestsFailed.Add(1)
	}
}

func (m *Metrics) RecordBatch() {
	m.batchesTotal.Add(1)
}

func (m *Metrics) RecordBatchStats(
	batchSize int,
	avgQueueWait time.Duration,
	inferenceTime time.Duration,
	success bool,
) {
	if batchSize < 0 {
		batchSize = 0
	}
	m.batchesTotal.Add(1)
	m.batchItemsTotal.Add(int64(batchSize))
	updateAtomicMax(&m.batchSizeMax, int64(batchSize))

	queueNanos := avgQueueWait.Nanoseconds()
	if queueNanos < 0 {
		queueNanos = 0
	}
	m.queueLatencyNanos.Add(queueNanos)
	updateAtomicMax(&m.queueLatencyMax, queueNanos)

	inferNanos := inferenceTime.Nanoseconds()
	if inferNanos < 0 {
		inferNanos = 0
	}
	m.inferenceNanos.Add(inferNanos)
	updateAtomicMax(&m.inferenceNanosMax, inferNanos)

	if !success {
		m.batchErrorsTotal.Add(1)
	}
}

func (m *Metrics) Snapshot() MetricsSnapshot {
	requestCount := m.requestsTotal.Load()
	latencyNanos := m.latencyNanos.Load()
	avgMillis := 0.0
	if requestCount > 0 {
		avgMillis = float64(latencyNanos) / float64(requestCount) / float64(time.Millisecond)
	}
	batchCount := m.batchesTotal.Load()
	avgQueue := 0.0
	avgInference := 0.0
	if batchCount > 0 {
		avgQueue = float64(m.queueLatencyNanos.Load()) / float64(batchCount) / float64(time.Millisecond)
		avgInference = float64(m.inferenceNanos.Load()) / float64(batchCount) / float64(time.Millisecond)
	}
	return MetricsSnapshot{
		RequestsTotal:      requestCount,
		RequestsFailed:     m.requestsFailed.Load(),
		BatchesTotal:       batchCount,
		BatchErrorsTotal:   m.batchErrorsTotal.Load(),
		BatchItemsTotal:    m.batchItemsTotal.Load(),
		BatchSizeMax:       m.batchSizeMax.Load(),
		InFlight:           m.inflight.Load(),
		AvgLatencyMillis:   avgMillis,
		AvgQueueMillis:     avgQueue,
		MaxQueueMillis:     float64(m.queueLatencyMax.Load()) / float64(time.Millisecond),
		AvgInferenceMillis: avgInference,
		MaxInferenceMillis: float64(m.inferenceNanosMax.Load()) / float64(time.Millisecond),
	}
}

func (s MetricsSnapshot) PrometheusText() string {
	avgBatchSize := 0.0
	if s.BatchesTotal > 0 {
		avgBatchSize = float64(s.BatchItemsTotal) / float64(s.BatchesTotal)
	}
	return fmt.Sprintf(
		"apexx_requests_total %d\n"+
			"apexx_requests_failed_total %d\n"+
			"apexx_batches_total %d\n"+
			"apexx_batch_errors_total %d\n"+
			"apexx_batch_size_avg %.6f\n"+
			"apexx_batch_size_max %d\n"+
			"apexx_inflight %d\n"+
			"apexx_request_latency_ms_avg %.6f\n"+
			"apexx_queue_latency_ms_avg %.6f\n"+
			"apexx_queue_latency_ms_max %.6f\n"+
			"apexx_inference_latency_ms_avg %.6f\n"+
			"apexx_inference_latency_ms_max %.6f\n",
		s.RequestsTotal,
		s.RequestsFailed,
		s.BatchesTotal,
		s.BatchErrorsTotal,
		avgBatchSize,
		s.BatchSizeMax,
		s.InFlight,
		s.AvgLatencyMillis,
		s.AvgQueueMillis,
		s.MaxQueueMillis,
		s.AvgInferenceMillis,
		s.MaxInferenceMillis,
	)
}

func updateAtomicMax(target *atomic.Int64, value int64) {
	for {
		current := target.Load()
		if value <= current {
			return
		}
		if target.CompareAndSwap(current, value) {
			return
		}
	}
}
