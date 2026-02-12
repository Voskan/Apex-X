package service

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log/slog"
	"net/http"
	"runtime/debug"
	"time"
)

type contextKey string

const (
	RequestIDKey contextKey = "request_id"
)

// generateID creates a random 16-byte hex string.
func generateID() string {
	b := make([]byte, 16)
	_, err := rand.Read(b)
	if err != nil {
		// Fallback for extreme cases
		return fmt.Sprintf("fallback-%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

// Chain chains multiple middlewares.
// The first middleware in the list is the outer-most one (executed first).
func Chain(h http.Handler, middlewares ...func(http.Handler) http.Handler) http.Handler {
	for i := len(middlewares) - 1; i >= 0; i-- {
		h = middlewares[i](h)
	}
	return h
}

// RequestIDMiddleware adds a unique ID to the request context and response header.
func RequestIDMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqID := r.Header.Get("X-Request-ID")
		if reqID == "" {
			reqID = generateID()
		}
		w.Header().Set("X-Request-ID", reqID)
		ctx := context.WithValue(r.Context(), RequestIDKey, reqID)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// LoggingMiddleware logs the incoming request and its duration.
func LoggingMiddleware(logger *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			reqID, _ := r.Context().Value(RequestIDKey).(string)

			// Wrap ResponseWriter to capture status code
			ww := &responseWriterWrapper{ResponseWriter: w, statusCode: http.StatusOK}

			next.ServeHTTP(ww, r)

			// Log key request details
			logger.Info("http_request",
				"method", r.Method,
				"path", r.URL.Path,
				"status", ww.statusCode,
				"duration_ms", time.Since(start).Milliseconds(),
				"request_id", reqID,
				"remote_addr", r.RemoteAddr,
				"user_agent", r.UserAgent(),
			)
		})
	}
}

// RecoveryMiddleware recovers from panics and returns a 500 error.
func RecoveryMiddleware(logger *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if err := recover(); err != nil {
					reqID, _ := r.Context().Value(RequestIDKey).(string)
					logger.Error("http_panic_recovered",
						"error", fmt.Sprintf("%v", err),
						"stack", string(debug.Stack()),
						"request_id", reqID,
					)
					http.Error(w, "Internal Server Error", http.StatusInternalServerError)
				}
			}()
			next.ServeHTTP(w, r)
		})
	}
}

// responseWriterWrapper captures the status code.
type responseWriterWrapper struct {
	http.ResponseWriter
	statusCode int
}

func (w *responseWriterWrapper) WriteHeader(code int) {
	w.statusCode = code
	w.ResponseWriter.WriteHeader(code)
}
