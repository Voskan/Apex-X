package service

import "errors"

var (
	ErrBackendUnavailable = errors.New("inference backend unavailable")
	ErrBackendInference   = errors.New("inference backend inference failed")
	ErrBackendProtocol    = errors.New("inference backend protocol failed")
)
