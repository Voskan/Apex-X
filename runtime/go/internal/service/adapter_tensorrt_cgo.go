//go:build tensorrt && cgo

package service

/*
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int64_t size_bytes;
  char* path;
} apexx_trt_engine_handle_t;

static int apexx_trt_engine_load(const char* path, apexx_trt_engine_handle_t** out_handle, char** out_err) {
  if (path == NULL || out_handle == NULL) {
    if (out_err != NULL) {
      *out_err = strdup("invalid loader arguments");
    }
    return 0;
  }
  FILE* f = fopen(path, "rb");
  if (f == NULL) {
    if (out_err != NULL) {
      *out_err = strdup("failed to open TensorRT engine file");
    }
    return 0;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    if (out_err != NULL) {
      *out_err = strdup("failed to seek TensorRT engine file");
    }
    return 0;
  }
  long size = ftell(f);
  fclose(f);
  if (size <= 0) {
    if (out_err != NULL) {
      *out_err = strdup("TensorRT engine file is empty");
    }
    return 0;
  }
  apexx_trt_engine_handle_t* handle = (apexx_trt_engine_handle_t*)malloc(sizeof(apexx_trt_engine_handle_t));
  if (handle == NULL) {
    if (out_err != NULL) {
      *out_err = strdup("failed to allocate engine handle");
    }
    return 0;
  }
  handle->size_bytes = (int64_t)size;
  handle->path = strdup(path);
  *out_handle = handle;
  return 1;
}

static int64_t apexx_trt_engine_size(const apexx_trt_engine_handle_t* handle) {
  if (handle == NULL) {
    return 0;
  }
  return handle->size_bytes;
}

static void apexx_trt_engine_free(apexx_trt_engine_handle_t* handle) {
  if (handle == NULL) {
    return;
  }
  if (handle->path != NULL) {
    free(handle->path);
  }
  free(handle);
}
*/
import "C"

import (
	"context"
	"fmt"
	"os"
	"sync"
	"unsafe"
)

type TensorRTAdapter struct {
	enginePath    string
	engineSize    int64
	handle        *C.apexx_trt_engine_handle_t
	closeOnce     sync.Once
	bridgeCommand []string
}

func NewTensorRTAdapter(enginePath string) (InferenceAdapter, error) {
	resolvedPath, err := resolveModelPath(
		enginePath,
		"APEXX_TRT_ENGINE_PATH",
		"tensorrt engine",
	)
	if err != nil {
		return nil, err
	}

	cPath := C.CString(resolvedPath)
	defer C.free(unsafe.Pointer(cPath))

	var handle *C.apexx_trt_engine_handle_t
	var cErr *C.char
	if C.apexx_trt_engine_load(cPath, &handle, &cErr) == 0 {
		defer func() {
			if cErr != nil {
				C.free(unsafe.Pointer(cErr))
			}
		}()
		if cErr != nil {
			return nil, fmt.Errorf("failed to load TensorRT engine: %s", C.GoString(cErr))
		}
		return nil, fmt.Errorf("failed to load TensorRT engine")
	}
	if handle == nil {
		return nil, fmt.Errorf("failed to load TensorRT engine: nil handle")
	}
	bridgeCommand, bridgeErr := parseBridgeCommand(os.Getenv("APEXX_TRT_BRIDGE_CMD"))
	if bridgeErr != nil {
		C.apexx_trt_engine_free(handle)
		return nil, fmt.Errorf("invalid APEXX_TRT_BRIDGE_CMD: %w", bridgeErr)
	}

	return &TensorRTAdapter{
		enginePath:    resolvedPath,
		engineSize:    int64(C.apexx_trt_engine_size(handle)),
		handle:        handle,
		bridgeCommand: bridgeCommand,
	}, nil
}

func (a *TensorRTAdapter) Name() string {
	return "tensorrt-cgo-baseline"
}

func (a *TensorRTAdapter) PredictBatch(
	ctx context.Context,
	reqs []PredictRequest,
) ([]PredictResponse, error) {
	if a.handle == nil {
		return nil, fmt.Errorf("TensorRT adapter is closed")
	}
	if len(a.bridgeCommand) == 0 {
		return nil, fmt.Errorf(
			"%w: tensorrt bridge command is not configured; set APEXX_TRT_BRIDGE_CMD",
			ErrBackendUnavailable,
		)
	}
	bridgeResponses, bridgeErr := runBridgePredictBatch(
		ctx,
		a.bridgeCommand,
		bridgePredictRequest{
			Backend:      "tensorrt",
			ArtifactPath: a.enginePath,
			Requests:     reqs,
		},
		a.Name(),
	)
	if bridgeErr != nil {
		return nil, fmt.Errorf("tensorrt bridge predict failed: %w", bridgeErr)
	}
	return bridgeResponses, nil
}

func (a *TensorRTAdapter) Close() error {
	a.closeOnce.Do(func() {
		if a.handle != nil {
			C.apexx_trt_engine_free(a.handle)
			a.handle = nil
		}
	})
	return nil
}
