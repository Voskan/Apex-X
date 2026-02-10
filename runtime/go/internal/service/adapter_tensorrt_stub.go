//go:build !tensorrt || !cgo

package service

import "fmt"

func NewTensorRTAdapter(_ string) (InferenceAdapter, error) {
	return nil, fmt.Errorf(
		"tensorrt adapter unavailable: build with -tags tensorrt and enable CGO",
	)
}
