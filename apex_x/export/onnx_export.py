"""ONNX export utilities for production deployment.

Exports trained models to ONNX format for cross-platform inference.
"""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
import numpy as np


class DictionaryWrapper(nn.Module):
    """Wraps a model that returns a dict to return a tuple for ONNX."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.output_keys = []
        
    def forward(self, x: torch.Tensor):
        out = self.model(x)
        if isinstance(out, dict):
            if not self.output_keys:
                self.output_keys = sorted(out.keys())
            return tuple(out[k] for k in self.output_keys)
        return out

def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    *,
    input_shape: tuple[int, int, int, int] = (1, 3, 1024, 1024),
    opset_version: int = 17,
    dynamic_axes: dict | None = None,
    verbose: bool = True,
) -> None:
    """Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (B, C, H, W)
        opset_version: ONNX opset version (default: 17)
        dynamic_axes: Dynamic axes configuration
        verbose: Enable verbose logging
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set model to eval mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Check for dict output and wrap if necessary
    with torch.no_grad():
        test_out = model(dummy_input)
    
    export_model = model
    if isinstance(test_out, dict):
        if verbose:
            print(f"📦 Detected dictionary output with keys: {sorted(test_out.keys())}")
            print("   Wrapping model for ONNX tuple-output compatibility.")
        export_model = DictionaryWrapper(model)
        # Initialize keys
        export_model(dummy_input)
        output_names = export_model.output_keys
    else:
        output_names = ['output']

    # Default dynamic axes (batch size and resolution)
    if dynamic_axes is None:
        dynamic_axes = {'input': {0: 'batch_size', 2: 'height', 3: 'width'}}
        for i, name in enumerate(output_names):
            dynamic_axes[name] = {0: 'batch_size'}
    
    # Export to ONNX
    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )
    
    if verbose:
        print(f"✅ Model exported to ONNX: {output_path}")
        print(f"   Input shape: {input_shape}")
        print(f"   Opset version: {opset_version}")


def verify_onnx_model(
    onnx_path: str | Path,
    pytorch_model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 1024, 1024),
    tolerance: float = 1e-5,
) -> bool:
    """Verify ONNX model outputs match PyTorch model.
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        input_shape: Input tensor shape
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if outputs match within tolerance
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️ onnxruntime not installed, skipping verification")
        return False
    
    # Create test input
    test_input = torch.randn(*input_shape)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
    
    # ONNX inference
    ort_session = ort.InferenceSession(str(onnx_path))
    onnx_output = ort_session.run(
        None,
        {'input': test_input.numpy()},
    )[0]
    
    # Compare outputs
    if isinstance(pytorch_output, dict):
        # Extract main output (detection/segmentation)
        pytorch_output = pytorch_output.get('det') or pytorch_output.get('masks')
    
    if isinstance(pytorch_output, torch.Tensor):
        pytorch_output = pytorch_output.numpy()
    
    # Compute difference
    diff = np.abs(pytorch_output - onnx_output).max()
    
    matches = diff < tolerance
    
    if matches:
        print(f"✅ ONNX verification passed (max diff: {diff:.2e})")
    else:
        print(f"❌ ONNX verification failed (max diff: {diff:.2e} > {tolerance:.2e})")
    
    return matches


def optimize_onnx_model(
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """Optimize ONNX model for faster inference.
    
    Applies graph optimizations like:
    - Constant folding
    - Dead code elimination
    - Operator fusion
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model
    """
    try:
        import onnx
        from onnxoptimizer import optimize
    except ImportError:
        print("⚠️ onnx/onnxoptimizer not installed, skipping optimization")
        return
    
    # Load model
    model = onnx.load(str(input_path))
    
    # Optimize
    optimized_model = optimize(model)
    
    # Save
    onnx.save(optimized_model, str(output_path))
    
    print(f"✅ Optimized ONNX model saved to: {output_path}")


__all__ = ['export_to_onnx', 'verify_onnx_model', 'optimize_onnx_model']
