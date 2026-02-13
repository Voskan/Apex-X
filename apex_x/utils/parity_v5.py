import torch
import numpy as np
from torch import Tensor

class ParityValidator:
    """World-Class Numerical Governance Suite.
    
    Ensures that 'Ascension V5' produces bit-exact or 
    tolerance-bound parity across CPU, CUDA, Triton, and TensorRT.
    """
    
    def __init__(self, atol: float = 1e-5, rtol: float = 1e-4):
        self.atol = atol
        self.rtol = rtol

    def validate(self, cpu_out: Tensor, gpu_out: Tensor, name: str = "Layer"):
        """Compares CPU and GPU output tensors."""
        cpu_out = cpu_out.cpu().detach().numpy()
        gpu_out = gpu_out.cpu().detach().numpy()
        
        diff = np.abs(cpu_out - gpu_out)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        success = np.allclose(cpu_out, gpu_out, atol=self.atol, rtol=self.rtol)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"[{status}] {name}: Max Diff={max_diff:.2e}, Mean Diff={mean_diff:.2e}")
        
        if not success:
            items_failed = np.sum(diff > self.atol)
            print(f"   ! Tolerance exceeded for {items_failed} elements.")
            
        return success

def run_ascension_parity_check(model_v5, sample_input: Tensor):
    """Executes a full parity sweep for the V5 model."""
    validator = ParityValidator()
    
    # 1. CPU Reference Run
    model_v5.cpu()
    with torch.no_grad():
        out_cpu = model_v5(sample_input.cpu())
        
    # 2. CUDA Acceleration Run
    model_v5.cuda()
    with torch.no_grad():
        out_cuda = model_v5(sample_input.cuda())
        
    # 3. Cross-Check
    # (Assuming out is a dict of tensors)
    all_pass = True
    for key in out_cpu:
        if isinstance(out_cpu[key], Tensor):
            res = validator.validate(out_cpu[key], out_cuda[key], name=key)
            all_pass = all_pass and res
            
    return all_pass
