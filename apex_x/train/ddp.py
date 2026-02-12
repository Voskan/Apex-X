"""Distributed Data Parallel (DDP) training wrapper.

Enables multi-GPU training with linear scaling for faster convergence.

Expected speedup: Nx with N GPUs (e.g., 8x with 8 GPUs).
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed training process group.
    
    Returns:
        Tuple of (world_size, rank, local_rank)
    """
    # Initialize process group from environment variables
    # Set by torchrun: WORLD_SIZE, RANK, LOCAL_RANK
    dist.init_process_group(backend='nccl')
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    
    return world_size, rank, local_rank


def cleanup_distributed() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


class DDPWrapper:
    """Wrapper for distributed data parallel training.
    
    Handles:
    - Process group initialization
    - Model wrapping with DDP
    - Distributed sampler for data loading
    - Gradient synchronization
    - Checkpoint saving (main process only)
    
    Usage:
        ```python
        ddp = DDPWrapper()
        model = ddp.wrap_model(model)
        train_loader = ddp.create_dataloader(dataset, batch_size=16)
        
        for epoch in range(epochs):
            ddp.set_epoch(epoch)  # Important for shuffling
            train_epoch(model, train_loader, ...)
            
            if ddp.is_main_process():
                save_checkpoint(...)
        
        ddp.cleanup()
        ```
    
    Args:
        find_unused_parameters: Whether to find unused parameters (default:  False)
        gradient_as_bucket_view: Use bucket view for gradients (default: True)
    """
    
    def __init__(
        self,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
    ) -> None:
        self.world_size, self.rank, self.local_rank = setup_distributed()
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        
        self.samplers = []  # Track samplers for epoch setting
    
    def wrap_model(self, model: torch.nn.Module) -> DDP:
        """Wrap model with DistributedDataParallel.
        
        Args:
            model: Model to wrap (should already be on correct device)
        
        Returns:
            DDP-wrapped model
        """
        # Move model to correct GPU
        model = model.to(self.local_rank)
        
        # Wrap with DDP
        model_ddp = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=self.gradient_as_bucket_view,
        )
        
        return model_ddp
    
    def create_dataloader(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        """Create distributed dataloader with DistributedSampler.
        
        Args:
            dataset: PyTorch dataset
            batch_size: Batch size PER GPU
            shuffle: Whether to shuffle (via sampler)
            num_workers: Number of data loading workers
            **kwargs: Additional DataLoader arguments
        
        Returns:
            DataLoader with DistributedSampler
        """
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
        )
        
        self.samplers.append(sampler)
        
        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            **kwargs
        )
        
        return loader
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for all samplers (important for proper shuffling).
        
        Args:
            epoch: Current epoch number
        """
        for sampler in self.samplers:
            sampler.set_epoch(epoch)
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0).
        
        Returns:
            True if rank == 0
        """
        return self.rank == 0
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        dist.barrier()
    
    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
        """All-reduce operation across all processes.
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation (SUM, AVG, etc.)
        
        Returns:
            Reduced tensor
        """
        dist.all_reduce(tensor, op=op)
        return tensor
    
    def gather(self, tensor: torch.Tensor, dst: int = 0) -> list[torch.Tensor] | None:
        """Gather tensors from all processes to destination process.
        
        Args:
            tensor: Tensor to gather
            dst: Destination rank (default: 0)
        
        Returns:
            List of tensors if dst==rank, else None
        """
        if self.rank == dst:
            gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.gather(tensor, gather_list, dst=dst)
            return gather_list
        else:
            dist.gather(tensor, dst=dst)
            return None
    
    def cleanup(self) -> None:
        """Cleanup distributed training."""
        cleanup_distributed()
    
    @property
    def device(self) -> torch.device:
        """Get the device for this process."""
        return torch.device(f'cuda:{self.local_rank}')


def reduce_dict(input_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Reduce dictionary of tensors across all processes (for logging).
    
    Args:
        input_dict: Dictionary of metric tensors
    
    Returns:
        Dictionary of averaged metrics
    """
    if not dist.is_initialized():
        return {k: v.item() if torch.is_tensor(v) else v for k, v in input_dict.items()}
    
    world_size = dist.get_world_size()
    
    reduced_dict = {}
    for k, v in input_dict.items():
        if torch.is_tensor(v):
            v_tensor = v.clone().detach()
            dist.all_reduce(v_tensor, op=dist.ReduceOp.SUM)
            reduced_dict[k] = (v_tensor / world_size).item()
        else:
            reduced_dict[k] = v
    
    return reduced_dict


__all__ = [
    'DDPWrapper',
    'setup_distributed',
    'cleanup_distributed',
    'is_main_process',
    'reduce_dict',
]
