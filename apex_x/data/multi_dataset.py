"""Multi-dataset training support.

Enables training on multiple datasets simultaneously with balanced sampling.
"""

from __future__ import annotations

from typing import Iterator
import random

import torch
from torch.utils.data import Dataset, Sampler


class MultiDatasetSampler(Sampler):
    """Sampler for balanced multi-dataset training.
    
    Samples from multiple datasets with equal probability per dataset,
    ensuring balanced representation during training.
    
    Args:
        dataset_lengths: List of dataset lengths
        samples_per_epoch: Number of samples per epoch
        shuffle: Whether to shuffle within datasets
        seed: Random seed
    """
    
    def __init__(
        self,
        dataset_lengths: list[int],
        samples_per_epoch: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset_lengths = dataset_lengths
        self.samples_per_epoch = samples_per_epoch
        self.shuffle = shuffle
        self.seed = seed
        
        self.num_datasets = len(dataset_lengths)
        self.epoch = 0
        
        # Compute cumulative offsets for indexing
        self.offsets = [0]
        for length in dataset_lengths:
            self.offsets.append(self.offsets[-1] + length)
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch."""
        # Set RNG seed for reproducibility
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices = []
        
        # Create per-dataset index pools
        dataset_indices = []
        for i, length in enumerate(self.dataset_lengths):
            # Generate indices for this dataset
            ds_indices = list(range(self.offsets[i], self.offsets[i] + length))
            
            if self.shuffle:
                # Shuffle this dataset
                random.Random(self.seed + self.epoch + i).shuffle(ds_indices)
            
            dataset_indices.append(ds_indices)
        
        # Sample with equal probability from each dataset
        dataset_counters = [0] * self.num_datasets
        
        for _ in range(self.samples_per_epoch):
            # Choose random dataset
            dataset_idx = random.Random().randint(0, self.num_datasets - 1)
            
            # Get next sample from this dataset (circular)
            counter = dataset_counters[dataset_idx]
            sample_idx = dataset_indices[dataset_idx][counter % len(dataset_indices[dataset_idx])]
            
            indices.append(sample_idx)
            dataset_counters[dataset_idx] += 1
        
        self.epoch += 1
        return iter(indices)
    
    def __len__(self) -> int:
        return self.samples_per_epoch


class MultiDataset(Dataset):
    """Wrapper for multiple datasets.
    
    Combines multiple datasets into one, with optional per-dataset
    transformations and filtering.
    
    Args:
        datasets: List of datasets to combine
        dataset_names: Optional names for each dataset
    """
    
    def __init__(
        self,
        datasets: list[Dataset],
        dataset_names: list[str] | None = None,
    ):
        self.datasets = datasets
        self.dataset_names = dataset_names or [f"dataset_{i}" for i in range(len(datasets))]
        
        # Compute cumulative lengths
        self.cumulative_lengths = [0]
        for ds in datasets:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))
        
        self.total_length = self.cumulative_lengths[-1]
    
    def __len__(self) -> int:
        return self.total_length
    
    def __getitem__(self, idx: int):
        """Get item from appropriate dataset."""
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range [0, {self.total_length})")
        
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths[1:]):
            if idx < cum_len:
                dataset_idx = i
                break
        
        # Get relative index within dataset
        relative_idx = idx - self.cumulative_lengths[dataset_idx]
        
        # Get sample from dataset
        sample = self.datasets[dataset_idx][relative_idx]
        
        # Add dataset metadata
        if isinstance(sample, dict):
            sample['dataset_name'] = self.dataset_names[dataset_idx]
            sample['dataset_idx'] = dataset_idx
        
        return sample


__all__ = ['MultiDataset', 'MultiDatasetSampler']
