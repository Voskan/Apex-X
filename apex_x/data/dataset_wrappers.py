"""Dataset wrappers for augmentation and adaptation."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from torch.utils.data import Dataset

from apex_x.data.transforms import TransformSample


class TransformProtocol(Protocol):
    def __call__(
        self,
        sample: TransformSample,
        *,
        rng: np.random.RandomState | None = None,
    ) -> TransformSample: ...


class AugmentedDataset(Dataset):
    """Wraps a dataset and applies geometric/pixel transformations.
    
    This allows separating dataset loading from augmentation, which is critical
    when augmentations (like CopyPaste/Mosaic) need access to the dataset itself.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        transform: TransformProtocol | None = None,
        seed: int = 42,
    ) -> None:
        """Initialize augmented dataset.
        
        Args:
            dataset: Underlying dataset returning TransformSample
            transform: Transformation pipeline to apply
            seed: Random seed for augmentations
        """
        self.dataset = dataset
        self.transform = transform
        self.seed = seed
        
    def __len__(self) -> int:
        return len(self.dataset)
        
    def __getitem__(self, idx: int) -> TransformSample:
        # Get raw sample
        sample = self.dataset[idx]
        
        # Apply transforms if present
        if self.transform is not None:
            # Deterministic RNG based on index + seed for reproducibility
            # but usually we want randomness across epochs. 
            # PyTorch dataloaders handle seeding of workers.
            # Here we just use a fresh RNG state or rely on global state.
            # Using RandomState ensures we control the source.
            rng = np.random.RandomState() 
            sample = self.transform(sample, rng=rng)
            
        return sample
        
    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to underlying dataset."""
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
