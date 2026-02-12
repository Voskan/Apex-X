"""Trainer utilities - adds train_epoch method to ApexXTrainer."""

from dataclasses import asdict, is_dataclass
import shutil
import torch
from apex_x.utils import get_logger
from .train_losses import compute_teacher_training_loss

LOGGER = get_logger(__name__)


def train_epoch_impl(
    trainer,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int = 0,
    total_epochs: int = 300,
    log_interval: int = 50,
) -> dict[str, float]:
    """Train for one epoch with SimOTA loss.
    
    Args:
        trainer: ApexXTrainer instance
        train_loader: Training data loader
        optimizer: Optimizer
        epoch: Current epoch
        total_epochs: Total epochs (for progressive loss)
        log_interval: Log every N batches
    
    Returns:
        Dictionary of training metrics
    """
    trainer.teacher.train()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0
    total_quality_loss = 0.0
    num_batches = 0
    
    # Get device from model
    device = next(trainer.teacher.parameters()).device
    
    for batch_idx, batch in enumerate(train_loader):
        if not isinstance(batch, list):
            continue
        
        # Stack images
        images = []
        for sample in batch:
            if isinstance(sample.image, torch.Tensor):
                images.append(sample.image)
            else:
                images.append(torch.from_numpy(sample.image))
        
        if len(images) == 0:
            continue
        
        image_batch = torch.stack(images).to(device)
        
        # Ensure NCHW format
        if image_batch.ndim == 3:
            image_batch = image_batch.unsqueeze(0)
        if image_batch.shape[1] != 3:
            image_batch = image_batch.permute(0, 3, 1, 2)
        
        # Forward pass
        optimizer.zero_grad()
        output = trainer.teacher(image_batch)
        
        # Compute loss with SimOTA
        loss, loss_dict = compute_teacher_training_loss(
            output=output,
            samples=batch,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.teacher.parameters(), max_norm=10.0)
        optimizer.step()
        
        # Update EMA if enabled
        if trainer.teacher.use_ema:
            trainer.teacher.update_ema()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_cls_loss += loss_dict.get('cls_loss', 0.0)
        total_box_loss += loss_dict.get('box_loss', 0.0)
        total_quality_loss += loss_dict.get('quality_loss', 0.0)
        num_batches += 1
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            LOGGER.info(
                f"Epoch [{epoch+1}/{total_epochs}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss: {avg_loss:.4f} "
                f"(Cls: {total_cls_loss/num_batches:.4f}, "
                f"Box: {total_box_loss/num_batches:.4f})"
            )
    
    # Return metrics
    return {
        'total_loss': total_loss / max(1, num_batches),
        'cls_loss': total_cls_loss / max(1, num_batches),
        'box_loss': total_box_loss / max(1, num_batches),
        'quality_loss': total_quality_loss / max(1, num_batches),
        'num_batches': num_batches,
    }


def add_train_epoch_method(trainer_class):
    """Add train_epoch method to ApexXTrainer dynamically."""
    
    def train_epoch(self, *args, **kwargs):
        return train_epoch_impl(self, *args, **kwargs)
    
    def save_checkpoint(self, path, epoch=0, metrics=None):
        """Simple checkpoint wrapper."""
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        previous_checkpoint_dir = getattr(self, "checkpoint_dir", None)
        self.checkpoint_dir = path.parent
        trainable_params = [p for p in self.teacher.parameters() if p.requires_grad]
        if not trainable_params:
            trainable_params = list(self.teacher.parameters())
        checkpoint_optimizer = torch.optim.SGD(trainable_params, lr=0.0, momentum=0.0)
        self.save_training_checkpoint(
            epoch=epoch,
            step=0,
            optimizer=checkpoint_optimizer,
            metrics=metrics,
            is_best=path.name == "best.pt",
        )
        canonical_epoch_path = path.parent / f"epoch_{int(epoch):04d}.pt"
        if canonical_epoch_path.exists() and canonical_epoch_path.resolve() != path.resolve():
            shutil.copy2(canonical_epoch_path, path)
        self.checkpoint_dir = previous_checkpoint_dir
    
    def load_checkpoint(self, path):
        """Simple load wrapper."""
        metadata = self.load_training_checkpoint(path, device="cpu")
        if hasattr(metadata, "to_dict"):
            return metadata.to_dict()
        if is_dataclass(metadata):
            return asdict(metadata)
        if isinstance(metadata, dict):
            return metadata
        return {}
    
    trainer_class.train_epoch = train_epoch
    if not hasattr(trainer_class, 'save_checkpoint'):
        trainer_class.save_checkpoint = save_checkpoint
    if not hasattr(trainer_class, 'load_checkpoint'):
        trainer_class.load_checkpoint = load_checkpoint
    
    return trainer_class


__all__ = ['train_epoch_impl', 'add_train_epoch_method']
