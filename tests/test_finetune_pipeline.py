import torch
import pytest
from apex_x.data.miner import DataMiner, compute_entropy
from apex_x.train.pseudo_label import PseudoLabeler

def test_compute_entropy():
    # Uniform distribution = high entropy
    logits = torch.ones((1, 10))
    entropy = compute_entropy(logits)
    assert entropy.item() > 3.0 # log2(10) is ~3.32
    
    # Sharp distribution = low entropy
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    entropy = compute_entropy(logits)
    assert entropy.item() < 0.1

def test_data_miner():
    miner = DataMiner()
    # Mock model output
    output = {
        "scores": torch.tensor([
            [0.5, 0.5], # High entropy
            [0.9, 0.1], # Low entropy
        ]),
        "predicted_quality": torch.tensor([0.2, 0.8]) # Low quality vs high quality
    }
    
    hard_indices = miner.find_hard_tiles(output, num_tiles=1)
    assert hard_indices[0] == 0 # Index 0 should be harder (high entropy, low quality)

def test_pseudo_labeling():
    labeler = PseudoLabeler(conf_threshold=0.8, quality_threshold=0.7)
    output = {
        "scores": torch.tensor([
            [0.9, 0.1], # High confidence
            [0.4, 0.6], # Low confidence
        ]),
        "masks": torch.zeros((2, 1, 28, 28)),
        "boxes": torch.zeros((2, 4)),
        "predicted_quality": torch.tensor([0.9, 0.2]) # High quality vs low quality
    }
    
    silver = labeler.generate_silver_labels(output)
    assert len(silver["boxes"]) == 1
    assert silver["scores"][0] == 0.9

if __name__ == "__main__":
    pytest.main([__file__])
