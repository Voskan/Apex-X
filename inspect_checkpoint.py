
import torch
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from apex_x.train.checkpoint import safe_torch_load, extract_model_state_dict

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'artifacts/train_output/checkpoints/best.pt'

if not os.path.exists(MODEL_PATH):
    print(f"❌ Checkpoint not found: {MODEL_PATH}")
    sys.exit(1)

try:
    payload = safe_torch_load(MODEL_PATH, map_location='cpu')
    state_dict, _ = extract_model_state_dict(payload)
    
    print(f"✅ Loaded checkpoint: {MODEL_PATH}")
    print(f"Total keys: {len(state_dict)}")
    
    print("\n--- Key Groups ---")
    groups = set()
    for key in state_dict.keys():
        parts = key.split('.')
        if len(parts) > 1:
            groups.add(parts[0])
        else:
            groups.add(key)
            
    for g in sorted(groups):
        print(f"Prefix: {g}")
        # Print first few keys of this group
        sample = [k for k in state_dict.keys() if k.startswith(g)][:3]
        for s in sample:
            print(f"  - {s}")

    print("\n--- Searching for Proposal/Anchor related keys ---")
    interesting = ['anchor', 'proposal', 'rpn', 'query', 'embed', 'seed']
    found = []
    for key in state_dict.keys():
        if any(x in key for x in interesting):
            found.append(key)
    
    for f in sorted(found)[:20]:
        print(f"Found: {f}")
        
except Exception as e:
    print(f"❌ Error: {e}")
