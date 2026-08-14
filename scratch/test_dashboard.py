import os
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import load_airnet_v3_model, validate_image_shape, compute_sobel_edge_map, explain_category_routing

def test():
    print("--- Testing Dashboard Core Components ---")
    info = load_airnet_v3_model()
    m = info['model']
    print(f"[OK] Model Loaded! Parameters: {info['num_params']:,} | Device: {info['device']}")

    sample_path = os.path.join(PROJECT_ROOT, "train", "train", "NoisyLR", "000000.npy")
    arr = np.load(sample_path).astype(np.float32)
    inp = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(info['device'])

    with torch.no_grad():
        out = m(inp)
        res = out['restored'].squeeze().cpu().numpy()

    print(f"[OK] Restored Output Shape: {res.shape} | Min: {res.min():.6f} | Max: {res.max():.6f} | Mean: {res.mean():.6f}")

    em_raw, em_vis, em_max = compute_sobel_edge_map(res)
    print(f"[OK] Sobel Edge Map computed! Max: {em_max:.6f} | Vis Shape: {em_vis.shape}")

    exp = explain_category_routing({'sobel_edge_index': 0.12, 'gradient_energy': 0.05, 'edge_density': 0.25}, 'EDGE_DOMINANT', {'EDGE_DOMINANT': 0.85})
    print(f"[OK] Routing Explanation Generated: '{exp[:70]}...'")

    print("==============================================================================")
    print("ALL DASHBOARD COMPONENT TESTS PASSED CLEANLY!")
    print("==============================================================================")

if __name__ == "__main__":
    test()
