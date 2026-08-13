import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageIndexer:
    """
    Image Characteristic Indexer for AIR-Net v3:
    Computes 10 normalized feature metrics strictly from the 128x128 INPUT image:
      1. Sobel Edge Index
      2. Gradient Energy
      3. Laplacian Energy
      4. High-Frequency Energy
      5. Texture Index
      6. Noise Index
      7. Contrast Index
      8. Entropy
      9. Edge Density
      10. Sparse Feature Index
    """
    def __init__(self, norm_params: dict = None):
        self.norm_params = norm_params

    def _to_tensor(self, img) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            t = torch.from_numpy(img.astype(np.float32))
        elif isinstance(img, torch.Tensor):
            t = img.float()
        else:
            raise TypeError("Expected numpy array or torch.Tensor")
        
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            t = t.unsqueeze(0)
        return t

    def compute_indices(self, img) -> dict:
        t = self._to_tensor(img)
        device = t.device
        b, c, h, w = t.shape
        t_float = t.float()

        # 1. Sobel Gradients
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        gx = F.conv2d(t_float, sobel_x, padding=1)
        gy = F.conv2d(t_float, sobel_y, padding=1)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)

        sobel_edge_index = float(torch.mean(grad_mag).item())
        gradient_energy = float(torch.mean(gx**2 + gy**2).item())

        # 2. Laplacian Energy
        lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        lap_map = F.conv2d(t_float, lap_kernel, padding=1)
        laplacian_energy = float(torch.mean(lap_map**2).item())

        # 3. High-Frequency Energy (x - GaussianBlur(x))
        coords = torch.arange(5, dtype=torch.float32, device=device) - 2.0
        g1d = torch.exp(-coords**2 / (2 * 1.0**2))
        g1d = g1d / g1d.sum()
        g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
        blur_kernel = g2d.view(1, 1, 5, 5)
        blurred = F.conv2d(t_float, blur_kernel, padding=2)
        hf_map = t_float - blurred
        hf_energy = float(torch.mean(hf_map**2).item())

        # 4. Texture Index (Local variance of HF map + local gradient variance)
        hf_sq = hf_map**2
        local_hf_var = F.avg_pool2d(hf_sq, kernel_size=5, stride=1, padding=2)
        texture_index = float(torch.mean(local_hf_var).item())

        # 5. Noise Index (HF energy minus structured edge magnitude)
        # Ratio of high-frequency noise variance to edge energy
        noise_proxy = hf_energy / (sobel_edge_index + 1e-4)
        noise_index = float(noise_proxy)

        # 6. Contrast Index (Standard deviation & 95-5 percentile spread)
        img_np = t_float.squeeze().cpu().numpy()
        std_contrast = float(np.std(img_np))
        p95, p05 = float(np.percentile(img_np, 95)), float(np.percentile(img_np, 5))
        contrast_index = float(std_contrast * (p95 - p05))

        # 7. Entropy (Intensity histogram entropy)
        hist, _ = np.histogram(img_np, bins=32, range=(0.0, 1.0), density=True)
        hist = hist[hist > 0]
        entropy_val = float(-np.sum(hist * np.log2(hist + 1e-8)))

        # 8. Edge Density (Proportion of pixels with grad_mag > threshold)
        grad_np = grad_mag.squeeze().cpu().numpy()
        edge_density = float(np.mean(grad_np > 0.15))

        # 9. Sparse Feature Index (Ratio of max local intensity/gradient peak to background mean)
        max_grad_peak = float(np.max(grad_np))
        mean_grad_bg = float(np.mean(grad_np) + 1e-6)
        sparse_feature_index = float(max_grad_peak / mean_grad_bg)

        raw_indices = {
            "sobel_edge_index": sobel_edge_index,
            "gradient_energy": gradient_energy,
            "laplacian_energy": laplacian_energy,
            "hf_energy": hf_energy,
            "texture_index": texture_index,
            "noise_index": noise_index,
            "contrast_index": contrast_index,
            "entropy": entropy_val,
            "edge_density": edge_density,
            "sparse_feature_index": sparse_feature_index
        }
        return raw_indices

    def normalize_indices(self, raw_indices: dict) -> torch.Tensor:
        """
        Converts raw 10-index dict into a normalized 10-D PyTorch tensor.
        Uses training-set min/max or mean/std if norm_params provided.
        """
        keys = [
            "sobel_edge_index", "gradient_energy", "laplacian_energy", "hf_energy",
            "texture_index", "noise_index", "contrast_index", "entropy",
            "edge_density", "sparse_feature_index"
        ]
        vec = []
        for k in keys:
            val = raw_indices[k]
            if self.norm_params and k in self.norm_params:
                mean = self.norm_params[k]["mean"]
                std = self.norm_params[k]["std"] + 1e-8
                norm_val = (val - mean) / std
            else:
                norm_val = val
            vec.append(norm_val)
        return torch.tensor(vec, dtype=torch.float32)

def fit_training_normalization(all_raw_indices: list, save_json_path: str = None) -> dict:
    """
    Fits mean and std for all 10 indices using TRAINING DATA ONLY.
    """
    keys = list(all_raw_indices[0].keys())
    norm_params = {}
    for k in keys:
        vals = [d[k] for d in all_raw_indices]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        min_v = float(np.min(vals))
        max_v = float(np.max(vals))
        norm_params[k] = {
            "mean": mean_v,
            "std": std_v,
            "min": min_v,
            "max": max_v
        }

    if save_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_json_path)), exist_ok=True)
        with open(save_json_path, "w") as f:
            json.dump(norm_params, f, indent=4)
        print(f"[OK] Training-Set Index Normalization saved to: '{save_json_path}'")

    return norm_params
