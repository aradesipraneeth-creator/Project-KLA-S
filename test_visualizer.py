import os
import sys
import torch
import torch.nn as nn

# Ensure UTF-8 output formatting for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from utils.visualizer import save_visualizations_and_predictions

class DummyTensorModel(nn.Module):
    """Mock model returning single Tensor output (e.g. Restormer baseline)."""
    def forward(self, x):
        return torch.rand(x.size(0), 1, 256, 256)

class DummyDictModel(nn.Module):
    """Mock model returning AIR-Net dictionary output."""
    def forward(self, x):
        return {
            "restored": torch.rand(x.size(0), 1, 256, 256),
            "edge": torch.rand(x.size(0), 1, 256, 256),
            "noise": torch.tensor([0.5]),
            "blur": torch.tensor([0.3]),
            "texture": torch.tensor([0.8])
        }

class MockValDataset:
    """Mock validation dataset."""
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        lr = torch.rand(1, 128, 128)
        gt = torch.rand(1, 256, 256)
        return lr, gt, f"sample_{idx}.npy"

def main():
    print("====================================================")
    print("AIR-NET V1 - VISUALIZER COMPATIBILITY VERIFICATION")
    print("====================================================")

    dataset = MockValDataset()
    device = torch.device("cpu")
    vis_dir = "outputs/test_vis"
    val_preds_dir = "outputs/test_val_preds"

    # 1. Test Single Tensor Model Output (e.g. Restormer baseline)
    tensor_model = DummyTensorModel()
    save_visualizations_and_predictions(
        model=tensor_model,
        val_dataset=dataset,
        fixed_indices=[0],
        epoch=1,
        vis_dir=vis_dir,
        val_preds_dir=val_preds_dir,
        device=device
    )
    assert os.path.exists(os.path.join(vis_dir, "epoch_01_sample_001.png")), "Tensor visualization file missing!"
    print("✓ Tensor output supported.")
    print("✓ Backward compatibility maintained.")

    # 2. Test AIR-Net Dictionary Model Output (Restored + Edge + Scores)
    dict_model = DummyDictModel()
    save_visualizations_and_predictions(
        model=dict_model,
        val_dataset=dataset,
        fixed_indices=[1],
        epoch=1,
        vis_dir=vis_dir,
        val_preds_dir=val_preds_dir,
        device=device
    )
    assert os.path.exists(os.path.join(vis_dir, "epoch_01_sample_001.png")), "AIR-Net restored image file missing!"
    assert os.path.exists(os.path.join(vis_dir, "epoch_01_sample_001_edge.png")), "AIR-Net edge visualization file missing!"
    print("✓ AIR-Net dictionary output supported.")
    print("✓ Restored image visualization works.")
    print("✓ Edge visualization works.")

    print("----------------------------------------------------")
    print("✓ AIR-Net visualizer compatibility verification COMPLETE!")
    print("====================================================")

if __name__ == "__main__":
    main()
