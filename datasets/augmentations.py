import random
import torch

class PairedTransform:
    """
    Applies synchronized spatial transformations to paired LR and GT images.
    Preserves original float values without clipping.
    Allowed transforms:
      - Random Horizontal Flip
      - Random Vertical Flip
      - Random Rotation (90, 180, 270 degrees)
    Disallowed:
      - Color jitter, brightness, contrast, gaussian blur.
    """
    def __init__(self, is_train: bool = True):
        self.is_train = is_train

    def __call__(self, lr: torch.Tensor, gt: torch.Tensor):
        """
        Args:
            lr: Tensor of shape (C, H_lr, W_lr) e.g., (1, 128, 128)
            gt: Tensor of shape (C, H_gt, W_gt) e.g., (1, 256, 256)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Augmented (lr, gt)
        """
        if not self.is_train:
            return lr, gt

        # Random Horizontal Flip
        if random.random() > 0.5:
            lr = torch.flip(lr, dims=[-1])
            gt = torch.flip(gt, dims=[-1])

        # Random Vertical Flip
        if random.random() > 0.5:
            lr = torch.flip(lr, dims=[-2])
            gt = torch.flip(gt, dims=[-2])

        # Random 90/180/270 Rotation
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            lr = torch.rot90(lr, k=k, dims=[-2, -1])
            gt = torch.rot90(gt, k=k, dims=[-2, -1])

        return lr, gt
