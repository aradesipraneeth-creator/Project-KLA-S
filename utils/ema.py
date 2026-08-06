import copy
import torch
import torch.nn as nn

class ModelEMA:
    """
    Exponential Moving Average (EMA) of model weights.
    Maintains a shadow copy of parameters updated via:
        shadow_param = decay * shadow_param + (1 - decay) * model_param
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model: nn.Module):
        """Updates the EMA parameters with current model parameters."""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)
