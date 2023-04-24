import torch
import pytorch_lightning as pl

class weighted_mse(pl.LightningModule):
    def __init__(self, weights=torch.ones((216))):
        super(weighted_mse, self).__init__()
        self.register_buffer("weights", weights)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
            weights # 216
            pred # B, 216
            target # B, 216
        """
        squared_error = (input - target) ** 2 # B, 216
        weighted_squared_error = self.weights * squared_error  # B, 216
        mwse = torch.mean(weighted_squared_error)
        return mwse
