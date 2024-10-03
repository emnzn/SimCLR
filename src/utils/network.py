import torch
import numpy as np
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torchvision.models.resnet import ResNet
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.models import (
    resnet18, 
    resnet34,
    resnet50, 
    resnet101, 
    resnet152
)
from .loss import NTXent


class Encoder(nn.Module):

    """
    The ResNet encoder with a two layer MLP projection head.

    Parameters
    ----------
    backbone: str
        Must be one of [`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`].

    hidden_dim: int
        The hidden dimension of the projection head.

    projection_dim: int
        The dimension to project the representations into.

    Returns
    -------
    z: torch.Tensor
        The projected representation of the image.
    """

    def __init__(
        self,
        backbone: str, 
        hidden_dim: int = 2048, 
        projection_dim: int = 128
        ):

        super().__init__()

        valid_models = [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152"
        ]

        assert backbone in valid_models, f"backbone must be one of {valid_models}"

        self.f, h_dim = get_model(backbone)
        self.g = G(
            nn.Linear(h_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.f(x)
        h = torch.flatten(h, 1)
        z = self.g(h)

        return z


class SimCLR(L.LightningModule):

    """
    A lightning implementation of SimCLR: 
    A Simple Framework for Contrastive Learning of Visual Representations.

    Parameters
    ----------
    encoder: Encoder
        The initialized ResNet Encoder.

    optimizer: torch.optim.Optimizer
        The initialized optimizer.

    lr_scheduler: _LRScheduler
        The initialized learning rate scheduler.

    temperature: float
        The temperature parameter in NTXent.
    """

    def __init__(
        self, 
        encoder: Encoder,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: _LRScheduler,
        temperature: float = 0.5
        ):
        super().__init__()

        self.training_step_losses = []
        self.min_loss = np.inf

        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer

        self.encoder = encoder
        self.criterion = NTXent(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)

        return z
    
    def training_step(self, batch, _):
        (x_i, x_j), _ = batch
        z_i, z_j = self(x_i), self(x_j)
        loss = self.criterion(z_i, z_j)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.training_step_losses.append(loss)

        return loss
    
    def on_train_epoch_end(self):
        mean_loss = torch.stack(self.training_step_losses).mean()
        print(f"\nLoss: {mean_loss}")

        if mean_loss < self.min_loss:
            print("\nNew minimum loss — model saved.")
            self.min_loss = mean_loss

        print("\n--------------------------------\n")
        self.training_step_losses.clear()

    def configure_optimizers(self):
        config = {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

        return config
    

def get_model(model: str) -> ResNet:

    """
    Return a model for the encoder.

    Parameters
    ----------
    model: str
        Must be one of [`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`].

    Returns
    -------
    backbone: ResNet
        The resnet backbone with the classification head removed.

    h_dim: int
        The final output dimension to be used to determine the input size to the projection head.
    """

    model_table = {
        "resnet18": resnet18(),
        "resnet34": resnet34(),
        "resnet50": resnet50(),
        "resnet101": resnet101(),
        "resnet152": resnet152()
    }

    model = model_table[model]
    encoder = list(model.children())[:-1]
    h_dim = list(model.children())[-1].in_features

    backbone = F(*encoder)

    return backbone, h_dim


class F(nn.Sequential):
    """
    Just used for better annotations in the model graph visualization.
    """
    def __init__(self, *args):
        super().__init__(*args)


class G(nn.Sequential):
    """
    Just used for better annotations in the model graph visualization.
    """
    def __init__(self, *args):
        super().__init__(*args)