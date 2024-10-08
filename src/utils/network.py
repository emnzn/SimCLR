import torch
import numpy as np
import torch.nn as nn
import lightning as L
from timm.optim import Lars
from sklearn.metrics import accuracy_score
from torchvision.models.resnet import ResNet
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

    learning_rate: float
        The learning rate of the optimizer.

    weight_decay: float
        The L2 regularization strength.

    eta_min: float
        The minimum value the learning rate decays to using CosineAnnealing.

    temperature: float
        The temperature parameter in NTXent.
    """

    def __init__(
        self, 
        encoder: Encoder,
        learning_rate: float,
        weight_decay: float,
        eta_min: float,
        temperature: float = 0.5
        ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eta_min = eta_min

        self.encoder = encoder
        self.criterion = NTXent(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)

        return z
    
    def training_step(self, batch, _):
        (x_i, x_j), _ = batch
        z_i, z_j = self(x_i), self(x_j)

        z_i = self.all_gather(z_i, sync_grads=True)
        z_j = self.all_gather(z_j, sync_grads=True)

        z_i = z_i.contiguous().reshape(-1, z_i.shape[-1])
        z_j = z_j.contiguous().reshape(-1, z_j.shape[-1])

        loss = self.criterion(z_i, z_j)
        self.log("Loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]["lr"]
        self.log("Learning Rate", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = Lars(self.encoder.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, eta_min=self.eta_min)

        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

        return config
    

class ResNetClassifier(L.LightningModule):

    """
    A constructor that takes in a pre-trained encoder, 
    and attaches a linear classification head for finetuning.

    This constructor assumes that the 2-layer non-linear 
    projection head has been removed.

    Parameters
    ----------
    encoder: Encoder
        The initialized ResNet Encoder with pre-trained weights.

    num_classes: int
        The number of output classes.

    embedding_dim: int
        The dimension of the image embedding produced by the encoder.

    freeze_encoder: bool
        Whether to freeze the encoder for linear evaluation of the representations.

    learning_rate: float
        The learning rate of the optimizer.

    weight_decay: float
        The L2 regularization strength.

    eta_min: float
        The minimum value the learning rate decays to using CosineAnnealing.
    """

    def __init__(
        self, 
        encoder: Encoder,
        num_classes: int,
        embedding_dim: int, 
        freeze_encoder: bool,
        learning_rate: float,
        weight_decay: float,
        eta_min: float
        ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eta_min = eta_min

        self.min_val_loss = np.inf
        self.max_val_accuracy = -np.inf
        self.criterion = nn.CrossEntropyLoss()

        self.encoder = encoder
        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(
        self, 
        x: torch.Tensor
        ) -> torch.Tensor:

        h = self.encoder(x)
        h = torch.flatten(h, 1)
        logits = self.fc(h)

        return logits
    
    def _pred_and_eval(self, batch):
        img, target = batch
        logits = self(img)
        confidence = nn.functional.softmax(logits, dim=1)
        pred = torch.argmax(confidence, dim=1)

        loss = self.criterion(logits, target)
        accuracy = accuracy_score(target.cpu(), pred.cpu())

        return loss, accuracy
    
    def training_step(self, batch, _):
        train_loss, train_accuracy = self._pred_and_eval(batch)
        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]["lr"]

        self.log("Train/Loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Train/Accuracy", train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Learning Rate", lr, on_step=False, on_epoch=True, prog_bar=True)

        metrics = {
            "loss": train_loss,
            "accuracy": train_accuracy
        }

        return metrics
    
    def validation_step(self, batch, _):
        val_loss, val_accuracy = self._pred_and_eval(batch)

        self.log("Validation/Loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("Validation/Accuracy", val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        metrics = {
            "loss": val_loss,
            "accuracy": val_accuracy
        }

        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.fc.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, eta_min=self.eta_min)

        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
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
    model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=3, bias=False)
    model.maxpool = nn.Identity()
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