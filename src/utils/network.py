import torch
import numpy as np
import torch.nn as nn
import lightning as L
from sklearn.metrics import accuracy_score
from torchvision.models.resnet import ResNet
from torch.optim.lr_scheduler import _LRScheduler
from lightning.pytorch.utilities import rank_zero
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
        self.save_hyperparameters(ignore=["encoder"])

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
        self.log("Loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]["lr"]
        self.log("Learning Rate", lr, on_step=False, on_epoch=True, prog_bar=True)

        return loss

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

    optimizer: torch.optim.Optimizer
        The initialized optimizer.

    lr_scheduler: _LRScheduler
        The initialized learning rate scheduler.
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

        self.train_running_metric = {
            "accumulated_loss": 0,
            "accumulated_accuracy": 0,
            "num_steps": 0
        }

        self.val_running_metric = {
            "accumulated_loss": 0,
            "accumulated_accuracy": 0,
            "num_steps": 0
        }

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

        self.train_running_metric["accumulated_loss"] += train_loss.detach().item()
        self.train_running_metric["accumulated_accuracy"] += train_accuracy
        self.train_running_metric["num_steps"] += 1

        optimizer = self.trainer.optimizers[0]
        lr = optimizer.param_groups[0]["lr"]
        self.log("Learning Rate", lr, on_step=False, on_epoch=True, prog_bar=True)

        metrics = {
            "loss": train_loss,
            "accuracy": train_accuracy
        }

        return metrics
    
    def validation_step(self, batch, _):
        val_loss, val_accuracy = self._pred_and_eval(batch)

        self.val_running_metric["accumulated_loss"] += val_loss.item()
        self.val_running_metric["accumulated_accuracy"] += val_accuracy
        self.val_running_metric["num_steps"] += 1

        metrics = {
            "loss": val_loss,
            "accuracy": val_accuracy
        }

        return metrics
    
    def on_train_epoch_end(self):
        mean_train_loss = self.train_running_metric["accumulated_loss"] / self.train_running_metric["num_steps"]
        mean_train_accuracy = self.train_running_metric["accumulated_accuracy"] / self.train_running_metric["num_steps"]

        self.log("Train/Loss", mean_train_loss, on_epoch=True, prog_bar=False)
        self.log("Train/Accuracy", mean_train_accuracy, on_epoch=True, prog_bar=False)

        mean_val_loss = self.val_running_metric["accumulated_loss"] / self.val_running_metric["num_steps"]
        mean_val_accuracy = self.val_running_metric["accumulated_accuracy"] / self.val_running_metric["num_steps"]

        self.log("Validation/Loss", mean_val_loss, on_epoch=True, prog_bar=False)
        self.log("Validation/Accuracy", mean_val_accuracy, on_epoch=True, prog_bar=False)

        print("\nValidation Statistics:")
        print(f"Loss: {mean_val_loss:.4f} | Accuracy: {mean_val_accuracy:.4f}\n")

        if mean_val_loss < self.min_val_loss:
            print("New minimum loss — model saved.")
            self.min_val_loss = mean_val_loss

        if mean_val_accuracy > self.max_val_accuracy:
            print("New maximum accuracy — model saved.")
            self.max_val_accuracy = mean_val_accuracy

        print("\n-------------------------------------------------------------------\n")
        
        self.val_running_metric = {k: 0 for k in self.val_running_metric.keys()}
        self.train_running_metric = {k: 0 for k in self.train_running_metric.keys()}

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