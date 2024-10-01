import os

import torch
import lightning as L
from timm.optim import Lars
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from utils import (
    SimCLR,
    Encoder,
    get_args,
    get_pretraining_dataset
)

def main():
    data_dir = os.path.join("..", "data")
    arg_dir = os.path.join("configs", "pre-train-config.yaml")
    args = get_args(arg_dir)

    save_dir = os.path.join("..", "assets", args["backbone"])
    os.makedirs(save_dir, exist_ok=True)

    logger = TensorBoardLogger("runs", name=args["backbone"])
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="min_loss",
        monitor="loss",
        mode="min",
        save_weights_only=False,
        save_on_train_epoch_end=True
    )
    
    pretrain_dataset = get_pretraining_dataset(data_dir, args["dataset"])
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=9, persistent_workers=True)
    
    encoder = Encoder(args["backbone"], args["hidden_dim"], args["projection_dim"])
    optimizer = Lars(encoder.parameters(), lr=args["learning_rate"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args["epochs"], eta_min=args["eta_min"])

    model = SimCLR(encoder, optimizer, lr_scheduler, args["temperature"])

    trainer = L.Trainer(
        logger=logger,
        devices='auto',
        accelerator='auto', 
        max_epochs=args["epochs"], 
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model=model, train_dataloaders=pretrain_loader)

if __name__ == "__main__":
    main()