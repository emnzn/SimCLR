import os

import torch
import lightning as L
from timm.optim import Lars
from lightning import seed_everything
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    TQDMProgressBar
    )

from utils import (
    SimCLR,
    Encoder,
    get_args,
    save_args,
    get_pretraining_dataset
)

def main():
    data_dir = os.path.join("..", "data")
    arg_dir = os.path.join("configs", "pre-train.yaml")
    args = get_args(arg_dir)
    seed_everything(args["seed"], workers=True)

    logger = TensorBoardLogger("pre-train-runs", name=args["backbone"], version=args["version"])
    log_dir = os.path.join("pre-train-runs", args["backbone"], f"version_{logger.version}")
    os.makedirs(log_dir)
    save_args(args, log_dir)

    save_dir = os.path.join("..", "assets", "model-weights", args["backbone"], "pre-train", f"version_{logger.version}")
    os.makedirs(save_dir, exist_ok=True)
    
    pbar = TQDMProgressBar(leave=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="min-loss",
        monitor="Loss",
        mode="min",
        save_weights_only=False,
        save_on_train_epoch_end=True,
        enable_version_counter=False
    )

    if args["lr_scaling_method"] == "square-root":
        lr = 0.075 * (args["batch_size"] ** 0.5)

    elif args["lr_scaling_method"] == "linear":
        lr = 0.3 * (args["batch_size"] / 256)
    
    pretrain_dataset = get_pretraining_dataset(data_dir, args["dataset"])
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=os.cpu_count(), persistent_workers=True)
    
    encoder = Encoder(args["backbone"], args["hidden_dim"], args["projection_dim"])
    strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
    precision = "16-mixed" if torch.cuda.is_available() else "32"

    optimizer = Lars(encoder.parameters(), lr=lr, weight_decay=args["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args["epochs"], eta_min=args["eta_min"])

    model = SimCLR(encoder, optimizer, lr_scheduler, args["temperature"])

    trainer = L.Trainer(
        logger=logger,
        devices=-1,
        strategy=strategy,
        accelerator="auto", 
        deterministic=True,
        precision=precision,
        max_epochs=args["epochs"], 
        callbacks=[checkpoint_callback, pbar]
    )
    
    trainer.fit(model=model, train_dataloaders=pretrain_loader)

if __name__ == "__main__":
    main()