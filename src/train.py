import os

import torch
import lightning as L
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
    get_pretrain_lr,
    get_pretraining_dataset
)

def main():
    num_gpus = torch.cuda.device_count()
    data_dir = os.path.join("..", "data")
    arg_dir = os.path.join("configs", "pre-train.yaml")
    args = get_args(arg_dir)
    seed_everything(args["seed"], workers=True)

    logger = TensorBoardLogger("pre-train-runs", name=args["backbone"], version=args["version"])
    log_dir = os.path.join("pre-train-runs", args["backbone"], f"version_{logger.version}")
    os.makedirs(log_dir, exist_ok=True)
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

    pretrain_dataset = get_pretraining_dataset(data_dir, args["dataset"])
    process_batch_size = args["batch_size"] if num_gpus == 0 else args["batch_size"] // num_gpus

    pretrain_loader = DataLoader(
        pretrain_dataset, 
        shuffle=True,
        persistent_workers=True,
        num_workers=os.cpu_count(), 
        batch_size=process_batch_size, 
        )
    
    encoder = Encoder(args["backbone"], args["hidden_dim"], args["projection_dim"])
    strategy = "ddp" if num_gpus > 1 else "auto"
    precision = "16-mixed" if torch.cuda.is_available() else "32"

    lr = get_pretrain_lr(args)
    model = SimCLR(encoder, lr, args["weight_decay"], args["eta_min"], args["temperature"])

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