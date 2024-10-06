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
    Encoder,
    ResNetClassifier,
    get_args,
    save_args,
    get_encoder_args,
    get_finetune_dataset
)

def main():
    data_dir = os.path.join("..", "data")
    arg_dir = os.path.join("configs", "finetune.yaml")
    args = get_args(arg_dir)
    seed_everything(args["seed"], workers=True)

    logger = TensorBoardLogger("finetune-runs", name=args["backbone"], version=args["experiment_version"])
    log_dir = os.path.join("finetune-runs", args["backbone"], f"version_{args["experiment_version"]}")
    os.makedirs(log_dir, exist_ok=True)
    save_args(args, log_dir)

    save_dir = os.path.join("..", "assets", "model-weights", args["backbone"], "finetune", f"version_{args["experiment_version"]}")
    os.makedirs(save_dir, exist_ok=True)

    pbar = TQDMProgressBar(leave=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="highest-accuracy",
        monitor="Validation/Accuracy",
        mode="max",
        save_weights_only=False,
        save_on_train_epoch_end=True,
        enable_version_counter=False
    )

    train_dataset, val_dataset = get_finetune_dataset(data_dir, dataset=args["dataset"])
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=os.cpu_count(), persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=os.cpu_count(), persistent_workers=True)

    run_dir = os.path.join("pre-train-runs", args["backbone"], f"version_{args['experiment_version']}", "run-config.yaml")
    ckpt_dir = os.path.join("..", "assets", "model-weights", args["backbone"], "pre-train", f"version_{args['experiment_version']}", "min-loss.ckpt")
    
    ckpt = torch.load(ckpt_dir, map_location=torch.device("cpu"))
    encoder_weights = {k.replace("encoder.", ""): v for k, v in ckpt["state_dict"].items()}

    encoder_args = get_encoder_args(run_dir)
    encoder = Encoder(**encoder_args)
    if not args["random_weight_init"]: encoder.load_state_dict(encoder_weights)

    embedding_dim = encoder.g[0].in_features
    encoder = encoder.f

    strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
    precision = "16-mixed" if torch.cuda.is_available() else "32"

    classifier = ResNetClassifier(
        encoder=encoder, 
        num_classes=10, 
        embedding_dim=embedding_dim, 
        freeze_encoder=args["freeze_encoder"], 
        learning_rate=args["learning_rate"], 
        weight_decay=args["weight_decay"], 
        eta_min=args["eta_min"]
        )

    trainer = L.Trainer(
        logger=logger,
        devices=-1,
        strategy=strategy,
        accelerator="auto", 
        deterministic=True,
        precision=precision,
        num_sanity_val_steps=0,
        max_epochs=args["epochs"], 
        callbacks=[checkpoint_callback, pbar]
    )

    trainer.fit(model=classifier, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()