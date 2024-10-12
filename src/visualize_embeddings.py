import os
from typing import Tuple, List

import torch
import numpy as np
from tqdm import tqdm
import plotly.express as px
from sklearn.manifold import TSNE
from lightning import seed_everything
from torch.utils.data import DataLoader

from utils import (
    Encoder,
    ResNetClassifier,
    get_args,
    get_encoder_args,
    get_finetune_dataset
)


@torch.no_grad()
def get_embeddings(
    data_loader: DataLoader,
    encoder: Encoder,
    device: str
    ) -> tuple[List[np.ndarray], List[str]]:

    encoder.eval()
    embeddings = []
    labels = []

    for img, label in tqdm(data_loader, desc="Encoding in progress"):
        img = img.to(device)
        label = label.cpu().numpy()

        embedding = encoder(img).squeeze().cpu().numpy()
        embeddings.extend(embedding)
        labels.extend(label)

    return embeddings, labels


def plot_embeddings(
    embeddings: List[np.ndarray],
    labels: List[str],
    train_type: str,
    save_dir: str,
    seed: int,
    ):

    label_map = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }

    embeddings = np.array(embeddings)
    labels = [label_map[i] for i in labels]

    tsne = TSNE(n_components=2, random_state=seed)
    projections = tsne.fit_transform(embeddings)

    fig = px.scatter(
        projections, x=0, y=1,
        color=labels, labels={"color": "label"}
    )
    fig.write_image(os.path.join(save_dir, f"{train_type}-embedding.png"))


def main():
    data_dir = os.path.join("..", "data")
    arg_dir = os.path.join("configs", "visualize-embedding.yaml")
    args = get_args(arg_dir)
    seed_everything(args["seed"], workers=True)

    device = "mps"
    train_type = "pre-trained" if args["freeze_encoder"] else "from-scratch"

    save_dir = os.path.join("..", "assets", "embedding-visualization", args["backbone"], "finetune", f"version_{args['experiment_version']}")
    os.makedirs(save_dir, exist_ok=True)

    _, val_dataset = get_finetune_dataset(data_dir, dataset=args["dataset"])

    data_loader = DataLoader(
        val_dataset, 
        batch_size=args["batch_size"], 
        shuffle=False, 
        num_workers=os.cpu_count() // 4, 
        persistent_workers=True
    )

    run_dir = os.path.join("pre-train-runs", args["backbone"], f"version_{args['experiment_version']}", "run-config.yaml")
    ckpt_dir = os.path.join("..", "assets", "model-weights", args["backbone"], "finetune", f"version_{args['experiment_version']}", train_type, "highest-accuracy.ckpt")
    
    ckpt = torch.load(ckpt_dir, map_location=torch.device(device))
    weights = ckpt["state_dict"]

    encoder_args = get_encoder_args(run_dir)
    encoder = Encoder(**encoder_args).to(device)

    embedding_dim = encoder.g[0].in_features
    encoder = encoder.f
    
    classifier = ResNetClassifier(
        encoder=encoder, 
        num_classes=args["num_classes"], 
        embedding_dim=embedding_dim, 
        freeze_encoder=args["freeze_encoder"], 
        learning_rate=None, 
        weight_decay=None, 
        eta_min=None
        )

    classifier.load_state_dict(weights)
    encoder = classifier.encoder
    embeddings, labels = get_embeddings(data_loader, encoder, device)

    plot_embeddings(embeddings, labels, train_type, save_dir, args["seed"])


if __name__ == "__main__":
    main()