from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from utils import (
    BASE_DIR,
    DEFAULT_DATA_DIR,
    plot_vae_training_metrics as plot_metrics,
    select_device,
    set_seed,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "vae_mnist"
DATASET_CHOICES = ("mnist", "fashion-mnist")
FASHION_MNIST_MIRROR = (
    "https://raw.githubusercontent.com/zalandoresearch/"
    "fashion-mnist/master/data/fashion/"
)


def dataset_slug(dataset_name: str) -> str:
    return dataset_name.replace("-", "_")


def get_dataset_class(dataset_name: str) -> type[datasets.MNIST]:
    if dataset_name == "mnist":
        return datasets.MNIST
    if dataset_name == "fashion-mnist":
        if FASHION_MNIST_MIRROR not in datasets.FashionMNIST.mirrors:
            datasets.FashionMNIST.mirrors = [
                FASHION_MNIST_MIRROR,
                *datasets.FashionMNIST.mirrors,
            ]
        return datasets.FashionMNIST
    raise ValueError(f"Unsupported dataset: {dataset_name}")


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20, hidden_dim: int = 400) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        return x.view(-1, 1, 28, 28)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction + beta * kl_divergence, reconstruction, kl_divergence


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(args: argparse.Namespace, device: torch.device) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    data_dir = Path(args.data_dir)
    dataset_class = get_dataset_class(args.dataset)

    train_dataset = dataset_class(
        root=str(data_dir),
        train=True,
        transform=transform,
        download=not args.no_download,
    )
    test_dataset = dataset_class(
        root=str(data_dir),
        train=False,
        transform=transform,
        download=not args.no_download,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


def run_epoch(
    model: VAE,
    data_loader: DataLoader,
    device: torch.device,
    beta: float,
    optimizer: optim.Optimizer | None = None,
    log_interval: int = 100,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0
    total_samples = 0

    for batch_idx, (x, _) in enumerate(data_loader, start=1):
        x = x.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            recon_x, mu, logvar = model(x)
            loss, reconstruction, kl_divergence = vae_loss(
                recon_x, x, mu, logvar, beta=beta
            )

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = x.size(0)
        total_samples += batch_size
        total_loss += loss.item()
        total_reconstruction += reconstruction.item()
        total_kl += kl_divergence.item()

        if is_train and log_interval > 0 and batch_idx % log_interval == 0:
            print(
                f"batch={batch_idx:04d} "
                f"loss={loss.item() / batch_size:.4f} "
                f"recon={reconstruction.item() / batch_size:.4f} "
                f"kl={kl_divergence.item() / batch_size:.4f}"
            )

    return (
        total_loss / total_samples,
        total_reconstruction / total_samples,
        total_kl / total_samples,
    )


@torch.no_grad()
def save_artifacts(
    model: VAE,
    fixed_batch: torch.Tensor,
    output_dir: Path,
    device: torch.device,
    epoch: int,
    latent_dim: int,
    reconstruction_count: int,
    sample_count: int,
) -> None:
    model.eval()

    x = fixed_batch[:reconstruction_count].to(device)
    recon_x, _, _ = model(x)
    comparison = torch.cat([x.cpu(), recon_x.cpu()])
    save_image(
        comparison,
        output_dir / f"reconstruction_epoch_{epoch:03d}.png",
        nrow=reconstruction_count,
    )

    z = torch.randn(sample_count, latent_dim, device=device)
    samples = model.decode(z).cpu()
    save_image(samples, output_dir / f"samples_epoch_{epoch:03d}.png", nrow=8)


def save_checkpoint(
    model: VAE,
    optimizer: optim.Optimizer,
    output_dir: Path,
    epoch: int,
    args: argparse.Namespace,
    filename: str,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        output_dir / filename,
    )




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VAE on MNIST or Fashion-MNIST.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="mnist")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/vae_<dataset>.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--reconstruction-count", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use an existing dataset in data-dir instead of downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = BASE_DIR / "outputs" / f"vae_{dataset_slug(args.dataset)}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    checkpoint_prefix = f"vae_{dataset_slug(args.dataset)}"
    print(f"Dataset: {args.dataset}")
    print(f"Using device: {device}")

    train_loader, test_loader = build_loaders(args, device)
    fixed_batch, _ = next(iter(test_loader))

    model = VAE(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    metrics_path = output_dir / "metrics.csv"
    best_test_loss = float("inf")

    with metrics_path.open("w", newline="") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_reconstruction",
                "train_kl",
                "test_loss",
                "test_reconstruction",
                "test_kl",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_recon, train_kl = run_epoch(
                model,
                train_loader,
                device,
                beta=args.beta,
                optimizer=optimizer,
                log_interval=args.log_interval,
            )
            test_loss, test_recon, test_kl = run_epoch(
                model,
                test_loader,
                device,
                beta=args.beta,
            )

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_reconstruction": train_recon,
                    "train_kl": train_kl,
                    "test_loss": test_loss,
                    "test_reconstruction": test_recon,
                    "test_kl": test_kl,
                }
            )
            metrics_file.flush()

            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_recon={test_recon:.4f} "
                f"test_kl={test_kl:.4f}"
            )

            save_artifacts(
                model,
                fixed_batch,
                output_dir,
                device,
                epoch,
                latent_dim=args.latent_dim,
                reconstruction_count=args.reconstruction_count,
                sample_count=args.sample_count,
            )
            save_checkpoint(
                model,
                optimizer,
                output_dir,
                epoch,
                args,
                f"{checkpoint_prefix}_last.pt",
            )

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save_checkpoint(
                    model,
                    optimizer,
                    output_dir,
                    epoch,
                    args,
                    f"{checkpoint_prefix}_best.pt",
                )

    metrics_plot_path = plot_metrics(metrics_path, output_dir / "metrics.png")
    print(f"Saved metrics plot to: {metrics_plot_path}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
