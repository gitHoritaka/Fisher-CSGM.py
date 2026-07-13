from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from sklearn.linear_model import Lasso

from train_vae import VAE
from utils import (
    BASE_DIR,
    DEFAULT_DATA_DIR,
    MetricRow,
    make_random_measurement_matrix,
    measure_images,
    metric_rows,
    plot_cs_summary,
    select_device,
    set_seed,
    summarize_metrics,
    write_per_image_metrics,
    write_summary_metrics,
)


DEFAULT_CHECKPOINT = BASE_DIR / "outputs" / "vae_mnist" / "vae_mnist_best.pt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "cs_comparison"
DEFAULT_MAX_MEASUREMENTS = 200
DEFAULT_MEASUREMENT_STEP = 5


def load_vae(
    checkpoint_path: Path,
    device: torch.device,
    latent_dim: int | None,
    hidden_dim: int | None,
) -> tuple[VAE, int, int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the VAE first with train_vae.py."
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = checkpoint.get("args", {})
    resolved_latent_dim = latent_dim or int(checkpoint_args.get("latent_dim", 20))
    resolved_hidden_dim = hidden_dim or int(checkpoint_args.get("hidden_dim", 400))

    model = VAE(
        latent_dim=resolved_latent_dim,
        hidden_dim=resolved_hidden_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model, resolved_latent_dim, resolved_hidden_dim


def build_eval_loader(args: argparse.Namespace) -> DataLoader:
    dataset = datasets.MNIST(
        root=str(args.data_dir),
        train=False,
        transform=transforms.ToTensor(),
        download=not args.no_download,
    )
    if args.num_samples > len(dataset):
        raise ValueError(
            f"--num-samples={args.num_samples} exceeds test set size {len(dataset)}"
        )

    indices = list(range(args.num_samples))
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def lasso_reconstruct(
    matrix: torch.Tensor,
    measurements: torch.Tensor,
    alpha: float,
    max_iter: int,
    tol: float,
) -> torch.Tensor:
    design_matrix = matrix.detach().cpu().numpy()
    measurement_batch = measurements.detach().cpu().numpy()

    estimates = []
    for measurement in measurement_batch:
        lasso = Lasso(
            alpha=alpha,
            fit_intercept=False,
            max_iter=max_iter,
            tol=tol,
            selection="cyclic",
        )
        lasso.fit(design_matrix, measurement)
        estimates.append(lasso.coef_)

    estimate_tensor = torch.as_tensor(
        np.asarray(estimates),
        dtype=measurements.dtype,
        device=measurements.device,
    )
    return estimate_tensor.view(-1, 1, 28, 28).clamp(0.0, 1.0)


def vae_reconstruct(
    model: VAE,
    matrix: torch.Tensor,
    measurements: torch.Tensor,
    latent_dim: int,
    restarts: int,
    steps: int,
    lr: float,
    z_prior_weight: float,
    log_interval: int,
) -> torch.Tensor:
    batch_size = measurements.shape[0]
    device = measurements.device
    candidates = batch_size * restarts

    expanded_measurements = (
        measurements[:, None, :]
        .expand(batch_size, restarts, measurements.shape[1])
        .reshape(candidates, measurements.shape[1])
    )
    z = torch.randn(candidates, latent_dim, device=device, requires_grad=True)
    optimizer = optim.Adam([z], lr=lr)

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        generated = model.decode(z).flatten(start_dim=1)
        predicted = generated @ matrix.T
        measurement_loss = F.mse_loss(
            predicted,
            expanded_measurements,
            reduction="none",
        ).mean(dim=1)
        prior_loss = z.pow(2).sum(dim=1)
        loss = (measurement_loss + z_prior_weight * prior_loss).mean()
        loss.backward()
        optimizer.step()

        if log_interval > 0 and step % log_interval == 0:
            print(f"vae_step={step:04d} latent_objective={loss.item():.6f}")

    with torch.no_grad():
        generated = model.decode(z).flatten(start_dim=1)
        predicted = generated @ matrix.T
        measurement_loss = F.mse_loss(
            predicted,
            expanded_measurements,
            reduction="none",
        ).mean(dim=1)
        prior_loss = z.pow(2).sum(dim=1)
        total_loss = measurement_loss + z_prior_weight * prior_loss
        total_loss = total_loss.view(batch_size, restarts)
        best_restart = total_loss.argmin(dim=1)

        z = z.detach().view(batch_size, restarts, latent_dim)
        z_best = z[torch.arange(batch_size, device=device), best_restart]
        return model.decode(z_best).clamp(0.0, 1.0)


def save_comparison_grid(
    targets: torch.Tensor,
    lasso_estimates: torch.Tensor,
    vae_estimates: torch.Tensor,
    output_path: Path,
    num_images: int,
) -> None:
    num_images = min(num_images, targets.shape[0])
    grid = torch.cat(
        [
            targets[:num_images].cpu(),
            lasso_estimates[:num_images].cpu(),
            vae_estimates[:num_images].cpu(),
        ],
        dim=0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path, nrow=num_images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare random-matrix compressed sensing with Lasso and VAE priors."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--measurements",
        type=int,
        nargs="+",
        default=None,
        help="Numbers of nested random Gaussian measurements to test.",
    )
    parser.add_argument("--max-measurements", type=int, default=DEFAULT_MAX_MEASUREMENTS)
    parser.add_argument("--measurement-step", type=int, default=DEFAULT_MEASUREMENT_STEP)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--lasso-alpha", type=float, default=1e-3)
    parser.add_argument("--lasso-max-iter", type=int, default=5000)
    parser.add_argument("--lasso-tol", type=float, default=1e-4)
    parser.add_argument("--vae-steps", type=int, default=1000)
    parser.add_argument("--vae-restarts", type=int, default=5)
    parser.add_argument("--vae-lr", type=float, default=5e-2)
    parser.add_argument("--z-prior-weight", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-images", type=int, default=8)
    parser.add_argument("--vae-log-interval", type=int, default=0)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use an existing MNIST dataset in data-dir instead of downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    print(f"Using device: {device}")

    model, latent_dim, hidden_dim = load_vae(
        Path(args.checkpoint),
        device,
        args.latent_dim,
        args.hidden_dim,
    )
    print(
        f"Loaded VAE checkpoint: {args.checkpoint} "
        f"(latent_dim={latent_dim}, hidden_dim={hidden_dim})"
    )

    eval_loader = build_eval_loader(args)
    rows: list[MetricRow] = []
    if args.measurements is None:
        if args.measurement_step <= 0:
            raise ValueError("--measurement-step must be a positive integer.")
        if args.max_measurements <= 0:
            raise ValueError("--max-measurements must be a positive integer.")
        measurement_counts = list(
            range(args.measurement_step, args.max_measurements + 1, args.measurement_step)
        )
    else:
        measurement_counts = sorted(set(args.measurements))

    if not measurement_counts:
        raise ValueError("No measurement counts were provided.")
    if measurement_counts[0] <= 0:
        raise ValueError("--measurements must contain positive integers.")

    max_measurements = measurement_counts[-1]
    fixed_matrix = make_random_measurement_matrix(
        max_measurements,
        args.seed,
        device,
    )
    print(
        f"Created one fixed Gaussian A with shape "
        f"{tuple(fixed_matrix.shape)} for all measurement counts."
    )

    for num_measurements in measurement_counts:
        matrix = fixed_matrix[:num_measurements]
        print(
            f"num_measurements={num_measurements} "
            f"using A[:{num_measurements}]"
        )

        sample_offset = 0
        saved_grid = False
        for images, labels in eval_loader:
            images = images.to(device)
            labels = labels.to(device)
            measurements = measure_images(images, matrix, args.noise_std)

            lasso_images = lasso_reconstruct(
                matrix,
                measurements,
                alpha=args.lasso_alpha,
                max_iter=args.lasso_max_iter,
                tol=args.lasso_tol,
            )
            vae_images = vae_reconstruct(
                model,
                matrix,
                measurements,
                latent_dim=latent_dim,
                restarts=args.vae_restarts,
                steps=args.vae_steps,
                lr=args.vae_lr,
                z_prior_weight=args.z_prior_weight,
                log_interval=args.vae_log_interval,
            )

            rows.extend(
                metric_rows(
                    "lasso",
                    lasso_images,
                    images,
                    labels,
                    num_measurements,
                    0,
                    sample_offset,
                )
            )
            rows.extend(
                metric_rows(
                    "vae",
                    vae_images,
                    images,
                    labels,
                    num_measurements,
                    0,
                    sample_offset,
                )
            )

            if not saved_grid:
                save_comparison_grid(
                    images,
                    lasso_images,
                    vae_images,
                    output_dir / f"comparison_m{num_measurements:04d}.png",
                    args.grid_images,
                )
                saved_grid = True

            sample_offset += images.shape[0]

    per_image_path = output_dir / "per_image_metrics.csv"
    summary_path = output_dir / "summary_metrics.csv"
    plot_path = output_dir / "metric_comparison.png"

    write_per_image_metrics(rows, per_image_path)
    summary_rows = summarize_metrics(rows)
    write_summary_metrics(summary_rows, summary_path)
    plot_cs_summary(summary_rows, plot_path)

    print(f"Saved per-image metrics to: {per_image_path}")
    print(f"Saved summary metrics to: {summary_path}")
    print(f"Saved comparison plot to: {plot_path}")
    print(f"Saved image grids to: {output_dir}")


if __name__ == "__main__":
    main()
