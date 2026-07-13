from __future__ import annotations

import csv
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"
IMAGE_DIM = 28 * 28


MetricRow = dict[str, float | int | str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib-cache"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_vae_training_metrics(
    metrics_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Plot train/test loss, reconstruction loss, and KL from metrics.csv."""
    plt = configure_matplotlib()

    metrics_path = Path(metrics_path)
    if output_path is None:
        output_path = metrics_path.with_suffix(".png")
    output_path = Path(output_path)

    with metrics_path.open(newline="") as metrics_file:
        rows = list(csv.DictReader(metrics_file))

    if not rows:
        raise ValueError(f"No rows found in metrics file: {metrics_path}")

    epochs = [int(row["epoch"]) for row in rows]

    def metric_values(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    figure, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plots = [
        ("Total loss per sample", "Loss", "train_loss", "test_loss"),
        (
            "Reconstruction BCE per sample",
            "Reconstruction",
            "train_reconstruction",
            "test_reconstruction",
        ),
        ("KL divergence per sample", "KL divergence", "train_kl", "test_kl"),
    ]

    for axis, (title, ylabel, train_key, test_key) in zip(axes, plots):
        axis.plot(epochs, metric_values(train_key), marker="o", label="train")
        axis.plot(epochs, metric_values(test_key), marker="o", label="test")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel("Epoch")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def make_random_measurement_matrix(
    num_measurements: int,
    seed: int,
    device: torch.device,
    image_dim: int = IMAGE_DIM,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(
        num_measurements,
        image_dim,
        generator=generator,
    ) / math.sqrt(num_measurements)
    return matrix.to(device)


def measure_images(
    images: torch.Tensor,
    matrix: torch.Tensor,
    noise_std: float,
) -> torch.Tensor:
    flattened = images.flatten(start_dim=1)
    measurements = flattened @ matrix.T
    if noise_std > 0:
        measurements = measurements + noise_std * torch.randn_like(measurements)
    return measurements


def psnr_rmse_batch(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    mse = (estimates - targets).pow(2).flatten(start_dim=1).mean(dim=1)
    rmse = torch.sqrt(mse)
    psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))
    return psnr, rmse


def gaussian_window(
    window_size: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords = coords - (window_size - 1) / 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d.view(1, 1, window_size, window_size)


def ssim_batch(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    window = gaussian_window(window_size, sigma, estimates.device)
    padding = window_size // 2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = F.conv2d(estimates, window, padding=padding)
    mu_y = F.conv2d(targets, window, padding=padding)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(estimates * estimates, window, padding=padding) - mu_x2
    sigma_y2 = F.conv2d(targets * targets, window, padding=padding) - mu_y2
    sigma_xy = F.conv2d(estimates * targets, window, padding=padding) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    return (numerator / denominator.clamp_min(1e-12)).flatten(start_dim=1).mean(dim=1)


def metric_rows(
    method: str,
    estimates: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    num_measurements: int,
    matrix_id: int,
    sample_offset: int,
) -> list[MetricRow]:
    psnr, rmse = psnr_rmse_batch(estimates, targets)
    ssim = ssim_batch(estimates, targets)
    rows: list[MetricRow] = []
    for batch_index in range(targets.shape[0]):
        rows.append(
            {
                "num_measurements": num_measurements,
                "matrix_id": matrix_id,
                "sample_index": sample_offset + batch_index,
                "label": int(labels[batch_index].item()),
                "method": method,
                "rmse": float(rmse[batch_index].item()),
                "psnr": float(psnr[batch_index].item()),
                "ssim": float(ssim[batch_index].item()),
            }
        )
    return rows


def write_per_image_metrics(
    rows: list[MetricRow],
    output_path: Path,
) -> None:
    fieldnames = [
        "num_measurements",
        "matrix_id",
        "sample_index",
        "label",
        "method",
        "rmse",
        "psnr",
        "ssim",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_metrics(rows: list[MetricRow]) -> list[MetricRow]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (int(row["num_measurements"]), str(row["method"]))
        for metric in ("rmse", "psnr", "ssim"):
            grouped[key][metric].append(float(row[metric]))

    summary_rows: list[MetricRow] = []
    for (num_measurements, method), metrics in sorted(grouped.items()):
        summary_row: MetricRow = {
            "num_measurements": num_measurements,
            "method": method,
            "count": len(metrics["rmse"]),
        }
        for metric, values in metrics.items():
            mean_value = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((value - mean_value) ** 2 for value in values)
                std_value = math.sqrt(variance / (len(values) - 1))
            else:
                std_value = 0.0
            summary_row[f"{metric}_mean"] = mean_value
            summary_row[f"{metric}_std"] = std_value
        summary_rows.append(summary_row)
    return summary_rows


def write_summary_metrics(
    rows: list[MetricRow],
    output_path: Path,
) -> None:
    fieldnames = [
        "num_measurements",
        "method",
        "count",
        "rmse_mean",
        "rmse_std",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_cs_summary(summary_rows: list[MetricRow], output_path: Path) -> None:
    plt = configure_matplotlib()

    methods = sorted({str(row["method"]) for row in summary_rows})
    metrics = [
        ("rmse", "RMSE", "lower is better"),
        ("psnr", "PSNR", "higher is better"),
        ("ssim", "SSIM", "higher is better"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for axis, (metric, title, subtitle) in zip(axes, metrics):
        for method in methods:
            method_rows = [
                row for row in summary_rows if str(row["method"]) == method
            ]
            method_rows.sort(key=lambda row: int(row["num_measurements"]))
            x = [int(row["num_measurements"]) for row in method_rows]
            y = [float(row[f"{metric}_mean"]) for row in method_rows]
            yerr = [float(row[f"{metric}_std"]) for row in method_rows]
            axis.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=method)
        axis.set_title(f"{title}\n{subtitle}")
        axis.set_xlabel("Measurements")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
