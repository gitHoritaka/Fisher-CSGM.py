from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import optim
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
DEFAULT_CANDIDATE_MEASUREMENTS = 784


def make_radon_observation_matrix(
    num_measurements: int,
    seed: int,
    device: torch.device,
    image_size: int = 28,
    num_angles: int = 28,
    num_detectors: int = 28,
    samples_per_line: int = 112,
    detector_span: float | None = None,
    angle_max: float = 180.0,
    normalize_rows: bool = True,
) -> torch.Tensor:
    """Create A with shape (num_measurements, image_size * image_size).

    Each row is a discretized Radon line-integral measurement v=(theta, s).
    A row stores how much that line contributes to each pixel.
    """
    if num_measurements <= 0:
        raise ValueError("num_measurements must be positive.")
    if num_angles <= 0:
        raise ValueError("num_angles must be positive.")
    if num_detectors <= 0:
        raise ValueError("num_detectors must be positive.")
    if samples_per_line < 2:
        raise ValueError("samples_per_line must be at least 2.")
    if angle_max <= 0:
        raise ValueError("angle_max must be positive.")

    center = (image_size - 1) / 2.0
    radius = math.sqrt(2.0) * center
    if detector_span is None:
        detector_span = center
    if detector_span <= 0:
        raise ValueError("detector_span must be positive.")

    angles = torch.linspace(
        0.0,
        math.radians(angle_max),
        steps=num_angles + 1,
        dtype=torch.float32,
    )[:-1]
    detector_positions = torch.linspace(
        -detector_span,
        detector_span,
        steps=num_detectors,
        dtype=torch.float32,
    )
    line_positions = torch.linspace(-radius, radius, steps=samples_per_line)
    line_step = float(line_positions[1] - line_positions[0])

    rows: list[torch.Tensor] = []
    for theta in angles:
        normal_x = torch.cos(theta)
        normal_y = torch.sin(theta)
        tangent_x = -torch.sin(theta)
        tangent_y = torch.cos(theta)

        for detector_position in detector_positions:
            x = detector_position * normal_x + line_positions * tangent_x
            y = detector_position * normal_y + line_positions * tangent_y

            col = x + center
            row = center - y
            inside = (
                (row >= 0.0)
                & (row <= image_size - 1)
                & (col >= 0.0)
                & (col <= image_size - 1)
            )
            weights = torch.zeros(image_size, image_size, dtype=torch.float32)
            if torch.any(inside):
                row = row[inside]
                col = col[inside]
                row0 = torch.floor(row).long()
                col0 = torch.floor(col).long()
                row1 = torch.clamp(row0 + 1, max=image_size - 1)
                col1 = torch.clamp(col0 + 1, max=image_size - 1)
                row_frac = row - row0.float()
                col_frac = col - col0.float()

                contributions = [
                    (row0, col0, (1.0 - row_frac) * (1.0 - col_frac)),
                    (row0, col1, (1.0 - row_frac) * col_frac),
                    (row1, col0, row_frac * (1.0 - col_frac)),
                    (row1, col1, row_frac * col_frac),
                ]
                for row_index, col_index, value in contributions:
                    weights.index_put_(
                        (row_index, col_index),
                        value * line_step,
                        accumulate=True,
                    )

            flat_weights = weights.flatten()
            if normalize_rows:
                norm = flat_weights.norm()
                if norm > 0:
                    flat_weights = flat_weights / norm
            rows.append(flat_weights)

    matrix = torch.stack(rows)
    nonzero_rows = matrix.norm(dim=1) > 0
    matrix = matrix[nonzero_rows]
    if num_measurements > matrix.shape[0]:
        raise ValueError(
            "Radon candidate pool is too small after removing zero-length "
            f"lines: requested {num_measurements}, but only "
            f"{matrix.shape[0]} nonzero candidates are available."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(matrix.shape[0], generator=generator)
    return matrix[order[:num_measurements]].to(device)


def make_observation_matrix(
    args: argparse.Namespace,
    num_measurements: int,
    device: torch.device,
) -> torch.Tensor:
    if args.measurement_type == "gaussian":
        return make_random_measurement_matrix(
            num_measurements,
            args.seed,
            device,
        )
    if args.measurement_type == "radon":
        return make_radon_observation_matrix(
            num_measurements,
            args.seed,
            device,
            num_angles=args.radon_num_angles,
            num_detectors=args.radon_num_detectors,
            samples_per_line=args.radon_samples_per_line,
            detector_span=args.radon_detector_span,
            angle_max=args.radon_angle_max,
            normalize_rows=args.radon_normalize_rows,
        )
    raise ValueError(f"Unsupported measurement_type: {args.measurement_type}")


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


def build_eval_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
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
        pin_memory=device.type == "cuda",
    )


def lasso_reconstruct(
    matrix: torch.Tensor,
    measurements: torch.Tensor,
    alpha: float,
    max_iter: int,
    tol: float,
) -> torch.Tensor:
    """Run sklearn Lasso on CPU, then return estimates on measurements.device."""
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

    estimate_tensor = torch.from_numpy(np.asarray(estimates)).to(
        device=measurements.device,
        dtype=measurements.dtype,
    )
    return estimate_tensor.view(-1, 1, 28, 28).clamp(0.0, 1.0)


def optimize_latent(
    model: VAE,
    matrix: torch.Tensor,
    measurements: torch.Tensor,
    latent_dim: int,
    restarts: int,
    steps: int,
    lr: float,
    prior_precision: float,
    noise_var: float,
    log_interval: int,
    initial_z: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = measurements.shape[0]
    device = measurements.device
    candidates = batch_size * restarts

    expanded_measurements = (
        measurements[:, None, :]
        .expand(batch_size, restarts, measurements.shape[1])
        .reshape(candidates, measurements.shape[1])
    )
    if initial_z is None:
        z_init = torch.randn(candidates, latent_dim, device=device)
    else:
        initial_z = initial_z.detach().to(device)
        z_init = initial_z[:, None, :].expand(batch_size, restarts, latent_dim).clone()
        if restarts > 1:
            z_init[:, 1:, :] = torch.randn(
                batch_size,
                restarts - 1,
                latent_dim,
                device=device,
            )
        z_init = z_init.reshape(candidates, latent_dim)

    z = z_init.requires_grad_(True)
    optimizer = optim.Adam([z], lr=lr)

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        generated = model.decode(z).flatten(start_dim=1)
        predicted = generated @ matrix.T
        residual = predicted - expanded_measurements
        measurement_loss = residual.pow(2).sum(dim=1) / (2.0 * noise_var)
        prior_loss = 0.5 * prior_precision * z.pow(2).sum(dim=1)
        loss = (measurement_loss + prior_loss).mean()
        loss.backward()
        optimizer.step()

        if log_interval > 0 and step % log_interval == 0:
            print(f"vae_step={step:04d} posterior_objective={loss.item():.6f}")

    with torch.no_grad():
        generated = model.decode(z).flatten(start_dim=1)
        predicted = generated @ matrix.T
        residual = predicted - expanded_measurements
        measurement_loss = residual.pow(2).sum(dim=1) / (2.0 * noise_var)
        prior_loss = 0.5 * prior_precision * z.pow(2).sum(dim=1)
        total_loss = measurement_loss + prior_loss
        total_loss = total_loss.view(batch_size, restarts)
        best_restart = total_loss.argmin(dim=1)

        z = z.detach().view(batch_size, restarts, latent_dim)
        z_best = z[torch.arange(batch_size, device=device), best_restart]
        return z_best


def vae_reconstruct(
    model: VAE,
    matrix: torch.Tensor,
    measurements: torch.Tensor,
    latent_dim: int,
    restarts: int,
    steps: int,
    lr: float,
    prior_precision: float,
    noise_var: float,
    log_interval: int,
    initial_z: torch.Tensor | None = None,
) -> torch.Tensor:
    z_best = optimize_latent(
        model,
        matrix,
        measurements,
        latent_dim=latent_dim,
        restarts=restarts,
        steps=steps,
        lr=lr,
        prior_precision=prior_precision,
        noise_var=noise_var,
        log_interval=log_interval,
        initial_z=initial_z,
    )
    with torch.no_grad():
        return model.decode(z_best).clamp(0.0, 1.0)


def decoder_jacobian(model: VAE, z: torch.Tensor) -> torch.Tensor:
    z = z.detach().clone().requires_grad_(True)

    def decode_flat(latent: torch.Tensor) -> torch.Tensor:
        return model.decode(latent.unsqueeze(0)).flatten()

    return torch.autograd.functional.jacobian(
        decode_flat,
        z,
        create_graph=False,
        vectorize=True,
    )


def select_next_measurement(
    candidate_matrix: torch.Tensor,
    selected_mask: torch.Tensor,
    jacobian: torch.Tensor,
    precision: torch.Tensor,
    noise_var: float,
) -> tuple[int, torch.Tensor, float]:
    available_indices = torch.nonzero(~selected_mask, as_tuple=False).flatten()
    available_matrix = candidate_matrix[available_indices]
    latent_grads = available_matrix @ jacobian
    solved = torch.linalg.solve(precision, latent_grads.T).T
    gains = (latent_grads * solved).sum(dim=1).clamp_min(0.0)
    scores = 0.5 * torch.log1p(gains / noise_var)

    best_position = int(torch.argmax(scores).item())
    best_index = int(available_indices[best_position].item())
    best_grad = latent_grads[best_position]
    best_score = float(scores[best_position].item())
    return best_index, best_grad, best_score


def rebuild_precision_from_selected(
    model: VAE,
    candidate_matrix: torch.Tensor,
    selected_indices: list[int],
    z_hat: torch.Tensor,
    latent_dim: int,
    prior_precision: float,
    noise_var: float,
) -> torch.Tensor:
    """Rebuild Lambda_t = lambda I + J_G^T A_S^T A_S J_G / sigma^2."""
    device = candidate_matrix.device
    precision = prior_precision * torch.eye(latent_dim, device=device)
    if not selected_indices:
        return precision

    z_vector = z_hat.squeeze(0)
    jacobian = decoder_jacobian(model, z_vector)
    selected_matrix = candidate_matrix[selected_indices]
    latent_grads = selected_matrix @ jacobian
    return precision + latent_grads.T @ latent_grads / noise_var


def active_reconstruct_image(
    model: VAE,
    image: torch.Tensor,
    label: torch.Tensor,
    candidate_matrix: torch.Tensor,
    measurement_counts: list[int],
    latent_dim: int,
    args: argparse.Namespace,
    sample_index: int,
) -> list[MetricRow]:
    device = image.device
    rows: list[MetricRow] = []
    image_batch = image.unsqueeze(0)
    label_batch = label.reshape(1)
    candidate_measurements = measure_images(
        image_batch,
        candidate_matrix,
        args.noise_std,
    ).squeeze(0)

    selected_indices: list[int] = []
    selected_mask = torch.zeros(
        candidate_matrix.shape[0],
        dtype=torch.bool,
        device=device,
    )
    z_hat = torch.zeros(1, latent_dim, device=device)
    precision = args.acquisition_prior_precision * torch.eye(latent_dim, device=device)
    noise_var = max(args.acquisition_noise_var, args.noise_std**2, 1e-8)
    max_measurements = measurement_counts[-1]
    shared_measurements = min(
        args.active_shared_random_measurements,
        max_measurements,
        candidate_matrix.shape[0],
    )
    if shared_measurements > 0:
        selected_indices.extend(range(shared_measurements))
        selected_mask[:shared_measurements] = True
        precision = rebuild_precision_from_selected(
            model,
            candidate_matrix,
            selected_indices,
            z_hat,
            latent_dim,
            args.acquisition_prior_precision,
            noise_var,
        )
        print(
            "active-init-random "
            f"sample={sample_index + 1}/{args.num_samples} "
            f"selected={shared_measurements}/{max_measurements}",
            flush=True,
        )

    for num_measurements in measurement_counts:
        while len(selected_indices) < num_measurements:
            jacobian = decoder_jacobian(model, z_hat.squeeze(0))
            next_index, next_grad, next_score = select_next_measurement(
                candidate_matrix,
                selected_mask,
                jacobian,
                precision,
                noise_var,
            )
            selected_indices.append(next_index)
            selected_mask[next_index] = True
            precision = precision + torch.outer(next_grad, next_grad) / noise_var
            selected_count = len(selected_indices)
            should_print_progress = (
                args.active_progress_interval > 0
                and (
                    selected_count % args.active_progress_interval == 0
                    or selected_count == num_measurements
                    or selected_count == max_measurements
                )
            )
            if should_print_progress:
                print(
                    "active-select "
                    f"sample={sample_index + 1}/{args.num_samples} "
                    f"selected={selected_count}/{max_measurements} "
                    f"target={num_measurements} "
                    f"row={next_index} "
                    f"score={next_score:.6f}",
                    flush=True,
                )

            should_refit = (
                len(selected_indices) % args.active_refit_interval == 0
                or len(selected_indices) == num_measurements
            )
            if should_refit:
                active_matrix = candidate_matrix[selected_indices]
                active_measurements = candidate_measurements[selected_indices].unsqueeze(0)
                z_hat = optimize_latent(
                    model,
                    active_matrix,
                    active_measurements,
                    latent_dim=latent_dim,
                    restarts=args.active_restarts,
                    steps=args.active_refit_steps,
                    lr=args.vae_lr,
                    prior_precision=args.acquisition_prior_precision,
                    noise_var=noise_var,
                    log_interval=0,
                    initial_z=z_hat,
                )
                precision = rebuild_precision_from_selected(
                    model,
                    candidate_matrix,
                    selected_indices,
                    z_hat,
                    latent_dim,
                    args.acquisition_prior_precision,
                    noise_var,
                )

        matrix = candidate_matrix[selected_indices]
        measurements = candidate_measurements[selected_indices].unsqueeze(0)
        z_hat = optimize_latent(
            model,
            matrix,
            measurements,
            latent_dim=latent_dim,
            restarts=args.vae_restarts,
            steps=args.vae_steps,
            lr=args.vae_lr,
            prior_precision=args.acquisition_prior_precision,
            noise_var=noise_var,
            log_interval=args.vae_log_interval,
            initial_z=z_hat,
        )
        if len(selected_indices) < max_measurements:
            precision = rebuild_precision_from_selected(
                model,
                candidate_matrix,
                selected_indices,
                z_hat,
                latent_dim,
                args.acquisition_prior_precision,
                noise_var,
            )
        with torch.no_grad():
            vae_image = model.decode(z_hat).clamp(0.0, 1.0)

        rows.extend(
            metric_rows(
                "active-vae",
                vae_image,
                image_batch,
                label_batch,
                num_measurements,
                0,
                sample_index,
            )
        )

    return rows


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
        help="Numbers of nested measurements to test.",
    )
    parser.add_argument("--max-measurements", type=int, default=DEFAULT_MAX_MEASUREMENTS)
    parser.add_argument("--measurement-step", type=int, default=DEFAULT_MEASUREMENT_STEP)
    parser.add_argument(
        "--measurement-type",
        choices=["radon", "gaussian"],
        default="radon",
        help="Use structured Radon line-integral rows or Gaussian random rows.",
    )
    parser.add_argument("--radon-num-angles", type=int, default=28)
    parser.add_argument("--radon-num-detectors", type=int, default=28)
    parser.add_argument("--radon-samples-per-line", type=int, default=112)
    parser.add_argument("--radon-detector-span", type=float, default=None)
    parser.add_argument("--radon-angle-max", type=float, default=180.0)
    parser.add_argument(
        "--radon-normalize-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize each Radon row to unit L2 norm.",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--lasso-alpha", type=float, default=1e-3)
    parser.add_argument("--lasso-max-iter", type=int, default=5000)
    parser.add_argument("--lasso-tol", type=float, default=1e-4)
    parser.add_argument("--vae-steps", type=int, default=1000)
    parser.add_argument("--vae-restarts", type=int, default=5)
    parser.add_argument("--vae-lr", type=float, default=5e-2)
    parser.add_argument(
        "--z-prior-weight",
        type=float,
        default=None,
        help="Deprecated alias for --acquisition-prior-precision.",
    )
    parser.add_argument(
        "--measurement-policy",
        choices=["random", "active", "both"],
        default="both",
        help=(
            "Estimators to run: random gives lasso and random-vae, "
            "active gives lasso and active-vae, both compares all three."
        ),
    )
    parser.add_argument("--candidate-measurements", type=int, default=DEFAULT_CANDIDATE_MEASUREMENTS)
    parser.add_argument("--active-refit-steps", type=int, default=500)
    parser.add_argument("--active-refit-interval", type=int, default=5)
    parser.add_argument(
        "--active-shared-random-measurements",
        type=int,
        default=5,
        help=(
            "Start active-vae from the same first N fixed random/candidate rows "
            "used by random-vae before Fisher selection begins."
        ),
    )
    parser.add_argument(
        "--active-progress-interval",
        type=int,
        default=5,
        help="Print active row-selection progress every N selected rows. Use 0 to disable.",
    )
    parser.add_argument("--active-restarts", type=int, default=1)
    parser.add_argument("--acquisition-prior-precision", type=float, default=1.0)
    parser.add_argument("--acquisition-noise-var", type=float, default=1e-2)
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

    eval_loader = build_eval_loader(args, device)
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
    if args.active_refit_interval <= 0:
        raise ValueError("--active-refit-interval must be a positive integer.")
    if args.active_shared_random_measurements < 0:
        raise ValueError("--active-shared-random-measurements must be non-negative.")
    if args.active_progress_interval < 0:
        raise ValueError("--active-progress-interval must be zero or a positive integer.")
    if args.z_prior_weight is not None:
        args.acquisition_prior_precision = args.z_prior_weight
        print(
            "Using deprecated --z-prior-weight as "
            f"--acquisition-prior-precision={args.acquisition_prior_precision}."
        )
    if args.acquisition_prior_precision <= 0:
        raise ValueError("--acquisition-prior-precision must be positive.")
    if args.acquisition_noise_var <= 0:
        raise ValueError("--acquisition-noise-var must be positive.")

    max_measurements = measurement_counts[-1]
    run_random_vae = args.measurement_policy in {"random", "both"}
    run_active_vae = args.measurement_policy in {"active", "both"}
    run_lasso = True
    if run_active_vae and args.candidate_measurements < max_measurements:
        raise ValueError("--candidate-measurements must be at least max(measurements).")
    candidate_measurements = args.candidate_measurements if run_active_vae else max_measurements
    fixed_matrix = make_observation_matrix(
        args,
        candidate_measurements,
        device,
    )
    posterior_noise_var = max(args.acquisition_noise_var, args.noise_std**2)
    print(
        f"Created {args.measurement_type} A with shape {tuple(fixed_matrix.shape)} "
        f"for policy={args.measurement_policy}."
    )
    print(
        "Posterior objective uses "
        f"prior_precision={args.acquisition_prior_precision} "
        f"and noise_var={posterior_noise_var}."
    )

    if run_lasso or run_random_vae:
        for num_measurements in measurement_counts:
            matrix = fixed_matrix[:num_measurements]
            print(
                f"num_measurements={num_measurements} "
                f"using fixed {args.measurement_type} A[:{num_measurements}]"
            )

            sample_offset = 0
            saved_grid = False
            for images, labels in eval_loader:
                images = images.to(device, non_blocking=device.type == "cuda")
                labels = labels.to(device, non_blocking=device.type == "cuda")
                measurements = measure_images(images, matrix, args.noise_std)

                lasso_images = None
                vae_images = None
                if run_lasso:
                    lasso_images = lasso_reconstruct(
                        matrix,
                        measurements,
                        alpha=args.lasso_alpha,
                        max_iter=args.lasso_max_iter,
                        tol=args.lasso_tol,
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

                if run_random_vae:
                    vae_images = vae_reconstruct(
                        model,
                        matrix,
                        measurements,
                        latent_dim=latent_dim,
                        restarts=args.vae_restarts,
                        steps=args.vae_steps,
                        lr=args.vae_lr,
                        prior_precision=args.acquisition_prior_precision,
                        noise_var=posterior_noise_var,
                        log_interval=args.vae_log_interval,
                    )
                    rows.extend(
                        metric_rows(
                            "random-vae",
                            vae_images,
                            images,
                            labels,
                            num_measurements,
                            0,
                            sample_offset,
                        )
                    )

                if not saved_grid and lasso_images is not None and vae_images is not None:
                    save_comparison_grid(
                        images,
                        lasso_images,
                        vae_images,
                        output_dir
                        / f"{args.measurement_type}_comparison_m{num_measurements:04d}.png",
                        args.grid_images,
                    )
                    saved_grid = True

                sample_offset += images.shape[0]

    if run_active_vae:
        sample_offset = 0
        for images, labels in eval_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            for batch_index in range(images.shape[0]):
                rows.extend(
                    active_reconstruct_image(
                        model,
                        images[batch_index],
                        labels[batch_index],
                        fixed_matrix,
                        measurement_counts,
                        latent_dim,
                        args,
                        sample_offset + batch_index,
                    )
                )
            sample_offset += images.shape[0]
            print(f"Processed active-vae samples: {sample_offset}/{args.num_samples}")

    output_prefix = args.measurement_type
    per_image_path = output_dir / f"{output_prefix}_per_image_metrics.csv"
    summary_path = output_dir / f"{output_prefix}_summary_metrics.csv"
    plot_path = output_dir / f"{output_prefix}_metric_comparison.png"

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
