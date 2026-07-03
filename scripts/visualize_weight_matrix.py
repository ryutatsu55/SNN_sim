import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

import matplotlib.pyplot as plt


WEIGHT_FILE_PATTERN = re.compile(r"weights_(.+)h\.npz$")
DEFAULT_EXCITATORY_COUNT = 80
DEFAULT_INHIBITORY_COUNT = 20


@dataclass(frozen=True)
class WeightFile:
    hour: float
    path: Path


@dataclass(frozen=True)
class GroupIds:
    excitatory: np.ndarray
    inhibitory: np.ndarray
    total_neurons: int
    source: str


def parse_hour_from_weight_path(path: Path) -> float:
    match = WEIGHT_FILE_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Invalid weight filename: {path.name}")
    return float(match.group(1))


def discover_weight_files(run_dir: Path) -> list[WeightFile]:
    files = []
    for path in run_dir.glob("weights_*h.npz"):
        files.append(WeightFile(hour=parse_hour_from_weight_path(path), path=path))
    return sorted(files, key=lambda item: item.hour)


def load_weight_matrix(path: Path) -> np.ndarray:
    data = np.load(path)
    if "weights" not in data.files:
        raise KeyError(f"{path} does not contain 'weights'.")
    weights = np.asarray(data["weights"], dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"{path} must contain a square 2D weight matrix.")
    return weights


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_group_ids(run_dir: Path, matrix_size: int | None = None) -> GroupIds:
    config = _load_yaml(run_dir / "config.yaml")
    neurons = config.get("neurons", {})
    simulation = config.get("simulation", {})

    if neurons and "seed" in simulation:
        total_neurons = int(simulation.get("N", sum(int(cfg["num"]) for cfg in neurons.values())))
        rng = np.random.RandomState(int(simulation["seed"]))
        available_indices = np.arange(total_neurons)
        excitatory = []
        inhibitory = []

        for cfg in neurons.values():
            num_neurons = int(cfg["num"])
            assigned = rng.choice(available_indices, size=num_neurons, replace=False)
            available_indices = np.setdiff1d(available_indices, assigned)
            assigned.sort()
            mode = cfg.get("mode")
            if (mode or "").startswith("excitatory"):
                excitatory.append(assigned)
            elif (mode or "").startswith("inhibitory"):
                inhibitory.append(assigned)

        return GroupIds(
            excitatory=np.concatenate(excitatory) if excitatory else np.array([], dtype=np.int32),
            inhibitory=np.concatenate(inhibitory) if inhibitory else np.array([], dtype=np.int32),
            total_neurons=total_neurons,
            source="config.yaml",
        )

    total_neurons = int(matrix_size or (DEFAULT_EXCITATORY_COUNT + DEFAULT_INHIBITORY_COUNT))
    exc_count = min(DEFAULT_EXCITATORY_COUNT, total_neurons)
    inh_count = max(total_neurons - exc_count, 0)
    return GroupIds(
        excitatory=np.arange(exc_count, dtype=np.int32),
        inhibitory=np.arange(exc_count, exc_count + inh_count, dtype=np.int32),
        total_neurons=total_neurons,
        source="default_80_20",
    )


def build_analysis_order(group_ids: GroupIds) -> np.ndarray:
    return np.concatenate([group_ids.excitatory, group_ids.inhibitory]).astype(np.int32)


def connection_mask_from_config(run_dir: Path, size: int) -> np.ndarray | None:
    config = _load_yaml(run_dir / "config.yaml")
    connection = config.get("network", {}).get("connection", {})
    profile = connection.get("profile_name")
    allow_self = bool(connection.get("allow_self_connections", False))
    p = connection.get("p")

    if profile == "constant_prob_full" or p == 1.0:
        mask = np.ones((size, size), dtype=bool)
        if not allow_self:
            np.fill_diagonal(mask, False)
        return mask
    return None


def _masked_values(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return values.reshape(-1)
    return values[mask]


def summarize_values(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "mean": np.nan,
            "max": np.nan,
            "nonzero_fraction": np.nan,
            "ge_0p5_fraction": np.nan,
            "ge_0p9_fraction": np.nan,
            "at_1_fraction": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "nonzero_fraction": float(np.mean(values > 0.0)),
        "ge_0p5_fraction": float(np.mean(values >= 0.5)),
        "ge_0p9_fraction": float(np.mean(values >= 0.9)),
        "at_1_fraction": float(np.mean(values >= 0.999)),
    }


def block_slices(group_ids: GroupIds) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "all": (
            np.arange(group_ids.total_neurons, dtype=np.int32),
            np.arange(group_ids.total_neurons, dtype=np.int32),
        ),
        "ee": (group_ids.excitatory, group_ids.excitatory),
        "ei": (group_ids.excitatory, group_ids.inhibitory),
        "ie": (group_ids.inhibitory, group_ids.excitatory),
        "ii": (group_ids.inhibitory, group_ids.inhibitory),
    }


def compute_block_metrics(
    hour: float,
    weights: np.ndarray,
    group_ids: GroupIds,
    connection_mask: np.ndarray | None = None,
    previous_weights: np.ndarray | None = None,
) -> list[dict[str, float | str]]:
    rows = []
    for block, (pre_ids, post_ids) in block_slices(group_ids).items():
        block_weights = weights[np.ix_(pre_ids, post_ids)]
        block_mask = None if connection_mask is None else connection_mask[np.ix_(pre_ids, post_ids)]
        values = _masked_values(block_weights, block_mask)
        stats = summarize_values(values)
        row = {"hour": hour, "block": block}
        row.update(stats)

        if previous_weights is None:
            row["mean_delta_from_previous"] = np.nan
        else:
            delta = previous_weights[np.ix_(pre_ids, post_ids)]
            delta = block_weights - delta
            row["mean_delta_from_previous"] = float(np.mean(_masked_values(delta, block_mask)))
        rows.append(row)
    return rows


def _draw_group_boundary(ax, exc_count: int, size: int) -> None:
    if 0 < exc_count < size:
        boundary = exc_count - 0.5
        ax.axhline(boundary, color="white", linewidth=1.2)
        ax.axvline(boundary, color="white", linewidth=1.2)
        ax.axhline(boundary, color="black", linewidth=0.4)
        ax.axvline(boundary, color="black", linewidth=0.4)


def plot_single_weight_matrix(
    weights: np.ndarray,
    group_ids: GroupIds,
    out_path: Path,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    order = build_analysis_order(group_ids)
    ordered = weights[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(6, 5.4))
    image = ax.imshow(ordered, origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax, cmap="viridis")
    _draw_group_boundary(ax, len(group_ids.excitatory), ordered.shape[0])
    ax.set_title(title)
    ax.set_xlabel("post neuron id")
    ax.set_ylabel("pre neuron id")
    ax.set_xticks([0, len(group_ids.excitatory) - 1, ordered.shape[0] - 1])
    ax.set_yticks([0, len(group_ids.excitatory) - 1, ordered.shape[0] - 1])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="weight")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_weight_panel(
    weight_items: list[tuple[float, np.ndarray]],
    group_ids: GroupIds,
    out_path: Path,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
) -> None:
    if not weight_items:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    order = build_analysis_order(group_ids)
    n_items = len(weight_items)
    n_cols = min(4, n_items)
    n_rows = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.9 * n_rows), squeeze=False)
    last_image = None
    for ax in axes.flat:
        ax.axis("off")
    for ax, (hour, weights) in zip(axes.flat, weight_items):
        ordered = weights[np.ix_(order, order)]
        last_image = ax.imshow(ordered, origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
        _draw_group_boundary(ax, len(group_ids.excitatory), ordered.shape[0])
        ax.set_title(f"{hour:g} h")
        ax.set_xlabel("post")
        ax.set_ylabel("pre")
        ax.axis("on")
    fig.suptitle(title)
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_metrics_csv(rows: list[dict[str, float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "hour",
        "block",
        "mean",
        "max",
        "nonzero_fraction",
        "ge_0p5_fraction",
        "ge_0p9_fraction",
        "at_1_fraction",
        "mean_delta_from_previous",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def visualize_run(run_dir: Path, output_dir: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    if output_dir is None:
        output_dir = run_dir / "weight_matrix"
    output_dir.mkdir(parents=True, exist_ok=True)

    weight_files = discover_weight_files(run_dir)
    if not weight_files:
        raise FileNotFoundError(f"No weights_*h.npz files found in: {run_dir}")

    first_weights = load_weight_matrix(weight_files[0].path)
    group_ids = infer_group_ids(run_dir, matrix_size=first_weights.shape[0])
    if group_ids.total_neurons != first_weights.shape[0]:
        raise ValueError(
            f"Config total_neurons={group_ids.total_neurons} does not match weight matrix size={first_weights.shape[0]}."
        )

    connection_mask = connection_mask_from_config(run_dir, first_weights.shape[0])
    weight_items = []
    delta_items = []
    metric_rows = []
    previous_weights = None

    for item in weight_files:
        weights = load_weight_matrix(item.path)
        if weights.shape != first_weights.shape:
            raise ValueError(f"Weight shape mismatch: {item.path}")
        weight_items.append((item.hour, weights))
        metric_rows.extend(
            compute_block_metrics(
                hour=item.hour,
                weights=weights,
                group_ids=group_ids,
                connection_mask=connection_mask,
                previous_weights=previous_weights,
            )
        )
        plot_single_weight_matrix(
            weights=weights,
            group_ids=group_ids,
            out_path=output_dir / f"weight_matrix_{item.hour:g}h.png",
            title=f"Weight matrix {item.hour:g} h",
        )
        if previous_weights is not None:
            delta_items.append((item.hour, weights - previous_weights))
        previous_weights = weights

    plot_weight_panel(
        weight_items=weight_items,
        group_ids=group_ids,
        out_path=output_dir / "weight_matrix_panel.png",
        title=f"Weight matrix timeline: {run_dir.name}",
    )
    if delta_items:
        plot_weight_panel(
            weight_items=delta_items,
            group_ids=group_ids,
            out_path=output_dir / "weight_delta_panel.png",
            title=f"Weight delta from previous record: {run_dir.name}",
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )

    write_metrics_csv(metric_rows, output_dir / "weight_block_metrics.csv")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="既存の weights_*h.npz から重み行列を可視化します。")
    parser.add_argument("run_dir", nargs="+", help="weights_*h.npz を含む実験ディレクトリ")
    parser.add_argument("--output-dir", default=None, help="単一 run_dir 実行時の出力先。未指定なら <run_dir>/weight_matrix")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if output_dir is not None and len(args.run_dir) > 1:
        print("--output-dir は run_dir が1つのときだけ指定できます。", file=sys.stderr)
        return 2

    for run_dir_arg in args.run_dir:
        out_dir = visualize_run(Path(run_dir_arg), output_dir=output_dir)
        print(f"Weight matrix visualizations saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
