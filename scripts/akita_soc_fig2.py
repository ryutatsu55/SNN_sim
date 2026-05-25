import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigManager
from src.core.NetworkBuilder import NetworkBuilder
from src.core.simulator import GeNNSimulator
from src.utils.akita_soc import (
    bimodality_d,
    burstiness_index,
    criticality_index_delta_cr,
    firing_rates,
    log_likelihood_ratio_power_vs_exponential,
    plot_avalanche_distribution,
    plot_raster,
    split_avalanches,
)

import src.models.neurons.akita_escape_lif
import src.models.network.connectors
import src.models.network.delays
import src.models.network.space
import src.models.network.weights
import src.models.plasticity.custom_Akita
import src.models.synapses.standard_models


TASK_NAME = "akita_soc_fig2"


def parse_args():
    parser = argparse.ArgumentParser(description="AkitaDai APL 2023 Fig.2相当の代表条件を実行します。")
    parser.add_argument("--config", default="configs/akita_soc_fig2.yaml")
    parser.add_argument("--duration-hours", type=float, default=None)
    parser.add_argument("--record-hours", type=float, nargs="*", default=None)
    parser.add_argument("--record-window-ms", type=float, default=None)
    parser.add_argument("--record-buffer-ms", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def apply_overrides(config, args):
    if args.duration_hours is not None:
        config.task.duration = args.duration_hours * 60.0 * 60.0 * 1000.0
    if args.record_hours is not None and len(args.record_hours) > 0:
        config.task.record_hours = args.record_hours
    if args.record_window_ms is not None:
        config.task.record_window_ms = args.record_window_ms
    if args.record_buffer_ms is not None:
        config.task.record_buffer_ms = args.record_buffer_ms
    if args.seed is not None:
        config.simulation.seed = args.seed


def run_steps(sim: GeNNSimulator, steps: int, chunk_steps: int, keep_spikes: bool):
    all_times = []
    all_ids = []
    remaining = int(steps)
    while remaining > 0:
        n_steps = min(remaining, chunk_steps)
        sim.step(n_steps)
        if keep_spikes:
            spikes = sim.get_global_spikes()
            if spikes["times"].size > 0:
                all_times.append(spikes["times"])
                all_ids.append(spikes["ids"])
        sim.flush_recording()
        remaining -= n_steps

    if not all_times:
        return {
            "times": np.array([], dtype=np.float32),
            "ids": np.array([], dtype=np.int32),
        }
    times = np.concatenate(all_times)
    ids = np.concatenate(all_ids)
    order = np.argsort(times)
    return {"times": times[order], "ids": ids[order]}


def save_config(config, out_dir: Path):
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, allow_unicode=True, sort_keys=False)


def main():
    args = parse_args()
    manager = ConfigManager()
    config = manager.resolve(args.config, TASK_NAME)
    apply_overrides(config, args)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / "akita_soc" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, out_dir)
    shutil.copy2(args.config, out_dir / "source_config.yaml")

    builder = NetworkBuilder(config)
    genn_model, group_info = builder.build(rec_spike=True)
    sim = GeNNSimulator(genn_model, config, builder)
    sim.setup()

    dt = float(config.simulation.dt)
    record_window_ms = float(config.task.record_window_ms)
    buffer_ms = float(getattr(config.task, "record_buffer_ms", record_window_ms))
    chunk_steps = max(1, int(buffer_ms / dt))
    record_window_steps = max(1, int(record_window_ms / dt))
    record_starts = sorted(float(hour) * 60.0 * 60.0 * 1000.0 for hour in config.task.record_hours)

    metrics_rows = []
    current_ms = 0.0
    for record_start_ms in record_starts:
        develop_ms = record_start_ms - current_ms
        if develop_ms < -1e-9:
            raise ValueError("record_hours must be sorted and non-overlapping.")
        if develop_ms > 0:
            run_steps(sim, int(round(develop_ms / dt)), chunk_steps, keep_spikes=False)
            current_ms += develop_ms

        hour = record_start_ms / (60.0 * 60.0 * 1000.0)
        weights = sim.pull_synapse("w")
        np.savez_compressed(out_dir / f"weights_{hour:g}h.npz", weights=weights)

        spikes = run_steps(sim, record_window_steps, chunk_steps, keep_spikes=True)
        current_ms += record_window_steps * dt
        local_times = spikes["times"] - record_start_ms
        np.savez_compressed(out_dir / f"spikes_{hour:g}h.npz", times=spikes["times"], ids=spikes["ids"])

        avalanche = split_avalanches(local_times)
        rates = firing_rates(spikes["ids"], builder.total_neurons, record_window_ms)
        row = {
            "hour": hour,
            "num_spikes": int(spikes["times"].size),
            "mean_rate_hz": float(np.mean(rates)),
            "avalanche_threshold_ms": avalanche.threshold_ms,
            "num_avalanches": int(avalanche.sizes.size),
            "llr": log_likelihood_ratio_power_vs_exponential(avalanche.sizes),
            "delta_cr": criticality_index_delta_cr(avalanche.sizes),
            "burstiness_index": burstiness_index(local_times, record_window_ms),
            "bimodality_d": bimodality_d(avalanche.sizes),
        }
        metrics_rows.append(row)

        plot_raster(local_times, spikes["ids"], out_dir / f"raster_{hour:g}h.png", f"Raster {hour:g} h")
        plot_avalanche_distribution(
            avalanche.sizes,
            out_dir / f"avalanche_{hour:g}h.png",
            f"Avalanche distribution {hour:g} h",
        )

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(f"Akita SoC results saved to: {out_dir}")


if __name__ == "__main__":
    main()
