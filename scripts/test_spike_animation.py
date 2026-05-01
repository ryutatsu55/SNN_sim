import os
import sys

# プロジェクトルートにパスを通す
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.registry import DATA_LOADERS
from src.core.config_manager import ConfigManager
from src.core.NetworkBuilder import NetworkBuilder
from src.core.simulator import GeNNSimulator
import src.utils.visualize as visualize

# --- プラグイン(モデル)の登録トリガー ---
import src.models.neurons.PQN_float
import src.models.neurons.PQN_int
import src.models.network.space
import src.models.network.connectors
import src.models.network.weights
import src.models.network.delays
import src.data.test_data

TASK_NAME = "pqn_test"
TARGET_POPULATION = "Layer_Exc"
OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/{TARGET_POPULATION}_spike_animation.mp4"
OUTPUT_RASTER = f"{OUTPUT_DIR}/{TARGET_POPULATION}_raster"
OUTPUT_SPIKE_CSV = f"{OUTPUT_DIR}/{TARGET_POPULATION}_spikes.csv"


def main():
    print("=== SNN_sim Spike Animation Pipeline Started ===")

    config_src = "test.yaml"
    print(f"Loading config from {config_src}...")
    manager = ConfigManager(config_src, TASK_NAME)
    config = manager.resolve()

    print("Building Network with GeNN...")
    builder = NetworkBuilder(config)
    genn_model, io_map = builder.build(rec_spike=True)

    print("Preparing Input Data...")
    data_loader_class = DATA_LOADERS.get(TASK_NAME)
    if data_loader_class is None:
        raise ValueError(f"DataLoader '{TASK_NAME}' not found in registry.")

    data_loader = data_loader_class(config, io_map)

    print("Initializing Simulator...")
    sim = GeNNSimulator(genn_model, config, io_map)
    sim.setup()

    print("Running Simulation Trials...")
    trial_results = None

    for trial_idx, (trial_inputs, meta) in enumerate(data_loader.generate()):
        print(f"  --- Trial {trial_idx + 1} ---")
        trial_results = sim.run(trial_inputs)
        sim.reset()

    if trial_results is None:
        raise RuntimeError("No trial results were generated.")

    if TARGET_POPULATION not in trial_results:
        raise ValueError(f"Population '{TARGET_POPULATION}' not found in trial results.")

    pop_results = trial_results[TARGET_POPULATION]
    if "spikes" not in pop_results:
        raise ValueError(f"Spike recording for '{TARGET_POPULATION}' is not available.")

    spike_times = pop_results["spikes"]["times"]
    spike_ids = pop_results["spikes"]["ids"]

    if spike_times.size == 0:
        raise ValueError(f"No spikes were recorded for '{TARGET_POPULATION}'.")

    coords = io_map["meta"]["global_coords"]
    duration_ms = float(config.task.duration)

    print(f"Saving spike csv to {OUTPUT_SPIKE_CSV} ...")
    visualize.export_spike_csv(spike_times, spike_ids, output_path=OUTPUT_SPIKE_CSV)

    print(f"Saving raster plot to {OUTPUT_RASTER}.png ...")
    visualize.raster(spike_times, spike_ids, title=OUTPUT_RASTER)

    print(f"Saving spike animation to {OUTPUT_VIDEO} ...")
    visualize.spike_animation(
        spike_time=spike_times,
        neuron_id=spike_ids,
        coords=coords,
        output_path=OUTPUT_VIDEO,
        fps=30,
        decay_tau_ms=20.0,
        duration_ms=duration_ms,
        title=f"{TARGET_POPULATION} Spike Animation",
    )

    print("=== Spike Animation Pipeline Complete! ===")


if __name__ == "__main__":
    main()
