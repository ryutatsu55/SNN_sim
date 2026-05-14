import os
import sys

# プロジェクトルートにパスを通す
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.data.test_data
import src.models.network.connectors
import src.models.network.delays
import src.models.network.space
import src.models.network.weights

# --- プラグイン(モデル)の登録トリガー ---
import src.models.neurons.pqn_float
import src.models.neurons.pqn_int
import src.models.neurons.lif
import src.models.neurons.akita_escape_lif
import src.models.synapses.standard_models
import src.models.synapses.custom
import src.models.plasticity.custom_Akita
import src.models.plasticity.standard_models
import src.utils.visualize as visualize
from src.core.config_manager import ConfigManager
from src.core.NetworkBuilder import NetworkBuilder
from src.core.registry import DATA_LOADERS
from src.core.simulator import GeNNSimulator

TASK_NAME = "pqn_test"
OUTPUT_DIR = "output"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/spike_animation.mp4"
OUTPUT_RASTER = f"{OUTPUT_DIR}/raster"
OUTPUT_SPIKE_CSV = f"{OUTPUT_DIR}/spikes.csv"


def main():
    print("=== SNN_sim Spike Animation Pipeline Started ===")

    config_src = "configs/test.yaml"
    print(f"Loading config from {config_src}...")
    manager = ConfigManager()
    config = manager.resolve(config_src, TASK_NAME)

    print("Building Network with GeNN...")
    builder = NetworkBuilder(config)
    genn_model, io_map = builder.build(rec_spike=True)

    print("Preparing Input Data...")
    data_loader_class = DATA_LOADERS.get(TASK_NAME)
    if data_loader_class is None:
        raise ValueError(f"DataLoader '{TASK_NAME}' not found in registry.")

    data_loader = data_loader_class(config, io_map)

    print("Initializing Simulator...")
    sim = GeNNSimulator(genn_model, config, builder)
    sim.setup()

    print("Running Simulation Trials...")
    trial_results = None

    for trial_idx, (trial_inputs, meta) in enumerate(data_loader.generate()):
        print(f"  --- Trial {trial_idx + 1} ---")
        
        # --- 新しい制御フロー ---     
        step=0
        for inputs, duration_steps in trial_inputs:
            for i in range(duration_steps):
                sim.step()
                step += 1

    trial_results = sim.get_global_spikes()
    spike_times = trial_results["times"]
    spike_ids = trial_results["ids"]

    if spike_times.size == 0:
        raise ValueError(f"No spikes were recorded.")

    coords = builder.global_coords
    duration_ms = float(config.task.duration)

    print(f"Saving spike csv to {OUTPUT_SPIKE_CSV} ...")
    visualize.export_spike_csv(spike_times, spike_ids, output_path=OUTPUT_SPIKE_CSV)

    print(f"Saving raster plot to {OUTPUT_RASTER}.png ...")
    visualize.raster(spike_times, spike_ids, title=OUTPUT_RASTER)

    visualize.network(
        weights=builder.global_weights, 
        coords=builder.global_coords, 
        config=config,
        save_path=OUTPUT_DIR
    )

    print(f"Saving spike animation to {OUTPUT_VIDEO} ...")
    visualize.spike_animation(
        spike_time=spike_times,
        neuron_id=spike_ids,
        coords=coords,
        output_path=OUTPUT_VIDEO,
        fps=60,
        decay_tau_ms=500.0,
        duration_ms=duration_ms,
        title=f"Spike Animation",
    )

    print("=== Spike Animation Pipeline Complete! ===")


if __name__ == "__main__":
    main()
