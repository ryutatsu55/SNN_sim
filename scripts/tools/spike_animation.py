"""config を 1 本回して、発火の時間発展を MP4 にする道具。

    python -m scripts.tools.spike_animation

**実験ではなく道具。** 記録窓を複数持たず、再解析の経路も無いので
`scripts/develop/` の 5 層は敷かない (判定軸は `docs/scripts_unification_plan.md`)。
出力は `outputs/spike_animation/<日時>/`。

図はすべて共有層 (`src/utils/plotting/`) のものを、契約 (`src/utils/runview.py`) 越しに
呼ぶ。記録ファイルを持たないので、走らせた結果は `MemoryWindow` に載せる ——
実験と違って「一度 npz に書いてから読み直す」先が無いための例外で、理由は
`MemoryWindow` の docstring にある。

ffmpeg が要る (`spike_animation`)。無い環境では動画だけ落ちるが、ラスターと
ネットワーク図は残る。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config_manager import ConfigManager
from src.core.NetworkBuilder import NetworkBuilder
from src.core.output_manager import create_run_output_dir
from src.core.registry import DATA_LOADERS
from src.core.simulator import GeNNSimulator
from src.utils.analysis.spikes import export_spike_csv
from src.utils.plotting import network, raster, spike_animation
from src.utils.runview import BuiltNetwork, MemoryWindow, Spikes, guard

# --- プラグイン(モデル)の登録トリガー ---
import src.data.test_data
import src.models.network.connectors
import src.models.network.delays
import src.models.network.space
import src.models.network.weights
import src.models.neurons.akita_escape_lif
import src.models.neurons.lif
import src.models.neurons.pqn_float
import src.models.neurons.pqn_int
import src.models.plasticity.custom_Akita
import src.models.plasticity.standard_models
import src.models.synapses.custom
import src.models.synapses.standard_models

CONFIG_PATH = "configs/test.yaml"
TASK_NAME = "pqn_test"
EXPERIMENT = "spike_animation"


def main():
    os.chdir(_PROJECT_ROOT)
    print("=== SNN_sim Spike Animation Pipeline Started ===")

    print(f"Loading config from {CONFIG_PATH}...")
    manager = ConfigManager()
    config = manager.resolve(CONFIG_PATH, TASK_NAME)
    output_dir = create_run_output_dir(EXPERIMENT)
    print(f"Output directory: {output_dir}")

    print("Building Network with GeNN...")
    builder = NetworkBuilder(config)
    genn_model, layout = builder.build(rec_spike=True)

    print("Preparing Input Data...")
    data_loader_class = DATA_LOADERS.get(TASK_NAME)
    if data_loader_class is None:
        raise ValueError(f"DataLoader '{TASK_NAME}' not found in registry.")
    data_loader = data_loader_class(config, layout)

    print("Initializing Simulator...")
    sim = GeNNSimulator(genn_model, config, builder)
    sim.setup()

    print("Running Simulation Trials...")
    for trial_idx, (trial_inputs, _meta) in enumerate(data_loader.generate()):
        print(f"  --- Trial {trial_idx + 1} ---")
        for _inputs, duration_steps in trial_inputs:
            for _ in range(duration_steps):
                sim.step()

    recorded = sim.get_global_spikes()
    if recorded["times"].size == 0:
        raise ValueError("No spikes were recorded.")

    # **記録を持たないので MemoryWindow に載せる。** 実験ならここは npz へ書いて
    # 読み直すところ (`scripts/develop/run_one.py` の record_once)。
    # 時刻は窓の先頭 (= t=0) を原点とするローカル時刻で、この run では絶対時刻と同じ。
    window = MemoryWindow(output_dir, config, layout,
                          spikes=Spikes(times=recorded["times"], ids=recorded["ids"]),
                          coords=builder.global_coords)
    built = BuiltNetwork.from_builder(builder, output_dir)

    print(f"Saving spike csv to {output_dir / 'spikes.csv'} ...")
    export_spike_csv(recorded["times"], recorded["ids"],
                     output_path=output_dir / "spikes.csv")

    guard("raster", raster, window, output_dir / "raster.png")
    guard("network", network, built, output_dir / "network.png")
    guard("spike animation", spike_animation, window,
          output_dir / "spike_animation.mp4")

    print(f"=== Spike Animation Pipeline Complete! Results saved to: {output_dir} ===")


if __name__ == "__main__":
    main()
