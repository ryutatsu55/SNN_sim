"""単一 LIF ニューロンの興奮性オータプスを実行する最小実験。"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigManager
from src.core.NetworkBuilder import NetworkBuilder
from src.core.output_manager import create_run_output_dir
from src.core.simulator import GeNNSimulator

# レジストリ登録のトリガー
import src.models.network.connectors  # noqa: F401,E402
import src.models.network.delays  # noqa: F401,E402
import src.models.network.space  # noqa: F401,E402
import src.models.network.weights  # noqa: F401,E402
import src.models.neurons.lif  # noqa: F401,E402
import src.models.plasticity.standard_models  # noqa: F401,E402
import src.models.synapses.standard_models  # noqa: F401,E402


TASK_NAME = "pqn_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="単一 LIF ニューロンに静的な興奮性オータプスを接続して実行します。"
    )
    parser.add_argument("--config", default="configs/autapse_simple.yaml")
    parser.add_argument("--duration-ms", type=float, default=200.0)
    parser.add_argument("--i-offset", type=float, default=0.7, help="定常入力電流 [nA]")
    parser.add_argument("--weight", type=float, default=1.5, help="自己シナプス重み")
    parser.add_argument("--delay-ms", type=float, default=5.0, help="自己シナプス遅延 [ms]")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--genn-code-dir", default="genn_code")
    parser.add_argument("--gpu", action="store_true", help="CUDAバックエンドを使用する")
    return parser.parse_args()


def validate_args(args: argparse.Namespace, dt: float) -> None:
    if args.duration_ms <= 0.0:
        raise ValueError("--duration-ms は正の値にしてください。")
    if args.delay_ms < 0.0:
        raise ValueError("--delay-ms は 0 以上にしてください。")
    if args.weight < 0.0:
        raise ValueError("興奮性実験の --weight は 0 以上にしてください。")
    if round(args.duration_ms / dt) < 1:
        raise ValueError("実行時間が1タイムステップ未満です。")


def apply_overrides(config, args: argparse.Namespace) -> None:
    config.simulation.seed = args.seed
    config.task.duration = args.duration_ms
    config.neurons["AutapseNeuron"].Ioffset = args.i_offset
    config.network.weight.base_weight = args.weight
    config.network.delay.value = args.delay_ms


def validate_autapse(builder: NetworkBuilder) -> None:
    index = builder.synapse_index.get("AutapseNeuron_to_AutapseNeuron")
    if index is None or index.num_synapses != 1:
        count = 0 if index is None else index.num_synapses
        raise RuntimeError(f"オータプスは1本必要ですが、{count}本生成されました。")
    if int(index.global_src[0]) != 0 or int(index.global_tgt[0]) != 0:
        raise RuntimeError("生成された接続が 0 -> 0 ではありません。")


def save_plot(output_dir: Path, times: np.ndarray, voltage: np.ndarray, spikes: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, voltage, color="tab:blue", linewidth=1.2, label="Membrane voltage")
    if spikes["times"].size:
        spike_y = np.interp(spikes["times"], times, voltage)
        ax.scatter(spikes["times"], spike_y, color="tab:red", s=20, zorder=3, label="Spike")
    ax.set(xlabel="Time [ms]", ylabel="Voltage [mV]", title="Single-neuron excitatory autapse")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "membrane_voltage.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manager = ConfigManager()
    config = manager.resolve(args.config, TASK_NAME)
    validate_args(args, config.simulation.dt)
    apply_overrides(config, args)
    config.simulation.backend = "cuda" if args.gpu else "cpu"

    output_dir = create_run_output_dir("single_autapse", base_dir=args.out_dir)
    builder = NetworkBuilder(
        config,
        model_name="SingleAutapse",
        code_gen_dir=args.genn_code_dir,
    )
    genn_model, _ = builder.build(rec_spike=True)
    validate_autapse(builder)

    sim = GeNNSimulator(genn_model, config, builder)
    sim.setup(backup_initial_states=False)

    steps = int(round(args.duration_ms / config.simulation.dt))
    times = (np.arange(steps, dtype=np.float64) + 1.0) * config.simulation.dt
    voltage = np.empty(steps, dtype=np.float64)
    for step in range(steps):
        sim.step()
        voltage[step] = sim.pull("V")[0]

    spikes = sim.get_global_spikes()
    np.savetxt(
        output_dir / "membrane_voltage.csv",
        np.column_stack((times, voltage)),
        delimiter=",",
        header="time_ms,voltage_mV",
        comments="",
    )
    np.savetxt(
        output_dir / "spikes.csv",
        np.column_stack((spikes["times"], spikes["ids"])),
        delimiter=",",
        header="time_ms,neuron_id",
        comments="",
    )
    manager.save_config(config, output_dir)
    save_plot(output_dir, times, voltage, spikes)

    print("=== オータプス実験完了 ===")
    print("接続: 0 -> 0（1本）")
    print(f"発火数: {spikes['times'].size}")
    print(f"発火時刻 [ms]: {spikes['times'].tolist()}")
    print(f"出力先: {output_dir}")


if __name__ == "__main__":
    main()
