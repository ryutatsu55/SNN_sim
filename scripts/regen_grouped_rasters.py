#!/usr/bin/env python3
"""既存 outputs のラスタープロットを、ニューロングループ順に並べ替えた版で上書きする。

重み分布の可視化と同様に、興奮性を先頭ブロック (表示ID 1..Nexc)、抑制性を後続ブロック
(Nexc+1..Nexc+Ninh) に配置し直したラスターを生成する。metrics/avalanche 等の他の成果物は
一切変更しない (raster_*h.png のみ上書き)。

グループ割当は各 run の config.yaml から NetworkLayout.from_config() で決定論的に再構築する
(config のニューロン順に連番割り当てするため、本番実行と同一の割当が得られる)。

使い方:
    python scripts/regen_grouped_rasters.py                 # outputs/ 以下すべて
    python scripts/regen_grouped_rasters.py outputs/akita_soc_delay
    python scripts/regen_grouped_rasters.py outputs/20260529-144639   # 単一 run
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.core.config_manager import ConfigManager
from src.core.layout import NetworkLayout
from src.utils.akita_soc import plot_raster

# NetworkBuilder が参照するコンポーネントの登録をトリガーする
import src.models.neurons.akita_escape_lif  # noqa: F401
import src.models.network.connectors  # noqa: F401
import src.models.network.delays  # noqa: F401
import src.models.network.space  # noqa: F401
import src.models.network.weights  # noqa: F401
import src.models.plasticity.custom_Akita  # noqa: F401
import src.models.synapses.standard_models  # noqa: F401

# akita_soc_fig2.py と同じ定数・ヘルパを再利用する
from scripts.akita_soc_fig2 import (
    PAPER_RASTER_XLIM_S,
    PAPER_RASTER_YLIM_NEURON,
    discover_spike_files,
)


def find_run_dirs(base: Path) -> list[Path]:
    """base 自身、または base 以下で config.yaml と spikes_*h.npz を持つ run ディレクトリを列挙。"""
    if (base / "config.yaml").exists() and any(base.glob("spikes_*h.npz")):
        return [base]
    run_dirs = set()
    for spike_path in base.rglob("spikes_*h.npz"):
        run_dir = spike_path.parent
        if (run_dir / "config.yaml").exists():
            run_dirs.add(run_dir)
    return sorted(run_dirs)


def reconstruct_group_ids(config):
    return NetworkLayout.from_config(config).ids_by_mode()


def regen_run(run_dir: Path, manager: ConfigManager) -> int:
    config = manager.load_resolved(run_dir / "config.yaml")
    group_ids = reconstruct_group_ids(config)
    spike_files = discover_spike_files(run_dir)
    count = 0
    for hour, spike_path in spike_files:
        spikes = np.load(spike_path)
        times = spikes["times"]
        ids = spikes["ids"]
        record_start_ms = hour * 60.0 * 60.0 * 1000.0
        local_times = times - record_start_ms
        plot_raster(
            local_times,
            ids,
            run_dir / f"raster_{hour:g}h.png",
            f"Raster {hour:g} h",
            xlim_s=PAPER_RASTER_XLIM_S,
            ylim_neuron=PAPER_RASTER_YLIM_NEURON,
            group_ids=group_ids,
        )
        count += 1
    return count


def main() -> None:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs")
    if not base.exists():
        print(f"Path not found: {base}")
        sys.exit(1)

    run_dirs = find_run_dirs(base)
    if not run_dirs:
        print(f"No run dirs (config.yaml + spikes_*h.npz) found under: {base}")
        sys.exit(1)

    manager = ConfigManager()
    ok = fail = total_plots = 0
    for run_dir in run_dirs:
        try:
            n = regen_run(run_dir, manager)
            total_plots += n
            ok += 1
            print(f"[OK]   {run_dir}  ({n} rasters)")
        except Exception as e:
            fail += 1
            print(f"[FAIL] {run_dir}  {type(e).__name__}: {e}")

    print(f"\nDone. runs OK={ok} FAIL={fail}, rasters overwritten={total_plots}")


if __name__ == "__main__":
    main()
