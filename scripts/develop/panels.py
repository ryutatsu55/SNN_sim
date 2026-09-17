"""記録窓ごとに描く図 (ラスター / アバランチ分布 / 膜電位トレース)。

**本番の run も再解析も同じここを通る。** 以前は同じ `plot_raster(...)` 呼び出しが
`develop.py` の中に 2 か所あり、引数がわずかに食い違っていた。

GeNN を import しないので、再解析は GeNN 無しで走る。

図が 1 枚描けなくても run は落とさない。ここへ来る時点で npz と metrics の行は確定して
いるので、**数時間回した結果を描画の都合で失わない**ようにする。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils.analysis.avalanche import split_avalanches
from src.utils.plotting.distributions import plot_avalanche_distribution
from src.utils.plotting.raster import plot_raster
from src.utils.plotting.traces import neuron_trace

from scripts.develop import paths

# 論文 (Ikeda-Akita-Takahashi 2023) Fig.2 の軸。図を並べて比べるための固定値。
PAPER_RASTER_XLIM_S = (0.0, 30.0)
PAPER_AVALANCHE_XLIM = (1.0, 1000.0)
PAPER_AVALANCHE_YLIM = (1e-5, 1.0)

# ラスターの並べ替え: モジュールでブロック化し、各モジュール内で興奮性→抑制性。
# ブロック境界にだけ破線が入り (E/I の境目は色で分かるので線なし)、モジュール構造が読める。
# module 軸は space: area_uniform が複合エリアの part から供給するので、
# axon_growth_grid2 のようなモジュラーエリアの config でのみ有効。
RASTER_ORDER_AXES = ("module", "polarity")
# 並べ替え軸が 1 つも使えないときの最後の拠り所。polarity は自動軸なので必ず存在する。
FALLBACK_ORDER_AXES = ("polarity",)


def resolve_order_axes(layout, axes=RASTER_ORDER_AXES):
    """layout が実際に持っている軸だけに絞った並べ替え軸を返す。

    `RASTER_ORDER_AXES` の module 軸は `space: area_uniform` + 複合エリアの run にしか無い。
    無い軸を `plot_raster` に渡すと `NetworkLayout` が KeyError を送出し、**記録時刻に
    到達した瞬間に長い run が落ちる**ので、ここで落としておく。
    """
    if layout is None or not axes:
        return None
    available = tuple(axis for axis in axes if layout.has_axis(axis))
    if available != tuple(axes):
        dropped = [axis for axis in axes if axis not in available]
        print(f"  Note: layout に無い並べ替え軸を除外しました: {dropped}")
    if not available:
        available = FALLBACK_ORDER_AXES
    return available


def raster_ylim(total_neurons: int) -> tuple[float, float]:
    """並べ替えを行わない場合のラスター y 範囲。論文の (0, 100) は N=100 のこと。"""
    return (0.0, float(total_neurons))


def draw_raster(run_dir: Path, hour: float, local_times, ids, total_neurons: int,
                layout=None, order_axes=None) -> None:
    try:
        plot_raster(
            local_times, ids,
            paths.fig_path(run_dir, paths.RASTER, f"raster_{hour:g}h.png"),
            f"Raster {hour:g} h",
            xlim_s=PAPER_RASTER_XLIM_S,
            ylim_neuron=raster_ylim(total_neurons),
            layout=layout, order_axes=order_axes,
        )
    except Exception as error:
        print(f"  Warning: raster generation failed at {hour:g}h: {error}")


def draw_avalanche(run_dir: Path, hour: float, local_times, smax: int) -> None:
    try:
        plot_avalanche_distribution(
            split_avalanches(np.asarray(local_times, dtype=np.float64)).sizes,
            paths.fig_path(run_dir, paths.AVALANCHE, f"avalanche_{hour:g}h.png"),
            f"Avalanche distribution {hour:g} h",
            xlim=PAPER_AVALANCHE_XLIM, ylim=PAPER_AVALANCHE_YLIM, fit_smax=smax,
        )
    except Exception as error:
        print(f"  Warning: avalanche plot failed at {hour:g}h: {error}")


def draw_trace(run_dir: Path, hour: float, V, I, spike_times, spike_ids, *,
               dt: float, neuron_id: int, window_s: float) -> None:
    try:
        neuron_trace(
            V, I, spike_times, spike_ids, dt=dt, id=neuron_id, window_s=window_s,
            title=f"neuron_trace_{hour:g}h",
            save_path=str(paths.fig_dir(run_dir, paths.TRACE)),
        )
    except Exception as error:
        print(f"  Warning: neuron trace generation failed at {hour:g}h: {error}")
