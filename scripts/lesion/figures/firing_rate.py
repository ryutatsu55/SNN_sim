"""ニューロン別の発火レートが切断をまたいでどう動いたか (develop の fig2d の損傷版)。

develop との違いは **窓幅を probe ごとに読む**こと。あちらは config の
`record_window_ms` 1 つで割ればよいが、損傷実験では baseline と post で窓幅を変えられる。
全部を同じ幅で割ると、窓の違いがそのまま発火率の段差として現れてしまう。

ニューロン数は切断で変わらない (消したのはシナプス) ので、重み軌跡と違って引き当ては要らない。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.analysis.spikes import firing_rates
from scripts.lesion.figures import save

FIGSIZE = (10, 6)
DPI = 200


def _rate_series(series) -> tuple[np.ndarray, np.ndarray]:
    """各 probe のニューロン別発火レートを `[probe 数, ニューロン数]` (Hz) で返す。"""
    rates = [firing_rates(window.spikes().ids, series.total_neurons,
                          window.record_window_ms)
             for window in series.windows]
    return series.hours, np.asarray(rates)


def firing_rate_scatter(series, out_path: Path) -> None:
    """ニューロン別の発火レート推移を散布図で描く。

    興奮性=赤 / 抑制性=青。切断の瞬間に縦の破線を引き、E/I それぞれの平均を重ねる ——
    点が N×T 個あると個々の軌跡は追えないので、群としての落ち込みと戻りは平均線で読ませる。
    """
    hours, rates = _rate_series(series)
    polarity = series.layout.ids_by("polarity")

    n_neurons = rates.shape[1]
    t_grid = np.repeat(hours[:, None], n_neurons, axis=1)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for key, color, label in (("inhibitory", "blue", "Inhibitory"),
                              ("excitatory", "red", "Excitatory")):
        ids = polarity.get(key)
        if ids is None or np.asarray(ids).size == 0:
            continue
        ids = np.asarray(ids)
        ids = ids[ids < n_neurons]
        ax.scatter(t_grid[:, ids].ravel(), rates[:, ids].ravel(),
                   s=6, c=color, alpha=0.35, edgecolors="none", label=label)
        ax.plot(hours, rates[:, ids].mean(axis=1), color=color, linewidth=1.8,
                label=f"{label} mean")

    ax.axvline(0.0, color="tab:blue", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Time since lesion [h]")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Per-neuron firing rate across the lesion\n(dashed line = lesion)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", markerscale=2.0, fontsize=8)
    span = float(hours.max() - hours.min())
    pad = 0.05 * span if span > 0 else 0.5
    ax.set_xlim(hours.min() - pad, hours.max() + pad)
    ax.set_ylim(bottom=0)
    save(fig, out_path, dpi=DPI)
