"""論文 (Ikeda-Akita-Takahashi 2023) Fig.2(d) 相当の図。

個々のニューロンの発火レート推移を散布図で描く (興奮性=赤, 抑制性=青)。

レートを組み立てる `_firing_rate_series()` は以前 `store/series.py` にあったが、
**唯一の利用者がこの図**なのでここへ移した。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.analysis.spikes import firing_rates
from scripts.develop.figures import save

FIGSIZE = (10, 6)
DPI = 300


def _firing_rate_series(series) -> tuple[np.ndarray, np.ndarray]:
    """各記録時刻のニューロン別発火レートを `[時刻数, ニューロン数]` (Hz) で返す。"""
    rates = [firing_rates(window.spikes().ids, series.total_neurons,
                          window.record_window_ms)
             for window in series.windows]
    return series.hours, np.array(rates)


def fig2d(series, out_path: Path) -> None:
    """各記録時刻の全ニューロンのレートを (x=時刻, y=レート) の点として重ねる。"""
    hours, rates = _firing_rate_series(series)
    polarity = series.layout.ids_by("polarity")

    # 時刻軸をニューロン数ぶん複製して [T, N] に揃える。
    n_neurons = rates.shape[1]
    t_grid = np.repeat(hours[:, None], n_neurons, axis=1)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for key, color, label in (("inhibitory", 'blue', 'Inhibitory'),
                              ("excitatory", 'red', 'Excitatory')):
        ids = polarity.get(key)
        if ids is None or ids.size == 0:
            continue
        ids = ids[ids < n_neurons]
        ax.scatter(t_grid[:, ids].ravel(), rates[:, ids].ravel(),
                   s=6, c=color, alpha=0.35, edgecolors='none', label=label)

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title('Per-neuron firing rate development')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='upper right', markerscale=2.0)
    if hours.size:
        ax.set_xlim(hours.min() - 0.5, hours.max() + 0.5)
    ax.set_ylim(bottom=0)

    save(fig, out_path, tight_layout=False, dpi=DPI, bbox_inches='tight')
