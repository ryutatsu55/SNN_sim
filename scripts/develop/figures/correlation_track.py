"""平均相関の時間発展。run 全体で 1 枚。

値は `metrics.csv` の `mean_correlation` をそのまま読む (再計算しない) ので、この図と
表と記録窓ごとの相関行列のタイトルが食い違わない。相関の定義は
`src/utils/analysis/correlation.py`。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.utils.analysis.correlation import RATE_BIN_MS
from src.utils.runview import MissingData
from scripts.develop.figures import save

COLUMN = "mean_correlation"
HOUR_COLUMN = "hour"

FIGSIZE = (9, 5)
DPI = 200
COLOR = "tab:purple"


def mean_correlation_track(series, out_path: Path) -> None:
    """平均相関が発達を通してどう動いたかを 1 本の折れ線で描く。

    **平均が上がっても、それだけでは同期の強まりとは読めない。** 黙ったニューロンは
    相関が未定義になり平均から抜けるので、`metrics.csv` の `correlation_pairs` が
    減っていないことを併せて確かめること。
    """
    table = series.metrics()
    if COLUMN not in table.columns:
        raise MissingData(COLUMN, f"metrics.csv に {COLUMN} 列がありません"
                                  " (この列を持たない古い run です)")
    table = table.sort_values(HOUR_COLUMN)
    hours = table[HOUR_COLUMN].to_numpy(dtype=np.float64)
    values = table[COLUMN].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(hours, values, marker="o", markersize=4, color=COLOR, linewidth=1.6)

    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Mean correlation")
    ax.set_title(f"Mean neuronal correlation development ({RATE_BIN_MS:g} ms bins)")
    ax.grid(True, linestyle="--", alpha=0.4)
    span = float(hours.max() - hours.min())
    pad = 0.05 * span if span > 0 else 0.5
    ax.set_xlim(hours.min() - pad, hours.max() + pad)
    # **0 で切らない。** 相関は負を取りうるので (負を潰すのは相関行列の図だけの約束)、
    # 下端はデータに合わせる。0 は基準線として必ず入れる。
    finite = values[np.isfinite(values)]
    top = float(finite.max()) if finite.size else 1.0
    bottom = float(finite.min()) if finite.size else 0.0
    height = max(top - min(bottom, 0.0), 1e-3)
    ax.set_ylim(min(bottom, 0.0) - 0.05 * height, top + 0.15 * height)
    ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
    save(fig, out_path, dpi=DPI)
