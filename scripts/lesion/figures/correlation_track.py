"""平均相関の時間発展。run 全体で 1 枚。

値は `metrics.csv` の `mean_correlation` をそのまま読む (再計算しない) ので、この図と
表と probe ごとの相関行列のタイトルが食い違わない。相関の定義は
`src/utils/analysis/correlation.py`。

横軸は**切断からの経過時間**なので、切断は x = 0。ベースラインは負の側に 1 点だけ置かれ、
それが**この run の測定ではなく親 run の窓**であることを図でも区別する (`store/series.py`)。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.utils.analysis.correlation import RATE_BIN_MS
from src.utils.runview import MissingData
from scripts.lesion.figures import save

COLUMN = "mean_correlation"
HOUR_COLUMN = "hours_since_cut"

FIGSIZE = (9, 5)
DPI = 200
POST_COLOR = "tab:purple"
BASELINE_COLOR = "0.35"
# 切断の目印。`figures/firing_rate.py` の縦線と同じ位置 (x=0) を指す。
CUT_COLOR = "tab:blue"


def mean_correlation_track(series, out_path: Path) -> None:
    """平均相関が切断をまたいでどう動いたかを 1 本の折れ線で描く。

    **平均が上がっても、それだけでは回復とは読めない。** 黙ったニューロンは相関が
    未定義になり平均から抜けるので、`metrics.csv` の `correlation_pairs` が減っていない
    ことを併せて確かめること。
    """
    table = series.metrics()
    if COLUMN not in table.columns:
        raise MissingData(COLUMN, f"metrics.csv に {COLUMN} 列がありません"
                                  " (この列を持たない古い run です)")
    table = table.sort_values(HOUR_COLUMN)
    hours = table[HOUR_COLUMN].to_numpy(dtype=np.float64)
    values = table[COLUMN].to_numpy(dtype=np.float64)

    # ベースラインは負の時刻に置かれる —— それが切断軸の定義そのもの (`store/series.py`)。
    # phase の文字列を見に行かずに済むので、この図は `store/` を知らなくてよい。
    is_baseline = hours < 0.0

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(hours[~is_baseline], values[~is_baseline], marker="o", markersize=4,
            color=POST_COLOR, linewidth=1.6, label="post-lesion")
    if is_baseline.any():
        # **親 run から引き継いだ点**なので、post の折れ線とは繋がない (繋ぐと
        # 「切断の瞬間に測った」ように見える)。
        ax.plot(hours[is_baseline], values[is_baseline], marker="s", markersize=7,
                linestyle="none", color=BASELINE_COLOR,
                label="baseline (from parent run)")
        baseline = float(values[is_baseline][-1])
        ax.axhline(baseline, color=BASELINE_COLOR, linestyle=":", linewidth=1.0)

    ax.axvline(0.0, color=CUT_COLOR, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Time since lesion [h]")
    ax.set_ylabel("Mean correlation")
    ax.set_title(f"Mean neuronal correlation across the lesion\n"
                 f"(dashed line = lesion, {RATE_BIN_MS:g} ms bins)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8)
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
