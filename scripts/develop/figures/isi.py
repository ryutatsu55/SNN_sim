"""スパイク間隔 (ISI) の分布。記録窓 1 つぶんの 2 枚組。

- 左: 全ニューロンをまとめた ISI のヒストグラム (対数軸)。上端に 1/ISI = 瞬時発火率
  [Hz] の目盛りを重ねる。**平均発火率だけでは見えない「たまに高頻度で撃つ」**が
  分布の左裾として読める
- 右: ニューロンごとの CV_ISI と発火率の散布図。CV = 1 が Poisson、> 1 がバースト的、
  < 1 が規則的 (不応期が効いている)

数値の定義は `src/utils/analysis/isi.py`、要約は `metrics.csv` の `cv_isi_*` /
`lv_*` / `fano_mean` / `silent_fraction`。
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.utils.analysis.isi import cv_isi, isi_per_neuron
from src.utils.analysis.weights import excitatory_flags
from scripts.develop.figures import save

FIGSIZE = (11, 4.2)
DPI = 200
NUM_BINS = 60
# 対数軸の下端。dt より細かい間隔は出ないので、そこで切る。
MIN_ISI_MS = 0.1
# 分布に重ねる分位点。左裾 (= 高頻度発火) が読めるものを選ぶ。
PERCENTILES = (1, 5, 50)
EXC_COLOR, INH_COLOR = "tab:red", "tab:blue"


def isi_distribution(window, out_path: Path) -> None:
    """記録窓 1 つの ISI 分布と、ニューロンごとの CV_ISI。

    ISI は**ニューロンごとに**取ってから束ねる (全スパイクを 1 列にした間隔ではない)。
    後者は集団の発火率の逆数になってしまい、1 本のニューロンの発火様式を表さない。
    """
    spikes = window.spikes()
    total = int(window.total_neurons)
    values, offsets = isi_per_neuron(spikes.times, spikes.ids, total)
    if values.size == 0:
        raise ValueError("ISI が 1 つも取れませんでした (スパイクが 2 発未満)。")

    counts = np.diff(offsets)
    rates = counts / (window.record_window_ms / 1000.0)
    cv = cv_isi(values, offsets)
    is_exc = excitatory_flags(window.layout, total)

    fig, (ax_hist, ax_cv) = plt.subplots(1, 2, figsize=FIGSIZE)

    # --- 左: ISI 分布 ---
    positive = values[values > 0]
    edges = np.logspace(np.log10(max(MIN_ISI_MS, positive.min())),
                        np.log10(positive.max()), NUM_BINS)
    ax_hist.hist(positive, bins=edges, color="0.4")
    ax_hist.set_xscale("log")
    # 分位点は**枠の内側**に、高さをずらして置く。外に出すとタイトルと重なる。
    for index, percentile in enumerate(PERCENTILES):
        value = float(np.percentile(positive, percentile))
        ax_hist.axvline(value, color="tab:orange", ls="--", lw=1.0)
        ax_hist.annotate(f"{percentile}%  {value:.0f} ms ({1000/value:.0f} Hz)",
                         xy=(value, 0.97 - 0.08 * index),
                         xycoords=("data", "axes fraction"),
                         xytext=(3, 0), textcoords="offset points",
                         ha="left", va="top", fontsize=7, color="tab:orange")
    ax_hist.set_xlabel("ISI [ms]  (log)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"ISI distribution  ({positive.size} intervals)")
    ax_hist.grid(True, which="both", ls="--", alpha=0.3)

    # --- 右: CV_ISI と発火率 ---
    finite = np.isfinite(cv)
    for mask, color, label in ((finite & is_exc, EXC_COLOR, "Excitatory"),
                               (finite & ~is_exc, INH_COLOR, "Inhibitory")):
        if mask.any():
            ax_cv.scatter(rates[mask], cv[mask], s=12, color=color, alpha=0.6,
                          edgecolors="none", label=label)
    # CV = 1 は Poisson。バースト的か規則的かの境目なので必ず引く。
    ax_cv.axhline(1.0, color="0.3", ls="--", lw=1.0)
    ax_cv.annotate("Poisson", xy=(0.99, 1.0), xycoords=("axes fraction", "data"),
                   ha="right", va="bottom", fontsize=7, color="0.3")
    ax_cv.set_xlabel("firing rate [Hz]")
    ax_cv.set_ylabel("CV of ISI")
    ax_cv.set_title("regularity per neuron")
    ax_cv.grid(True, ls="--", alpha=0.3)
    if finite.any():
        ax_cv.legend(loc="upper right", fontsize=8, markerscale=1.5)

    fig.suptitle(f"ISI {window.hour:g} h")
    save(fig, out_path, dpi=DPI)
