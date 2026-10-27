"""損傷の用量反応。損傷 run の束 (`Collection`) を 1 run = 1 点で描く 5 枚。

    cut_weight          切断数 [%] → 切った重み [%] (+ y = x)
    activity_drop       切断数 [%] → 発火率の低下
    recovery_vs_drop    発火率の低下 → 回復 (+ 対照の幅)
    recovery_vs_corr_drop 平均相関の低下 → 回復 (+ 対照の幅)
    recovery_vs_cut     切断数 [%] → 回復 (+ 対照の幅)

色 = 切断の種類。同じ seed の割合違い (入れ子の抽選) は切断数の順に細線で結ぶ。凡例に種類ごとの
Spearman ρ (* は p < 0.05、多重比較の補正なし)。列の定義は `analysis/summary.py`。
**全部が同じ描き方 (色・線・凡例) を共有する**ので 1 ファイルに置く。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from scripts.lesion.figures import save

# 種類 → (凡例の名前, 色, マーカー)。並びが凡例の順。
KIND_STYLE = {
    "bridge": ("bridge cut", "#2a78d6", "o"),
    "random": ("random, all synapses", "#eb6834", "s"),
    "random_exc": ("random, E-origin only", "#1baf7a", "^"),
}
# 割合を変えて同じ seed で入れ子に切った種類。細線で結ぶ。
NESTED_KINDS = ("random", "random_exc")

INK, MUTED = "#1f1f1e", "#6b6a64"
CONTROL_COLOR = "#b9b8b0"
FIGSIZE = (5.6, 4.4)
DPI = 150
RC = {
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3dc", "grid.linewidth": 0.6,
    "legend.frameon": False,
}

CUT_LABEL = "synapses cut [% of all synapses]"
DROP_LABEL = "rate drop  (pre − 0 h) / pre"
CORR_DROP_LABEL = "mean correlation drop  (pre − 0 h) / pre"


def _rec_label(runs) -> str:
    """回復の軸名。測った区間は表の列 (`late_from_h` / `late_to_h`) から取る。"""
    lo, hi = runs.late_from_h.iloc[0], runs.late_to_h.iloc[0]
    return f"recovery  (mean {lo:g}–{hi:g} h − 0 h) / pre"


def _scatter(ax, runs, x: str, y: str) -> None:
    for kind, (name, color, marker) in KIND_STYLE.items():
        sub = runs[runs.kind == kind]
        if sub.empty:
            continue
        if kind in NESTED_KINDS:
            for _, same_seed in sub.sort_values("cut_pct").groupby("seed"):
                ax.plot(same_seed[x], same_seed[y], color=color, lw=0.6, alpha=0.35, zorder=1)
        rho, p = spearmanr(sub[x], sub[y])
        star = "*" if p < 0.05 else ""
        ax.scatter(sub[x], sub[y], s=26, color=color, marker=marker, edgecolor="white",
                   linewidth=0.8, zorder=3,
                   label=f"{name} (n={len(sub)})  ρ={rho:+.2f}{star}")


def _control_band(ax, control) -> None:
    ax.axhspan(control.rec.min(), control.rec.max(), color=CONTROL_COLOR, alpha=0.35, lw=0,
               zorder=0, label=f"no-cut control, min–max (n={len(control)})")
    ax.axhline(0.0, color=MUTED, lw=0.8, zorder=0)


def _finish(fig, ax, collection, title: str, out_path) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    ax.legend(loc="upper left", fontsize=7.5, handletextpad=0.3)
    fig.text(0.01, 0.012, f"{collection.cond_dir.name}   ρ: Spearman within kind (* p<0.05)",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out_path, tight_layout=False, dpi=DPI)


def cut_weight(collection, out_path) -> None:
    """切断数 [%] と切った重み [%]。y = x に乗れば、切った結合は平均的な重みを持つ。"""
    runs = collection.runs()
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        _scatter(ax, runs, "cut_pct", "cut_weight_pct")
        lim = max(runs.cut_pct.max(), runs.cut_weight_pct.max()) * 1.05
        ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--", zorder=0)
        ax.text(lim * 0.97, lim * 0.82, "y = x", color=MUTED, ha="right", va="top", fontsize=8)
        ax.set(xlabel=CUT_LABEL, ylabel="weight cut [% of total weight]")
        _finish(fig, ax, collection, "Cut weight vs. cut count", out_path)


def activity_drop(collection, out_path) -> None:
    """切断数 [%] と発火率の低下。"""
    runs = collection.runs()
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        _scatter(ax, runs, "cut_pct", "drop")
        ax.set(xlabel=CUT_LABEL, ylabel=DROP_LABEL)
        _finish(fig, ax, collection, "Activity drop vs. cut count", out_path)


def recovery_vs_drop(collection, out_path) -> None:
    """発火率の低下と回復。同じ低下で種類ごとの回復を比べる。"""
    runs, control = collection.runs(), collection.control()
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        _control_band(ax, control)
        _scatter(ax, runs, "drop", "rec")
        ax.set(xlabel=DROP_LABEL, ylabel=_rec_label(runs))
        _finish(fig, ax, collection, "Recovery vs. activity drop", out_path)


def recovery_vs_corr_drop(collection, out_path) -> None:
    """平均相関の低下と回復。同期の崩れ方で種類ごとの回復を比べる。"""
    runs, control = collection.runs(), collection.control()
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        _control_band(ax, control)
        _scatter(ax, runs, "corr_drop", "rec")
        ax.set(xlabel=CORR_DROP_LABEL, ylabel=_rec_label(runs))
        _finish(fig, ax, collection, "Recovery vs. correlation drop", out_path)


def recovery_vs_cut(collection, out_path) -> None:
    """切断数 [%] と回復。対照 (切らなかった同じ時間) の幅と比べる。"""
    runs, control = collection.runs(), collection.control()
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        _control_band(ax, control)
        _scatter(ax, runs, "cut_pct", "rec")
        ax.set(xlabel=CUT_LABEL, ylabel=_rec_label(runs))
        _finish(fig, ax, collection, "Recovery vs. cut count", out_path)
