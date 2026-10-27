"""切断後に片方の STDP を止めた run と止めない run を比べる 2 枚。損傷 run の束 (`Collection`) から描く。

    freeze_rate_track      切断からの時刻 → ネットワーク全体の発火率 / 切断前
    freeze_recovery_split  各則の寄与 Δe / Δi と Δe + Δi、止めない run の回復

条件は止めない (`none`)・e-STDP を止めた・i-STDP を止めた の 3 つと、対照 (切らずに走り続けた run)。
定義は `analysis/freeze_split.py`。見た目は用量反応の図 (`dose_response.py`) に揃える。
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.utils.runview import MissingData
from scripts.lesion.analysis.freeze_split import (CONTROL, E_RULE, I_RULE, NONE,
                                                  measure_collection, split_table)
from scripts.lesion.analysis.summary import LATE_HOURS
from scripts.lesion.figures import save
from scripts.lesion.figures.dose_response import CONTROL_COLOR, DPI, INK, MUTED, RC

# 条件 → (凡例の名前, 色)。並びが凡例の順。
RULE_STYLE = {
    NONE: ("no freeze (both STDP)", INK),
    E_RULE: ("e-STDP frozen", "#2a78d6"),
    I_RULE: ("i-STDP frozen", "#eb6834"),
    CONTROL: ("control (no cut)", CONTROL_COLOR),
}
TRACK_FIGSIZE = (6.4, 4.4)
SPLIT_FIGSIZE = (6.4, 4.6)
SEED_LINE_ALPHA = 0.25
BAND_ALPHA = 0.18
SEED_LINK_COLOR = "#cfcec7"
MEAN_BAR_HALF_WIDTH = 0.22
REC_LABEL = f"recovery  (mean {LATE_HOURS[0]:g}–{LATE_HOURS[1]:g} h − 0 h) / pre"


def _one_spec(measured) -> str:
    specs = sorted(set(measured.runs.spec))
    if len(specs) != 1:
        raise MissingData("freeze figures", f"止めた run を持つ spec が {len(specs)} 個あります: {specs}")
    return specs[0]


def _mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """seed を行にした (seed, 時刻) の配列の、列ごとの平均と SEM。"""
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0]) if values.shape[0] > 1 \
        else np.zeros_like(mean)
    return mean, sem


def freeze_rate_track(collection, out_path) -> None:
    """切断からの時刻ごとのネットワーク全体の発火率 / 切断前を、条件ごとに seed 平均 ± SEM で描く。

    細い線は seed 1 本ずつ。対照は切断時刻を 0 h に読み替え、損傷 run の記録の範囲だけ描く。
    """
    measured = measure_collection(collection)
    spec = _one_spec(measured)
    tracks = measured.tracks
    lesion_hours = tracks.loc[tracks.rule != CONTROL, "hour"]
    lo, hi = float(lesion_hours.min()), float(lesion_hours.max())
    tracks = tracks[(tracks.hour >= lo - 1e-9) & (tracks.hour <= hi + 1e-9)]

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=TRACK_FIGSIZE)
        ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
        ax.axvline(0.0, color=MUTED, lw=0.8, ls=":")
        for rule, (name, color) in RULE_STYLE.items():
            sub = tracks[tracks.rule == rule]
            if sub.empty:
                continue
            table = sub.pivot(index="seed", columns="hour", values="rate").dropna(axis=1)
            hours, values = table.columns.to_numpy(dtype=float), table.to_numpy()
            for row in values:
                ax.plot(hours, row, color=color, lw=0.6, alpha=SEED_LINE_ALPHA)
            mean, sem = _mean_sem(values)
            ax.fill_between(hours, mean - sem, mean + sem, color=color, alpha=BAND_ALPHA, lw=0)
            ax.plot(hours, mean, color=color, lw=1.8, label=f"{name} (n={values.shape[0]})")
        ax.set_xlabel("time since lesion [h]")
        ax.set_ylabel("firing rate / pre-lesion")
        ax.set_title(f"Firing rate with one STDP rule frozen  [{spec}]\n"
                     "mean ± SEM over seeds (thin lines: each seed)", fontsize=9)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=8)
        save(fig, out_path, tight_layout=False, dpi=DPI, bbox_inches="tight")


def freeze_recovery_split(collection, out_path) -> None:
    """各則の寄与 Δe / Δi、Δe + Δi、止めない run の回復を、seed ごとの点と平均の横棒で描く。

    同じ seed の点を線で結ぶ。
    """
    measured = measure_collection(collection)
    spec = _one_spec(measured)
    split = split_table(measured.runs)
    columns = ["delta_e", "delta_i", "delta_sum", "rec_none"]
    names = ["Δe = rec(i-STDP frozen)", "Δi = rec(e-STDP frozen)", "Δe + Δi", "rec(none)"]
    # Δe は i-STDP を止めた run の回復そのものなので、その run の色で描く (Δi も同様)。
    colors = [RULE_STYLE[I_RULE][1], RULE_STYLE[E_RULE][1], MUTED, INK]
    positions = np.arange(len(columns))

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=SPLIT_FIGSIZE)
        ax.axhline(0.0, color=MUTED, lw=0.8)
        for _, row in split.iterrows():
            ax.plot(positions, row[columns].to_numpy(dtype=float), color=SEED_LINK_COLOR,
                    lw=0.8, zorder=1)
        for x, column, color in zip(positions, columns, colors):
            values = split[column].to_numpy(dtype=float)
            ax.scatter(np.full(values.size, x), values, s=22, color=color, zorder=2)
            ax.hlines(values.mean(), x - MEAN_BAR_HALF_WIDTH, x + MEAN_BAR_HALF_WIDTH,
                      color=color, lw=2.4, zorder=3)
        ax.set_xticks(positions, names, fontsize=8)
        ax.set_ylabel(f"contribution to {REC_LABEL}")
        ax.set_title(f"Contribution of each STDP rule to recovery  [{spec}], n={len(split)} seeds\n"
                     "points: seeds (lines join the same seed), bars: mean", fontsize=9)
        ax.text(0.99, 0.02, "no run froze both rules: additivity is not tested",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)
        save(fig, out_path, dpi=DPI)
