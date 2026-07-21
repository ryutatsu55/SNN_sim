#!/usr/bin/env python
"""
C. elegansネットワークの結合距離・結合重みの分布を可視化するスクリプト

- 接続を持つニューロン対のユークリッド距離の分布 (全体 / 化学 / 電気)
- 化学シナプス重み・電気シナプス重みの分布 (非ゼロ要素)

synapse_mask.csv の値の意味:
    0: 接続なし, 1: 化学のみ, -1: 電気のみ, 2: 化学+電気の両方

使用方法:
    python scripts/c_elegans_distributions.py
    # output/c_elegans_distance_dist.png と
    #   output/c_elegans_weight_dist.png を生成
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pair_distances(coords, sources, targets):
    """始点・終点インデックス配列から各対のユークリッド距離を返す"""
    diff = coords[sources] - coords[targets]
    return np.linalg.norm(diff, axis=1)


def print_stats(name, values):
    if len(values) == 0:
        print(f"  {name}: (要素なし)")
        return
    print(
        f"  {name}: n={len(values)}, "
        f"mean={np.mean(values):.3f}, median={np.median(values):.3f}, "
        f"std={np.std(values):.3f}, min={np.min(values):.3f}, max={np.max(values):.3f}"
    )


def main():
    data_dir = Path("src/models/network/data/c_elegans")

    print("Loading C. elegans data...")
    df = pd.read_csv(data_dir / "ordered_coords.csv")
    mask = np.loadtxt(data_dir / "synapse_mask.csv", delimiter=",").astype(int)
    weight_chem = np.loadtxt(data_dir / "weight_matrix_chem.csv", delimiter=",")
    weight_elec = np.loadtxt(data_dir / "weight_matrix_elec.csv", delimiter=",")

    coords = df[["X", "Y", "Z"]].values.astype(np.float64)
    n = len(df)
    print(f"Loaded {n} neurons, mask shape {mask.shape}")

    # 自己結合は距離0なので除外する
    no_self = ~np.eye(n, dtype=bool)

    # --- 結合ごとの接続対を抽出 -------------------------------------------
    chem_conn = ((mask == 1) | (mask == 2)) & no_self   # 化学 (有向)
    elec_conn = ((mask == -1) | (mask == 2)) & no_self  # 電気 (対称)
    any_conn = (mask != 0) & no_self                    # いずれかの接続

    chem_s, chem_t = np.where(chem_conn)
    elec_s, elec_t = np.where(elec_conn)
    any_s, any_t = np.where(any_conn)

    d_chem = pair_distances(coords, chem_s, chem_t)
    d_elec = pair_distances(coords, elec_s, elec_t)
    d_any = pair_distances(coords, any_s, any_t)

    print("\n接続ニューロン間の距離:")
    print_stats("all connections", d_any)
    print_stats("chemical", d_chem)
    print_stats("electrical", d_elec)

    # --- 距離分布のプロット ------------------------------------------------
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 共通ビンで比較できるようにする
    dist_max = d_any.max() if len(d_any) else 1.0
    bins = np.linspace(0, dist_max, 30)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (data, label, color) in zip(
        axes,
        [
            (d_any, "All connections", "#555555"),
            (d_chem, "Chemical (mask=1,2)", "#1f77b4"),
            (d_elec, "Electrical (mask=-1,2)", "#d62728"),
        ],
    ):
        ax.hist(data, bins=bins, color=color, alpha=0.8, edgecolor="white")
        if len(data):
            ax.axvline(np.mean(data), color="k", linestyle="--", linewidth=1,
                       label=f"mean={np.mean(data):.1f}")
            ax.legend()
        ax.set_title(f"{label}\n(n={len(data)})")
        ax.set_xlabel("Euclidean distance")
        ax.set_ylabel("Count")
    fig.suptitle("C. elegans: distance between connected neurons")
    fig.tight_layout()
    dist_path = output_dir / "c_elegans_distance_dist.png"
    fig.savefig(dist_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {dist_path}")

    # --- 重み分布のプロット ------------------------------------------------
    w_chem = weight_chem[weight_chem != 0]
    w_elec = weight_elec[weight_elec != 0]

    print("\nシナプス重み (非ゼロ要素):")
    print_stats("chemical weight", w_chem)
    print_stats("electrical weight", w_elec)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (data, label, color) in zip(
        axes,
        [
            (w_chem, "Chemical weight", "#1f77b4"),
            (w_elec, "Electrical weight", "#d62728"),
        ],
    ):
        if len(data):
            wmax = data.max()
            wbins = np.arange(0, wmax + 2) - 0.5  # 整数値向けの離散ビン
            ax.hist(data, bins=wbins, color=color, alpha=0.8, edgecolor="white")
            ax.axvline(np.mean(data), color="k", linestyle="--", linewidth=1,
                       label=f"mean={np.mean(data):.2f}")
            ax.legend()
        ax.set_title(f"{label}\n(n={len(data)})")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")
    fig.suptitle("C. elegans: synaptic weight distribution (nonzero)")
    fig.tight_layout()
    weight_path = output_dir / "c_elegans_weight_dist.png"
    fig.savefig(weight_path, dpi=150)
    plt.close(fig)
    print(f"Saved {weight_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
