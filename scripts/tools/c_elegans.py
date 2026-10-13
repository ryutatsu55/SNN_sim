"""C. elegans のコネクトームを可視化する道具。

    python -m scripts.tools.c_elegans -o output/c_elegans

**実験ではなく道具。** シミュレーションを回さず、`src/models/network/data/c_elegans/` の
CSV をそのまま描くだけなので、run ディレクトリも記録窓も存在しない。したがって
`scripts/develop/` の 5 層 (report / store / analysis / figures) は敷かない —— あれは
「記録窓ごとの図と run 全体の図が両方出る」実験のための形で、ここには当てはまらない。

出す図は 4 枚:

    c_elegans_network_xy.png        細胞体の 2D 配置と結合 (化学=赤の矢印 / 電気=緑の破線)
    c_elegans_network_3d.png        同じものを 3D で
    c_elegans_network_legend.png    凡例だけの図 (本体に載せると経路を隠すため)
    c_elegans_distance_dist.png     結合しているニューロン対の距離分布 (全体 / 化学 / 電気)
    c_elegans_weight_dist.png       化学・電気それぞれの重み分布 (非ゼロ要素)

`synapse_mask.csv` の値の意味:

    0: 接続なし / 1: 化学のみ / -1: 電気のみ / 2: 化学+電気の両方

保存先は `out_path` で受ける (`src/utils/plotting/` と同じ約束)。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D 投影の登録)

from src.utils.plotting.common import save_figure

# データの置き場所。モデル側が読むのと同じ CSV を見る。
DATA_DIR = Path("src/models/network/data/c_elegans")

# 化学 / 電気の描画色。透明度は重みで変えるので、ここでは RGB だけ持つ。
CHEM_RGB = (0.9, 0.2, 0.2)
ELEC_RGB = (0.2, 0.8, 0.2)
NODE_SIZE = 100
NODE_MARGIN = 6


# ======================================================================================
# 読み出し
# ======================================================================================

class Connectome:
    """CSV 一式を読んだもの。**この道具にとっての「リーダー」。**

    実験の `store/` と同じ役割 (どこに何があり、どう読むか) だが、run ディレクトリでは
    なくリポジトリ同梱の CSV を読む。`src/utils/runview.py` の契約は敷かない ——
    記録窓も config も無いので、契約の形が 1 つも埋まらない。
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        data_dir = Path(data_dir)
        frame = pd.read_csv(data_dir / "ordered_coords.csv")
        self.coords = frame[["X", "Y", "Z"]].values.astype(np.float64)
        self.names = frame["Neuron"].tolist()
        self.layers = frame["Layer"].tolist()
        self.mask = np.loadtxt(data_dir / "synapse_mask.csv", delimiter=",").astype(int)
        self.weight_chem = np.loadtxt(data_dir / "weight_matrix_chem.csv", delimiter=",")
        self.weight_elec = np.loadtxt(data_dir / "weight_matrix_elec.csv", delimiter=",")

    @property
    def num_neurons(self) -> int:
        return len(self.names)

    def layer_colors(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """(層名, 層ごとの色, ニューロンごとの色) を返す。"""
        layer_names = sorted(set(self.layers))
        colors = plt.cm.tab10(np.linspace(0, 1, len(layer_names)))
        of_layer = {name: colors[i] for i, name in enumerate(layer_names)}
        return layer_names, colors, np.array([of_layer[layer] for layer in self.layers])

    def pairs(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """`kind` ("chemical" / "electrical" / "any") の結合対を (source, target) で返す。

        **自己結合は除く** (距離 0 が分布の先頭に立って形を壊すため)。
        """
        no_self = ~np.eye(self.num_neurons, dtype=bool)
        if kind == "chemical":
            selected = (self.mask == 1) | (self.mask == 2)
        elif kind == "electrical":
            selected = (self.mask == -1) | (self.mask == 2)
        elif kind == "any":
            selected = self.mask != 0
        else:
            raise ValueError(f"未知の結合種別: {kind!r}")
        return np.where(selected & no_self)

    def distances(self, kind: str) -> np.ndarray:
        """`kind` の結合を持つ対のユークリッド距離。"""
        source, target = self.pairs(kind)
        return np.linalg.norm(self.coords[source] - self.coords[target], axis=1)


# ======================================================================================
# 図
# ======================================================================================

def _edge_style(weight: float, max_weight: float, rgb: tuple[float, float, float]):
    """重みから線幅と色 (透明度込み) を決める。化学も電気も同じ換算を使う。"""
    ratio = weight / max_weight
    return ratio * 2.0 + 0.3, (*rgb, ratio * 0.7 + 0.2)


def network_xy(connectome: Connectome, out_path: Path) -> None:
    """細胞体の XY 配置と結合を描く。化学は矢印、電気は破線 (向きが無いため)。"""
    _, _, node_colors = connectome.layer_colors()
    x, y = connectome.coords[:, 0], connectome.coords[:, 1]
    max_chem = max(connectome.weight_chem.max(), 1.0)
    max_elec = max(connectome.weight_elec.max(), 1.0)

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(x, y, s=NODE_SIZE, c=node_colors, edgecolors="black", linewidth=1.5, zorder=3)

    for source, target in zip(*connectome.pairs("chemical")):
        weight = connectome.weight_chem[source, target]
        if weight == 0:
            continue
        width, color = _edge_style(weight, max_chem, CHEM_RGB)
        ax.annotate(
            "", xy=(x[target], y[target]), xytext=(x[source], y[source]),
            arrowprops=dict(arrowstyle="->, head_length=0.4, head_width=0.3",
                            color=color, linewidth=width,
                            shrinkA=NODE_MARGIN, shrinkB=NODE_MARGIN,
                            connectionstyle="arc3,rad=0.05"),
            zorder=1)

    for source, target in zip(*connectome.pairs("electrical")):
        weight = connectome.weight_elec[source, target]
        if weight == 0:
            continue
        width, color = _edge_style(weight, max_elec, ELEC_RGB)
        ax.plot([x[source], x[target]], [y[source], y[target]],
                linestyle="--", color=color, linewidth=width, zorder=1)

    ax.set_title("C. elegans network (XY plane)", fontsize=27, fontweight="bold")
    ax.set_xlabel("X Coordinate [um]", fontsize=24)
    ax.set_ylabel("Y Coordinate [um]", fontsize=24)
    ax.tick_params(labelsize=21)
    ax.grid(True, alpha=0.3)
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")


def network_3d(connectome: Connectome, out_path: Path) -> None:
    """同じネットワークを 3D で。矢印は使えないので化学も直線で描く。"""
    _, _, node_colors = connectome.layer_colors()
    x, y, z = connectome.coords[:, 0], connectome.coords[:, 1], connectome.coords[:, 2]
    max_chem = max(connectome.weight_chem.max(), 1.0)
    max_elec = max(connectome.weight_elec.max(), 1.0)

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, s=NODE_SIZE, c=node_colors, edgecolors="black", linewidth=1.5)

    for kind, weights, max_weight, rgb, style in (
        ("chemical", connectome.weight_chem, max_chem, CHEM_RGB, "-"),
        ("electrical", connectome.weight_elec, max_elec, ELEC_RGB, "--"),
    ):
        for source, target in zip(*connectome.pairs(kind)):
            weight = weights[source, target]
            if weight == 0:
                continue
            width, color = _edge_style(weight, max_weight, rgb)
            ax.plot([x[source], x[target]], [y[source], y[target]], [z[source], z[target]],
                    linestyle=style, color=color, linewidth=width, zorder=1)

    ax.set_title("C. elegans network (3D)", fontsize=27, fontweight="bold")
    ax.set_xlabel("X [um]", fontsize=24, labelpad=25)
    ax.set_ylabel("Y [um]", fontsize=24, labelpad=25)
    ax.set_zlabel("Z [um]", fontsize=24, labelpad=25)
    ax.tick_params(labelsize=18)
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")


def network_legend(connectome: Connectome, out_path: Path) -> None:
    """凡例だけの図。**本体に載せると結合の経路を隠す**ので別ファイルにしてある

    (`src/utils/plotting/network.py` の `axon_network` と同じ理由)。
    """
    layer_names, layer_colors, _ = connectome.layer_colors()
    handles = [mpatches.Patch(color=(*CHEM_RGB, 0.5), label="Chemical"),
               mpatches.Patch(color=(*ELEC_RGB, 0.5), label="Electrical (gap junction)")]
    handles += [mpatches.Patch(color=color, label=f"Layer: {name}")
                for name, color in zip(layer_names, layer_colors)]

    fig, ax = plt.subplots(figsize=(4, 0.5 * len(handles) + 0.5))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", fontsize=21, frameon=True)
    save_figure(fig, out_path, dpi=300, bbox_inches="tight")


def distance_distribution(connectome: Connectome, out_path: Path) -> None:
    """結合しているニューロン対の距離分布 (全体 / 化学 / 電気)。**ビンは 3 枚で共通**。

    共通ビンにするのは、3 枚を並べて「化学の方が近距離に寄っているか」を読むため。
    パネルごとに自動ビンにすると横軸が揃わず、比較の意味が消える。
    """
    series = [("All connections", connectome.distances("any"), "#555555"),
              ("Chemical (mask=1,2)", connectome.distances("chemical"), "#1f77b4"),
              ("Electrical (mask=-1,2)", connectome.distances("electrical"), "#d62728")]
    longest = series[0][1].max() if series[0][1].size else 1.0
    bins = np.linspace(0, longest, 30)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (label, values, color) in zip(axes, series):
        ax.hist(values, bins=bins, color=color, alpha=0.8, edgecolor="white")
        if values.size:
            ax.axvline(values.mean(), color="k", linestyle="--", linewidth=1,
                       label=f"mean={values.mean():.1f}")
            ax.legend()
        ax.set_title(f"{label}\n(n={values.size})")
        ax.set_xlabel("Euclidean distance [um]")
        ax.set_ylabel("Count")
    fig.suptitle("C. elegans: distance between connected neurons")
    save_figure(fig, out_path, dpi=150)


def weight_distribution(connectome: Connectome, out_path: Path) -> None:
    """化学・電気それぞれの重み分布 (非ゼロ要素のみ)。

    重みは接触本数の整数なので、ビンは整数中心の離散ビンにする (連続ビンだと隣り合う
    整数が 1 本の棒にまとまって、1 本と 2 本の差が読めなくなる)。
    """
    series = [("Chemical weight", connectome.weight_chem, "#1f77b4"),
              ("Electrical weight", connectome.weight_elec, "#d62728")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (label, matrix, color) in zip(axes, series):
        values = matrix[matrix != 0]
        if values.size:
            bins = np.arange(0, values.max() + 2) - 0.5
            ax.hist(values, bins=bins, color=color, alpha=0.8, edgecolor="white")
            ax.axvline(values.mean(), color="k", linestyle="--", linewidth=1,
                       label=f"mean={values.mean():.2f}")
            ax.legend()
        ax.set_title(f"{label}\n(n={values.size})")
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")
    fig.suptitle("C. elegans: synaptic weight distribution (nonzero)")
    save_figure(fig, out_path, dpi=150)


# ======================================================================================
# CLI
# ======================================================================================

FIGURES = (
    ("c_elegans_network_xy.png", network_xy),
    ("c_elegans_network_3d.png", network_3d),
    ("c_elegans_network_legend.png", network_legend),
    ("c_elegans_distance_dist.png", distance_distribution),
    ("c_elegans_weight_dist.png", weight_distribution),
)


def _describe(connectome: Connectome) -> None:
    """読めたデータの要約を出す。図だけでは本数が読めないので数字でも残す。"""
    print(f"  {connectome.num_neurons} neurons, mask shape {connectome.mask.shape}")
    print(f"  chemical synapses  : {int(np.count_nonzero(connectome.weight_chem))}")
    print(f"  electrical synapses: {int(np.count_nonzero(connectome.weight_elec))}")
    for kind in ("any", "chemical", "electrical"):
        values = connectome.distances(kind)
        if values.size == 0:
            print(f"  distance ({kind}): (要素なし)")
            continue
        print(f"  distance ({kind}): n={values.size}, mean={values.mean():.3f}, "
              f"median={np.median(values):.3f}, std={values.std():.3f}, "
              f"min={values.min():.3f}, max={values.max():.3f}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="C. elegans コネクトームの図を出す。")
    parser.add_argument("-o", "--output", default="output/c_elegans",
                        help="出力先ディレクトリ (既定: output/c_elegans)")
    parser.add_argument("--data-dir", default=str(DATA_DIR),
                        help=f"CSV の置き場所 (既定: {DATA_DIR})")
    args = parser.parse_args(argv)

    # ConfigManager と同じく、データのパスをリポジトリルート基準で解決する。
    os.chdir(_PROJECT_ROOT)
    out_dir = Path(args.output)

    print("Loading C. elegans data...")
    connectome = Connectome(Path(args.data_dir))
    _describe(connectome)

    for name, draw in FIGURES:
        draw(connectome, out_dir / name)
        print(f"  saved {out_dir / name}")
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
