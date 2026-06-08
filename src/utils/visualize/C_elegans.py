import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D


def c_elegans_network(coords: np.ndarray, neuron_names: list, layers: list,
                      mask: np.ndarray, weight_chem: np.ndarray, weight_elec: np.ndarray,
                      mode: str = "xy", title: str = "C.elegans_network", save_path: str = "."):
    """
    C. elegansのニューロンネットワークを可視化する。

    Parameters:
        coords (np.ndarray): ニューロンの座標配列、形状 (N, 3) [X, Y, Z]
        neuron_names (list): ニューロン名のリスト
        layers (list): レイヤー分類のリスト
        mask (np.ndarray): synapse_mask 行列 (N, N)
            0: 接続なし, 1: 化学のみ, -1: 電気のみ, 2: 両方
        weight_chem (np.ndarray): 化学シナプス重み行列 (N, N)
        weight_elec (np.ndarray): 電気シナプス重み行列 (N, N)
        mode (str): "xy" for 2D or "3d" for 3D visualization
        title (str): グラフのタイトル
        save_path (str): 保存先ディレクトリ
    """
    os.makedirs(save_path, exist_ok=True)

    layer_names = sorted(set(layers))
    layer_colors = plt.cm.tab10(np.linspace(0, 1, len(layer_names)))
    layer_to_color = {layer: layer_colors[i] for i, layer in enumerate(layer_names)}
    neuron_colors = np.array([layer_to_color[layer] for layer in layers])

    max_weight_chem = np.max(weight_chem) if np.max(weight_chem) > 0 else 1.0
    max_weight_elec = np.max(weight_elec) if np.max(weight_elec) > 0 else 1.0

    _save_legend(layer_names, layer_colors, title, save_path)

    if mode == "xy":
        _c_elegans_network_2d(coords, neuron_colors, mask, weight_chem, weight_elec,
                              max_weight_chem, max_weight_elec, title, save_path)
    elif mode == "3d":
        _c_elegans_network_3d(coords, neuron_colors, mask, weight_chem, weight_elec,
                              max_weight_chem, max_weight_elec, title, save_path)
    else:
        raise ValueError(f"mode must be 'xy' or '3d', got {mode}")


def _save_legend(layer_names, layer_colors, title, save_path):
    """凡例を単独の画像として保存する"""
    legend_elements = [
        mpatches.Patch(color=(0.9, 0.2, 0.2, 0.5), label='Chemical'),
        mpatches.Patch(color=(0.2, 0.8, 0.2, 0.5), label='Electrical (Gap Junction)'),
    ]
    for layer_name, color in zip(layer_names, layer_colors):
        legend_elements.append(mpatches.Patch(color=color, label=f'Layer: {layer_name}'))

    fig, ax = plt.subplots(figsize=(4, 0.5 * len(legend_elements) + 0.5))
    ax.axis('off')
    ax.legend(handles=legend_elements, loc='center', fontsize=21, frameon=True)

    fig.tight_layout()
    output_path = os.path.join(save_path, f"{title}_legend.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"C. elegans legend saved to {output_path}")
    plt.close()


def _c_elegans_network_2d(coords, neuron_colors, mask, weight_chem, weight_elec,
                          max_weight_chem, max_weight_elec, title, save_path):
    """2D (XY平面) ネットワーク可視化"""
    x = coords[:, 0]
    y = coords[:, 1]

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.scatter(x, y, s=100, c=neuron_colors, edgecolors='black', linewidth=1.5, zorder=3)

    node_margin = 6

    # 化学シナプス（矢印付き）
    sources, targets = np.where((mask == 1) | (mask == 2))
    for s, t in zip(sources, targets):
        w = weight_chem[s, t]
        if w != 0:
            lw = (w / max_weight_chem) * 2.0 + 0.3
            alpha = (w / max_weight_chem) * 0.7 + 0.2
            color = (0.9, 0.2, 0.2, alpha)

            ax.annotate(
                "",
                xy=(x[t], y[t]),
                xytext=(x[s], y[s]),
                arrowprops=dict(
                    arrowstyle="->, head_length=0.4, head_width=0.3",
                    color=color,
                    linewidth=lw,
                    shrinkA=node_margin,
                    shrinkB=node_margin,
                    connectionstyle="arc3,rad=0.05"
                ),
                zorder=1
            )

    # 電気シナプス（破線、矢印なし）
    sources, targets = np.where((mask == -1) | (mask == 2))
    for s, t in zip(sources, targets):
        w = weight_elec[s, t]
        if w != 0:
            lw = (w / max_weight_elec) * 2.0 + 0.3
            alpha = (w / max_weight_elec) * 0.7 + 0.2
            color = (0.2, 0.8, 0.2, alpha)

            ax.plot(
                [x[s], x[t]],
                [y[s], y[t]],
                linestyle='--',
                color=color,
                linewidth=lw,
                zorder=1
            )

    ax.set_title(f"{title} (XY Plane)", fontsize=27, fontweight='bold')
    ax.set_xlabel("X Coordinate [μm]", fontsize=24)
    ax.set_ylabel("Y Coordinate [μm]", fontsize=24)
    ax.tick_params(labelsize=21)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(save_path, f"{title}_xy.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"C. elegans network (2D) visualization saved to {output_path}")
    plt.close()


def _c_elegans_network_3d(coords, neuron_colors, mask, weight_chem, weight_elec,
                          max_weight_chem, max_weight_elec, title, save_path):
    """3D ネットワーク可視化"""
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, s=100, c=neuron_colors, edgecolors='black', linewidth=1.5, zorder=3)

    # 化学シナプス
    sources, targets = np.where((mask == 1) | (mask == 2))
    for s, t in zip(sources, targets):
        w = weight_chem[s, t]
        if w != 0:
            lw = (w / max_weight_chem) * 2.0 + 0.3
            alpha = (w / max_weight_chem) * 0.7 + 0.2
            color = (0.9, 0.2, 0.2, alpha)

            ax.plot(
                [x[s], x[t]],
                [y[s], y[t]],
                [z[s], z[t]],
                color=color,
                linewidth=lw,
                zorder=1
            )

    # 電気シナプス（破線）
    sources, targets = np.where((mask == -1) | (mask == 2))
    for s, t in zip(sources, targets):
        w = weight_elec[s, t]
        if w != 0:
            lw = (w / max_weight_elec) * 2.0 + 0.3
            alpha = (w / max_weight_elec) * 0.7 + 0.2
            color = (0.2, 0.8, 0.2, alpha)

            ax.plot(
                [x[s], x[t]],
                [y[s], y[t]],
                [z[s], z[t]],
                linestyle='--',
                color=color,
                linewidth=lw,
                zorder=1
            )

    ax.set_title(f"{title} (3D)", fontsize=27, fontweight='bold')
    ax.set_xlabel("X [μm]", fontsize=24, labelpad=25)
    ax.set_ylabel("Y [μm]", fontsize=24, labelpad=25)
    ax.set_zlabel("Z [μm]", fontsize=24, labelpad=25)
    ax.tick_params(labelsize=18)

    plt.tight_layout()
    output_path = os.path.join(save_path, f"{title}_3d.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"C. elegans network (3D) visualization saved to {output_path}")
    plt.close()
