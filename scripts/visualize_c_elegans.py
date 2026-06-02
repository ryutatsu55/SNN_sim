#!/usr/bin/env python
"""
C. elegansネットワークの可視化スクリプト

使用方法:
    python scripts/visualize_c_elegans.py
    # output/c_elegans_network_xy.png と output/c_elegans_network_3d.png を生成
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.utils.visualize.visualize import c_elegans_network


def main():
    # データファイルのパス
    data_dir = Path("src/models/network/data/c_elegans")

    # データロード
    print("Loading C. elegans data...")
    df = pd.read_csv(data_dir / "ordered_coords.csv")
    mask = np.loadtxt(data_dir / "synapse_mask.csv", delimiter=",")
    weight_chem = np.loadtxt(data_dir / "weight_matrix_chem.csv", delimiter=",")
    weight_elec = np.loadtxt(data_dir / "weight_matrix_elec.csv", delimiter=",")

    # データの前処理
    coords = df[['X', 'Y', 'Z']].values.astype(np.float32)
    neuron_names = df['Neuron'].tolist()
    layers = df['Layer'].tolist()

    print(f"Loaded {len(neuron_names)} neurons")
    print(f"Connectivity matrix shape: {mask.shape}")
    print(f"Chemical synapses: {np.count_nonzero(weight_chem)} connections")
    print(f"Electrical synapses: {np.count_nonzero(weight_elec)} connections")

    # 出力ディレクトリ作成
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 2D可視化
    print("\nGenerating 2D visualization (XY plane)...")
    c_elegans_network(
        coords, neuron_names, layers, mask, weight_chem, weight_elec,
        mode="xy", title="c_elegans_network", save_path=str(output_dir)
    )

    # 3D可視化
    print("\nGenerating 3D visualization...")
    c_elegans_network(
        coords, neuron_names, layers, mask, weight_chem, weight_elec,
        mode="3d", title="c_elegans_network", save_path=str(output_dir)
    )

    print(f"\nVisualization complete! Check {output_dir}/ for output images.")


if __name__ == "__main__":
    main()
