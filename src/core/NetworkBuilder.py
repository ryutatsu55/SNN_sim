# src/core/NetworkBuilder.py
import numpy as np
from src.core.registry import TOPOLOGY_MODELS, WEIGHT_MODELS, DELAY_MODELS, SYNAPSE_MODELS, NEURON_MODELS

class NetworkBuilder:
    def __init__(self, network_config: dict, rng: np.random.RandomState):
        """
        network_config: resolved_cfg["network"] の辞書
        """
        self.cfg = network_config
        self.rng = rng
        self.N = self.cfg["total_n"]

    def build(self) -> dict:
        print("=== ネットワーク構築 (Network Building) ===")
        
        # 1. トポロジー(結合有無のマスクと興奮/抑制のタイプ)の生成
        topo_cfg = self.cfg["topology"]
        TopoClass = TOPOLOGY_MODELS.get(topo_cfg["type"])
        topo_builder = TopoClass(topo_cfg, self.N, self.rng)
        mask, neuron_types = topo_builder.generate()
        
        # 2. 重みの生成
        weight_cfg = self.cfg["weight"]
        WeightClass = WEIGHT_MODELS.get(weight_cfg["type"])
        weight_builder = WeightClass(weight_cfg, self.N, self.rng)
        weight_matrix = weight_builder.generate(mask, neuron_types)

        # 3. 伝播遅延の生成
        delay_cfg = self.cfg["delay"]
        DelayClass = DELAY_MODELS.get(delay_cfg["type"])
        delay_builder = DelayClass(delay_cfg, self.N, self.rng)
        delay_matrix = delay_builder.generate(mask)

        # 4. GPU転送用などのフォーマット変換 (CSR/COO相当の一次元化など)
        # ※以前の calc_init に相当する処理をここで一括で行うとクリーンです
        N_S = np.count_nonzero(weight_matrix)
        col_indices, row_indices = np.where(weight_matrix.T != 0)
        
        resovoir_weight_calc = np.zeros(N_S, dtype=np.float32)
        delay_row = np.zeros(N_S, dtype=np.int32)
        neuron_from = np.zeros(N_S, dtype=np.int32)
        neuron_to = np.zeros(N_S, dtype=np.int32)
        
        for i in range(N_S):
            r = row_indices[i]
            c = col_indices[i]
            neuron_from[i] = c
            neuron_to[i] = r
            resovoir_weight_calc[i] = weight_matrix[r, c]
            delay_row[i] = delay_matrix[r, c]

        print(f"  - Total Neurons: {self.N}")
        print(f"  - Total Synapses (N_S): {N_S}")

        # 最終的にSimulator（やGPUカーネル）が必要とする全情報をまとめた辞書を返す
        return {
            "N": self.N,
            "N_S": N_S,
            "neuron_types": neuron_types,
            "weight_matrix": weight_matrix,
            "delay_matrix": delay_matrix,
            "gpu_data": {
                "neuron_from": neuron_from,
                "neuron_to": neuron_to,
                "resovoir_weight_calc": resovoir_weight_calc,
                "delay_row": delay_row,
            }
        }