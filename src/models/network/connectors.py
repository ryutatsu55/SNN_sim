import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from scipy.spatial.distance import pdist, squareform
from src.core.registry import CONNECTION_MODELS

class BaseConnection(ABC):
    """シナプス結合の有無(マスク)を決定する基底クラス"""
    def __init__(self, config: Dict[str, Any], num_neurons: int, coords: Optional[np.ndarray], rng: np.random.RandomState):
        self.config = config
        self.num_neurons = num_neurons
        self.coords = coords
        self.rng = rng

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: 形状 (num_neurons, num_neurons) の結合マスク (0 or 1の np.int8 配列など)
        """
        pass

@CONNECTION_MODELS.register("constant_prob")
class ConstantProbabilityTopology(BaseConnection):
    def generate(self):
        """空間配置を無視し、純粋な確率で結合マスクを生成"""
        prob = self.config.p_out
        
        mask = self.rng.random((self.num_neurons, self.num_neurons)) < prob
        np.fill_diagonal(mask, 0) # 自己結合（自分自身へのシナプス）を排除
        return mask.astype(np.int8)

@CONNECTION_MODELS.register("distance_based")
class DistanceBasedTopology(BaseConnection):
    def generate(self):
        """空間座標間の距離に基づき、ガウス分布で減衰する確率結合を生成"""
        if self.coords is None:
            raise ValueError("DistanceBasedTopology requires spatial coordinates (coords cannot be None).")
        
        topo_cfg = self.config.get("connection", {})
        sigma = topo_cfg.get("sigma", 2.0)
        max_prob = topo_cfg.get("max_prob", 0.5)

        # N x N の距離行列を一括計算
        dist_matrix = squareform(pdist(self.coords))
        
        # 距離に応じた結合確率を計算
        prob_matrix = max_prob * np.exp(-(dist_matrix**2) / (2 * sigma**2))
        
        # 確率に基づいてマスクを生成
        mask = self.rng.random((self.num_neurons, self.num_neurons)) < prob_matrix
        np.fill_diagonal(mask, 0)
        
        return mask.astype(np.int8)

@CONNECTION_MODELS.register("module_based")
class ModuleBasedTopology(BaseConnection):
    def generate(self):
        """ニューロンをモジュール（クラスター）に分割し、モジュール内は高確率、モジュール間は低確率で結合するマスクを生成"""
        num_modules = self.config.num_modules
        within_module_connection_prob = self.config.within_module_connection_prob
        between_module_connection_prob = self.config.between_module_connection_prob

        mask = np.zeros((self.num_neurons, self.num_neurons), dtype=np.int8)

        # ニューロンをモジュールに分割するためのインデックス範囲を計算
        module_ranges: list[tuple[int, int]] = []
        for module_idx in range(num_modules):
            start = int(module_idx * self.num_neurons / num_modules)
            end = int((module_idx + 1) * self.num_neurons / num_modules)
            module_ranges.append((start, end))

        # モジュール内の結合を生成
        for start, end in module_ranges:
            block_shape = (end - start, end - start)
            block_mask = self.rng.rand(*block_shape) < within_module_connection_prob
            mask[start:end, start:end] = block_mask.astype(np.int8)

        # モジュール間の結合を生成
        for module_idx in range(num_modules):
            current_module_start, current_module_end = module_ranges[module_idx]
            # module_idx + 1 == num_modulesのとき、IndexErrorになる。よって、%演算子を使うことで、module_idx + 1 == num_modulesのときは、0に戻るようにする。
            next_module_start, next_module_end = module_ranges[(module_idx + 1) % num_modules]

            forward_block_shape = (
                current_module_end - current_module_start,
                next_module_end - next_module_start,
            )
            backward_block_shape = (
                next_module_end - next_module_start,
                current_module_end - current_module_start,
            )

            # 各モジュールを隣接モジュールと双方向に接続する。
            forward_mask = self.rng.rand(*forward_block_shape) < between_module_connection_prob
            backward_mask = self.rng.rand(*backward_block_shape) < between_module_connection_prob

            mask[current_module_start:current_module_end, next_module_start:next_module_end] = forward_mask.astype(np.int8)
            mask[next_module_start:next_module_end, current_module_start:current_module_end] = backward_mask.astype(np.int8)

        return mask
