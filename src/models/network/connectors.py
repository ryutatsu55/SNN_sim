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
        prob = self.config.p
        
        mask = self.rng.random((self.num_neurons, self.num_neurons)) < prob
        if not self.config.allow_self_connections:
            np.fill_diagonal(mask, 0) # 自己結合（自分自身へのシナプス）を排除
        return mask.astype(np.int8)

@CONNECTION_MODELS.register("optional_connect")
class OptionalConnection(BaseConnection):
    def generate(self):
        """任意の結合を指定する"""        
        mask = np.zeros((self.num_neurons, self.num_neurons))
        mask[self.config.src_ID, self.config.tgt_ID] = 1
        return mask.astype(np.int8)

@CONNECTION_MODELS.register("distance_based")
class DistanceBasedTopology(BaseConnection):
    def generate(self):
        """空間座標間の距離に基づき、ガウス分布で減衰する確率結合を生成"""
        if self.coords is None:
            raise ValueError("DistanceBasedTopology requires spatial coordinates (coords cannot be None).")

        # N x N の距離行列を一括計算
        dist_matrix = squareform(pdist(self.coords))
        
        # 距離に応じた結合確率を計算
        prob_matrix = self.config.max_prob * np.exp(-(dist_matrix**2) / (2 * self.config.sigma**2))
        
        # 確率に基づいてマスクを生成
        mask = self.rng.random((self.num_neurons, self.num_neurons)) < prob_matrix
        
        return mask.astype(np.int8)

@CONNECTION_MODELS.register("prob_based_block")
class BlockRandomTopology(BaseConnection):
    def generate(self):
        """ブロック分割ベースの確率結合を生成し、num_modules=1 では単一ブロックのランダム結合として振る舞う"""
        num_modules = self.config.num_modules
        if not isinstance(num_modules, int) or isinstance(num_modules, bool):
            raise ValueError("num_modules must be an integer.")
        if num_modules < 1:
            raise ValueError("num_modules must be at least 1.")
        if num_modules > self.num_neurons:
            raise ValueError("num_modules must not exceed num_neurons.")

        within_module_connection_prob = self.config.within_module_connection_prob
        between_module_connection_prob = self.config.between_module_connection_prob
        allow_self_connections = self.config.allow_self_connections

        for prob_name, prob_value in (
            ("within_module_connection_prob", within_module_connection_prob),
            ("between_module_connection_prob", between_module_connection_prob),
        ):
            if not isinstance(prob_value, (int, float)) or isinstance(prob_value, bool):
                raise ValueError(f"{prob_name} must be a real number.")
            if not 0.0 <= prob_value <= 1.0:
                raise ValueError(f"{prob_name} must be between 0.0 and 1.0.")

        if not isinstance(allow_self_connections, bool):
            raise ValueError("allow_self_connections must be a boolean.")

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

        if not allow_self_connections:
            np.fill_diagonal(mask, 0)

        return mask
