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