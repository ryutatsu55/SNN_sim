import numpy as np
from scipy.spatial.distance import pdist, squareform
from src.core.registry import CONNECTION_MODELS

@CONNECTION_MODELS.register("constant_prob")
class RandomProbabilityTopology:
    def __init__(self, config: dict, coords: np.ndarray, rng: np.random.RandomState):
        self.N = len(coords)
        self.rng = rng
        self.prob = config.get("connection", {}).get("prob", 0.2)

    def generate(self):
        """空間配置を無視し、純粋な確率で結合マスクを生成"""
        mask = self.rng.random((self.N, self.N)) < self.prob
        np.fill_diagonal(mask, 0) # 自己結合（自分自身へのシナプス）を排除
        return mask.astype(np.int8)

@CONNECTION_MODELS.register("distance_based")
class DistanceBasedTopology:
    def __init__(self, config: dict, num_neurons: int, coords: np.ndarray, rng: np.random.RandomState):
        self.N = num_neurons
        self.coords = coords
        self.rng = rng
        
        topo_cfg = config.get("connection", {})
        self.sigma = topo_cfg.get("sigma", 2.0)
        self.max_prob = topo_cfg.get("max_prob", 0.5)

    def generate(self):
        """空間座標間の距離に基づき、ガウス分布で減衰する確率結合を生成"""
        if self.coords is None:
            raise ValueError("DistanceBasedTopology requires spatial coordinates (coords cannot be None).")
        
        # N x N の距離行列を一括計算
        dist_matrix = squareform(pdist(self.coords))
        
        # 距離に応じた結合確率を計算 (近いほど max_prob に近づく)
        prob_matrix = self.max_prob * np.exp(-(dist_matrix**2) / (2 * self.sigma**2))
        
        # 確率行列に従ってマスクを生成
        mask = self.rng.random((self.N, self.N)) < prob_matrix
        np.fill_diagonal(mask, 0)
        return mask.astype(np.int8)