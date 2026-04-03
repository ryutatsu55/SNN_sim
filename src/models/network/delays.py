import numpy as np
from scipy.spatial.distance import pdist, squareform
from src.core.registry import DELAY_MODELS

@DELAY_MODELS.register("constant")
class ConstantDelay:
    def __init__(self, config: dict, coords: np.ndarray, mask: np.ndarray, rng: np.random.RandomState):
        self.N = len(coords)
        self.val = config.get("delay", {}).get("value", 1.0)
        self.mask = mask

    def generate(self):
        """全シナプスで共通の伝播遅延"""
        delays = np.zeros((self.N, self.N), dtype=np.float32)
        delays[self.mask != 0] = self.val
        return delays

@DELAY_MODELS.register("distance_based")
class DistanceBasedDelay:
    def __init__(self, config: dict, num_neurons: int, coords: np.ndarray, rng: np.random.RandomState):
        self.N = num_neurons
        self.coords = coords
        
        d_cfg = config.get("delay", {})
        self.velocity = d_cfg.get("velocity", 1.0) # 伝播速度
        self.min_delay = d_cfg.get("min_delay", 0.1) # 最小遅延 (シナプス間隙の遅延など)

    def generate(self, mask: np.ndarray):
        """物理的な距離を伝播速度で割って遅延を決定する"""
        if self.coords is None:
            raise ValueError("DistanceBasedDelay requires spatial coordinates.")
            
        dist_matrix = squareform(pdist(self.coords))
        delays = np.zeros((self.N, self.N), dtype=np.float32)
        
        idx = (mask != 0)
        # 距離に応じた遅延時間 (Distance / Velocity + MinDelay)
        delays[idx] = self.min_delay + (dist_matrix[idx] / self.velocity)
        return delays