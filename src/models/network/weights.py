import numpy as np
from src.core.registry import WEIGHT_MODELS

@WEIGHT_MODELS.register("constant")
class ConstantWeight:
    def __init__(self, config: dict, coords: np.ndarray, mask: np.ndarray, rng: np.random.RandomState):
        self.N = len(coords)
        self.val = config.get("weight", {}).get("value", 1.0)
        self.mask = mask

    def generate(self):
        """結合がある箇所すべてに同じ重みを割り当てる"""
        weights = np.zeros((self.N, self.N), dtype=np.float32)
        weights[self.mask != 0] = self.val
        return weights

@WEIGHT_MODELS.register("normal_random")
class NormalRandomWeight:
    def __init__(self, config: dict, num_neurons: int, coords: np.ndarray, rng: np.random.RandomState):
        self.N = num_neurons
        self.rng = rng
        w_cfg = config.get("weight", {})
        self.mean = w_cfg.get("mean", 1.0)
        self.std = w_cfg.get("std", 0.2)

    def generate(self, mask: np.ndarray):
        """結合がある箇所に、正規分布に従うランダムな重みを割り当てる"""
        weights = np.zeros((self.N, self.N), dtype=np.float32)
        
        # 結合が存在するインデックスを抽出
        idx = (mask != 0)
        num_conns = np.sum(idx)
        
        # 負の重みが混ざらないように絶対値をとる (E/Iは後でスケール倍されるため)
        weights[idx] = np.abs(self.rng.normal(self.mean, self.std, num_conns))
        return weights