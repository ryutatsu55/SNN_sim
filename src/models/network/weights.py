import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from src.core.registry import WEIGHT_MODELS

class BaseWeight(ABC):
    """シナプスの重みを生成する基底クラス"""
    def __init__(self, config: Dict[str, Any], num_neurons: int, coords: Optional[np.ndarray], mask: np.ndarray, rng: np.random.RandomState):
        self.config = config
        self.num_neurons = num_neurons
        self.coords = coords
        self.mask = mask
        self.rng = rng

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: 形状 (num_neurons, num_neurons) の重み行列 (np.float32)
        """
        pass

@WEIGHT_MODELS.register("constant")
class ConstantWeight(BaseWeight):
    def generate(self):
        """結合がある箇所すべてに同じ重みを割り当てる"""
        val = self.config.base_weight
        weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        weights[self.mask != 0] = val
        return weights

@WEIGHT_MODELS.register("normal_random")
class NormalRandomWeight(BaseWeight):
    def generate(self):
        """結合がある箇所に、正規分布に従うランダムな重みを割り当てる"""
        mean = self.config.get("mean", 1.0)
        std = self.config.get("std", 0.2)
        
        weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        idx = (self.mask != 0)
        num_conns = np.sum(idx)
        
        weights[idx] = self.rng.normal(mean, std, size=num_conns)
        return weights

@WEIGHT_MODELS.register("module_based")
class ModuleBasedWeight(BaseWeight):
    def generate(self):
        """モジュールベースで有効な結合に対してランダムな重みを割り当てる。"""
        offset = self.config.offset
        g_scale = self.config.g_scale

        weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        idx = self.mask != 0
        num_conns = np.sum(idx)

        if num_conns == 0:
            return weights

        weights[idx] = offset + (g_scale * self.rng.randn(num_conns))
        return weights
