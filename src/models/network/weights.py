import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from numbers import Real
from scipy.spatial.distance import pdist, squareform
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

@WEIGHT_MODELS.register("normal_broad")
class NormalRandomWeight(BaseWeight):
    def generate(self):
        """結合がある箇所に、正規分布に従うランダムな重みを割り当てる"""
        mean = self.config.mean
        std = self.config.std
        
        weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        idx = (self.mask != 0)
        num_conns = np.sum(idx)
        
        weights[idx] = self.rng.normal(mean, std, size=num_conns)
        return weights

@WEIGHT_MODELS.register("lognormal_broad")
class LogNormalRandomWeight(BaseWeight):
    def generate(self):
        """結合がある箇所に、対数正規分布に従うランダムな重みを割り当てる"""
        mean = self.config.mean
        std = self.config.std
        
        weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        idx = (self.mask != 0)
        num_conns = np.sum(idx)
        
        weights[idx] = self.rng.lognormal(mean, std, size=num_conns)
        return weights

@WEIGHT_MODELS.register("offset_scaled_normal")
class OffsetScaledNormalWeight(BaseWeight):
    def generate(self):
        """有効な結合に対して、offset + g_scale * N(0, 1) の重みを割り当てる。"""
        offset = self.config.offset
        g_scale = self.config.g_scale

        if not isinstance(offset, Real) or isinstance(offset, bool):
            raise ValueError("offset must be a real number.")
        if not isinstance(g_scale, Real) or isinstance(g_scale, bool):
            raise ValueError("g_scale must be a real number.")
        if g_scale < 0.0:
            raise ValueError("g_scale must be greater than or equal to 0.0.")

        weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        idx = self.mask != 0
        num_conns = np.sum(idx)

        if num_conns == 0:
            return weights

        weights[idx] = offset + (g_scale * self.rng.randn(num_conns))
        return weights

@WEIGHT_MODELS.register("distance_dependent")
class DistanceDependentWeight(BaseWeight):
    def generate(self):
        """
        結合がある箇所に、距離に応じた重みを割り当てる。
        近いほど重みが大きく（最大 amplitude）、遠いほど指数関数的に小さくなる。
        """
        # パラメータの取得（設定がない場合の安全なフォールバック値を設定）
        amplitude = self.config.max_weight
        decay_length = self.config.decay_length
        
        # N x N の距離行列を一括計算
        dist_matrix = squareform(pdist(self.coords))

        weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        idx = (self.mask != 0)
        
        # 結合が存在する箇所（マスクが有効な箇所）の距離データを取得
        active_distances = dist_matrix[idx]
        
        # 距離に応じた指数減衰の計算： w = A * exp(-d / λ)
        weights[idx] = amplitude * np.exp(-active_distances / decay_length)
        
        return weights