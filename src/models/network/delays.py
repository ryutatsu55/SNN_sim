import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from numbers import Real
from scipy.spatial.distance import pdist, squareform
from src.core.registry import DELAY_MODELS

class BaseDelay(ABC):
    """シナプスの伝播遅延を生成する基底クラス"""
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
            np.ndarray: 形状 (num_neurons, num_neurons) の遅延行列 (np.float32)
        """
        pass

@DELAY_MODELS.register("constant")
class ConstantDelay(BaseDelay):
    def generate(self):
        """全シナプスで共通の伝播遅延"""
        val = self.config.value
        
        delays = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        delays[self.mask != 0] = val
        return delays

@DELAY_MODELS.register("distance_based")
class DistanceBasedDelay(BaseDelay):
    def generate(self):
        """物理的な距離を伝播速度で割って遅延を決定する"""
        if self.coords is None:
            raise ValueError("DistanceBasedDelay requires spatial coordinates (coords cannot be None).")
            
        velocity = self.config.velocity
        min_delay = self.config.min_delay

        # 距離行列の計算
        dist_matrix = squareform(pdist(self.coords))
        
        # 距離 / 速度 で遅延を計算し、最小遅延を足す
        calc_delays = (dist_matrix / velocity) + min_delay
        
        # 結合がない箇所の遅延は0にする（メモリ効率とバグ防止のため）
        delays = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        idx = (self.mask != 0)
        delays[idx] = calc_delays[idx]
        
        return delays

@DELAY_MODELS.register("random")
class RandomDelay(BaseDelay):
    def generate(self) -> np.ndarray:
        """結合がある箇所に正規分布ベースのランダム遅延を割り当てる"""
        mean = self.config.mean
        std = self.config.std
        min_delay = self.config.min
        max_delay = self.config.max

        if not isinstance(mean, Real) or isinstance(mean, bool):
            raise ValueError("mean must be a real number.")
        if not isinstance(std, Real) or isinstance(std, bool):
            raise ValueError("std must be a real number.")
        if std < 0:
            raise ValueError("std must be greater than or equal to 0.0.")

        for bound_name, bound_value in (("min", min_delay), ("max", max_delay)):
            if bound_value is None:
                continue

            if not isinstance(bound_value, Real) or isinstance(bound_value, bool):
                raise ValueError(f"{bound_name} must be a real number or None.")
            if bound_value < 0.0:
                raise ValueError(f"{bound_name} must be greater than or equal to 0.0.")

        if min_delay is None or max_delay is None:
            pass  # 片方がNoneなら比較は不要
        elif min_delay > max_delay:
            raise ValueError("min must be less than or equal to max.")

        delays = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        valid_mask = self.mask != 0
        num_connections = int(np.count_nonzero(valid_mask))

        if num_connections == 0:
            return delays

        sampled_delays = self.rng.normal(float(mean), float(std), size=num_connections).astype(np.float32)

        lower = -np.inf if min_delay is None else float(min_delay)
        upper = np.inf if max_delay is None else float(max_delay)
        sampled_delays = np.clip(sampled_delays, lower, upper)

        delays[valid_mask] = sampled_delays
        return delays
