import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from src.core.registry import SPATIAL_MODELS

class BaseSpace(ABC):
    """空間座標を生成する基底クラス"""
    def __init__(self, config: Dict[str, Any], num_neurons: int, rng: np.random.RandomState):
        self.config = config
        self.num_neurons = num_neurons
        self.rng = rng

    @abstractmethod
    def generate(self) -> Optional[np.ndarray]:
        """
        Returns:
            np.ndarray: 形状 (num_neurons, D) の座標配列
        """
        pass

@SPATIAL_MODELS.register("no_space")
class NoSpace(BaseSpace):
    def generate(self):
        """空間配置を必要としない場合は None を返す"""
        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)
        coords[:,:] = np.nan

        return coords

@SPATIAL_MODELS.register("grid_3d")
class Grid3DSpace(BaseSpace):
    def generate(self):
        """指定された範囲内に一様乱数で3次元座標を生成する"""
        spatial_cfg = self.config.get("space", {})
        x_range = spatial_cfg.get("x_range", [0.0, 10.0])
        y_range = spatial_cfg.get("y_range", [0.0, 10.0])
        z_range = spatial_cfg.get("z_range", [0.0, 10.0])

        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)
        coords[:, 0] = self.rng.uniform(x_range[0], x_range[1], self.num_neurons)
        coords[:, 1] = self.rng.uniform(y_range[0], y_range[1], self.num_neurons)
        coords[:, 2] = self.rng.uniform(z_range[0], z_range[1], self.num_neurons)
        
        return coords