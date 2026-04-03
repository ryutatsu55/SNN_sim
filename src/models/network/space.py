import numpy as np
from src.core.registry import SPATIAL_MODELS

@SPATIAL_MODELS.register("none")
class NoSpace:
    def __init__(self, config: dict, num_neurons: int, rng: np.random.RandomState):
        pass
    
    def generate(self):
        """空間配置を必要としない場合は None を返す"""
        return None

@SPATIAL_MODELS.register("grid_3d")
class Grid3DSpace:
    def __init__(self, config: dict, num_neurons: int, rng: np.random.RandomState):
        self.N = num_neurons
        self.rng = rng
        # config['spatial'] の中身を想定
        spatial_cfg = config.get("space", {})
        self.x_range = spatial_cfg.get("x_range", [0.0, 10.0])
        self.y_range = spatial_cfg.get("y_range", [0.0, 10.0])
        self.z_range = spatial_cfg.get("z_range", [0.0, 10.0])

    def generate(self):
        """指定された範囲内に一様乱数で3次元座標を生成する (ダミー実装)"""
        coords = np.zeros((self.N, 3), dtype=np.float32)
        coords[:, 0] = self.rng.uniform(self.x_range[0], self.x_range[1], self.N)
        coords[:, 1] = self.rng.uniform(self.y_range[0], self.y_range[1], self.N)
        coords[:, 2] = self.rng.uniform(self.z_range[0], self.z_range[1], self.N)
        return coords