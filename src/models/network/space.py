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

@SPATIAL_MODELS.register("grid_2d")
class Grid2DSpace(BaseSpace):
    def generate(self):
        """指定された範囲内に一定間隔で2次元座標を生成する"""
        x_range = self.config.x_range
        y_range = self.config.y_range
        
        nx = int(np.ceil(np.sqrt(self.num_neurons)))
        ny = int(np.ceil(self.num_neurons / nx))
        
        # 各軸の一定間隔の座標を生成
        x_steps = np.linspace(x_range[0], x_range[1], nx)
        y_steps = np.linspace(y_range[0], y_range[1], ny)
        
        # メッシュグリッドを作成
        X, Y = np.meshgrid(x_steps, y_steps)
        
        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)
        
        # 生成したグリッド座標を1次元に平坦化し、必要なニューロン数だけ切り出して割り当て
        coords[:, 0] = X.ravel()[:self.num_neurons]
        coords[:, 1] = Y.ravel()[:self.num_neurons]
        
        return coords
    
@SPATIAL_MODELS.register("random_2d")
class Random2DSpace(BaseSpace):
    def generate(self):
        """指定された範囲内に一様乱数で2次元座標を生成する"""

        x_range = self.config.x_range
        y_range = self.config.y_range

        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)
        coords[:, 0] = self.rng.uniform(x_range[0], x_range[1], self.num_neurons)
        coords[:, 1] = self.rng.uniform(y_range[0], y_range[1], self.num_neurons)

        return coords

@SPATIAL_MODELS.register("block_2d")
class Block2DSpace(BaseSpace):
    def generate(self):
        """四角形のモジュール領域を定義し、一様ランダムに配置する"""
        x_range = self.config.x_range
        y_range = self.config.y_range
        
        num_modules = self.config.num_modules
        gap_ratio = self.config.margin
        
        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)
        
        # --- 1. グリッドの分割数を計算 ---
        nx = int(np.ceil(np.sqrt(num_modules)))
        ny = int(np.ceil(num_modules / nx))
        
        # 全体の幅と高さ
        total_w = x_range[1] - x_range[0]
        total_h = y_range[1] - y_range[0]
        
        # 隙間の絶対サイズ
        gap_x = (total_w * gap_ratio)
        gap_y = (total_h * gap_ratio)
        
        # 1つのモジュールの幅と高さ
        module_w = (total_w - gap_x) / nx
        module_h = (total_h - gap_y) / ny
        
        # --- 2. 各モジュールへの割り当てニューロン数を計算（均等分割＋端数処理） ---
        neurons_per_module = np.full(num_modules, self.num_neurons // num_modules)
        # 余りが出た場合、先頭のモジュールから順に1つずつ追加して吸収する
        neurons_per_module[:self.num_neurons % num_modules] += 1
        
        # --- 3. 四角形領域の計算と内部へのランダム散布（連番割り当て） ---
        current_idx = 0
        for i in range(num_modules):
            n = neurons_per_module[i]
                
            # グリッド上のインデックス (ix, iy)
            ix = i % nx
            iy = i // nx
            
            # このモジュールの描画範囲 (四角形の境界) を計算
            min_x = x_range[0] + ix * (module_w + gap_x/nx) + gap_x/(2*nx)
            max_x = min_x + module_w
            
            min_y = y_range[0] + iy * (module_h + gap_y/ny) + gap_y/(2*ny)
            max_y = min_y + module_h
            
            # 連番のインデックス範囲に対して、四角形の範囲内に一様分布でランダムに散布
            coords[current_idx:current_idx+n, 0] = self.rng.uniform(min_x, max_x, size=n)
            coords[current_idx:current_idx+n, 1] = self.rng.uniform(min_y, max_y, size=n)
            
            # 次のモジュールへインデックスを進める
            current_idx += n
            
        return coords