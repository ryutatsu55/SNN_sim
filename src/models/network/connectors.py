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

@CONNECTION_MODELS.register("module_based")
class ModuleBasedTopology(BaseConnection):
    def generate(self):
        """ニューロンをモジュール（クラスター）に分割し、モジュール内は高確率、モジュール間は低確率で結合するマスクを生成"""
        reservoir_weight = np.zeros((self.num_neurons, self.num_neurons))
        # TODO: self.num_neurons / self.config.num_modules も何かしらの変数で置きたい。
        block_size = self.num_neurons // self.config.num_modules

        G = 0.5 # TODO: これも config から取るようにしたい
        p = 0.1 # TODO: これも config から取るようにしたい
        offset = 1.0
        
        crust_idx = 0
        while crust_idx != self.config.num_modules:
            i1 = int(crust_idx * self.num_neurons / self.config.num_modules)
            i2 = int((crust_idx + 1) * self.num_neurons / self.config.num_modules)
            reservoir_weight[i1:i2, i1:i2] = ((G * self.rng.randn(block_size, block_size)) + offset) * (
                self.rng.rand(block_size, block_size) < p
            )
            # reservoir_weight[i1:i2, i1:i2] = (
            #     G*(self.rng.rand(block_size, block_size)-0.5) + offset
            #     ) * (self.rng.rand(block_size, block_size) < p)

            # 1. ニューロンごとに異なる「接続確率」を作成する
            # 対数正規分布を使って「ムラ」を作る (sigmaが大きいほどムラが激しくなる)
            # size=(1, block_size) にすることで「列（前ニューロン）」ごとに確率を変える
            # variability = self.rng.lognormal(mean=0.0, sigma=1.0, size=(1, block_size))
            # # 平均が元の p (0.08) になるように正規化
            # variability = variability / np.mean(variability)
            # p_vec = p * variability
            # # 確率が 1.0 を超えないようにクリップ
            # p_vec = np.clip(p_vec, 0.0, 1.0)
            # # self.rng.rand(self.num_neurons, self.num_neurons) < (1, self.num_neurons) の比較により、ブロードキャスト
            # mask = self.rng.rand(block_size, block_size) < p_vec
            # reservoir_weight[i1:i2, i1:i2] = (
            #     G * (rng.rand(block_size, block_size) - 0.5) + offset
            # ) * mask
            
            crust_idx += 1
        
        # クラスター間の接続
        M = 4
        G = 0.5
        p = 0.01
        offset = 1.0
        # TODO: hogeの名前を変えたい。さすがにダメ。
        for hoge in range(M):
            i_range1 = int((hoge * self.num_neurons / self.config.num_modules) % self.num_neurons)
            i_range2 = int((hoge + 1) * self.num_neurons / self.config.num_modules)
            if i_range2 > self.num_neurons:
                i_range2 = i_range2 % self.num_neurons
            j_range1 = int(((hoge + 1) * self.num_neurons / self.config.num_modules) % self.num_neurons)
            j_range2 = int((hoge + 2) * self.num_neurons / self.config.num_modules)
            if j_range2 > self.num_neurons:
                j_range2 = j_range2 % self.num_neurons
            reservoir_weight[i_range1:i_range2, j_range1:j_range2] = (
                (G * self.rng.randn(block_size, block_size)) + offset
            ) * (self.rng.rand(block_size, block_size) < p)
            # reservoir_weight[i_range1:i_range2, j_range1:j_range2] = (
            #     G*(self.rng.rand(block_size, block_size)-0.5) + offset
            #     ) * (self.rng.rand(block_size, block_size) < p)

            # variability = self.rng.lognormal(mean=0.0, sigma=2.0, size=(1, block_size))
            # variability = variability / np.mean(variability)
            # p_vec = p * variability
            # p_vec = np.clip(p_vec, 0.0, 1.0)
            # mask = self.rng.rand(block_size, block_size) < p_vec
            # reservoir_weight[i_range1:i_range2, j_range1:j_range2] = (
            #     G * (self.rng.rand(block_size, block_size) - 0.5) + offset
            # ) * mask

            i_range1 = int(((hoge + 1) * self.num_neurons / self.config.num_modules) % self.num_neurons)
            i_range2 = int((hoge + 2) * self.num_neurons / self.config.num_modules)
            if i_range2 > self.num_neurons:
                i_range2 = i_range2 % self.num_neurons
            j_range1 = int((hoge * self.num_neurons / self.config.num_modules) % self.num_neurons)
            j_range2 = int((hoge + 1) * self.num_neurons / self.config.num_modules)
            if j_range2 > self.num_neurons:
                j_range2 = j_range2 % self.num_neurons
            reservoir_weight[i_range1:i_range2, j_range1:j_range2] = (
                (G * self.rng.randn(block_size, block_size)) + offset
            ) * (self.rng.rand(block_size, block_size) < p)
            # reservoir_weight[i_range1:i_range2, j_range1:j_range2] = (
            #     G*(self.rng.rand(block_size, block_size)-0.5) + offset
            #     ) * (self.rng.rand(block_size, block_size) < p)

            # variability = self.rng.lognormal(mean=0.0, sigma=2.0, size=(1, block_size))
            # variability = variability / np.mean(variability)
            # p_vec = p * variability
            # p_vec = np.clip(p_vec, 0.0, 1.0)
            # mask = self.rng.rand(block_size, block_size) < p_vec
            # reservoir_weight[i_range1:i_range2, j_range1:j_range2] = (
            #     G * (self.rng.rand(block_size, block_size) - 0.5) + offset
            # ) * mask

        # 抑制結合の設定
        mask = np.ones((self.num_neurons, self.num_neurons))
        inhi_idx = self.rng.choice(np.arange(self.num_neurons), int(self.num_neurons/4), replace=False)
        mask[:, inhi_idx] = -1
        reservoir_weight = reservoir_weight * mask
        # reservoir_weight = np.zeros((N, N))#test
        # reservoir_weight[0, 1] = 1       #test
        type = np.where(mask[0,:] == 1, 0, 1)
        mask = (reservoir_weight != 0) * mask

        return reservoir_weight, mask, type
        mask = (reservoir_weight != 0) * mask

        return reservoir_weight, mask, type
