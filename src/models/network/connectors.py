import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from src.core.registry import CONNECTION_MODELS

class BaseConnection(ABC):
    """シナプス結合の有無(マスク)を決定する基底クラス"""
    def __init__(self, config: Dict[str, Any], num_neurons: int, coords: Optional[np.ndarray], rng: np.random.RandomState, layout=None):
        self.config = config
        self.num_neurons = num_neurons
        self.coords = coords
        self.rng = rng
        # NetworkLayout。ニューロン種ごとの意図的バイアスや無相関化(シャッフル)を
        # 具象クラス側で実装したい場合に self.layout.items() / ids_by_mode() を参照する。
        self.layout = layout

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
        prob = self.config.p
        
        mask = self.rng.random((self.num_neurons, self.num_neurons)) < prob
        if not self.config.allow_self_connections:
            np.fill_diagonal(mask, 0) # 自己結合（自分自身へのシナプス）を排除
        return mask.astype(np.int8)

@CONNECTION_MODELS.register("constant_prob_full")
class FullProbabilityTopology(ConstantProbabilityTopology):
    """論文再現用の全ペア結合候補。具体値はYAMLプロファイルから読む。"""
    pass

@CONNECTION_MODELS.register("constant_prob_full_autapse")
class FullAutapseProbabilityTopology(ConstantProbabilityTopology):
    """自己結合あり検証用の全ペア結合候補。具体値はYAMLプロファイルから読む。"""
    pass

@CONNECTION_MODELS.register("constant_prob_sparse")
class SparseProbabilityTopology(ConstantProbabilityTopology):
    """Akita SoC安定化候補用の疎な確率結合。具体値はYAMLプロファイルから読む。"""
    pass

@CONNECTION_MODELS.register("optional_connect")
class OptionalConnection(BaseConnection):
    def generate(self):
        """任意の結合を指定する"""        
        mask = np.zeros((self.num_neurons, self.num_neurons))
        mask[self.config.src_ID, self.config.tgt_ID] = 1
        return mask.astype(np.int8)
    
@CONNECTION_MODELS.register("C.elegans")
class C_elegansConnection(BaseConnection):
    def generate(self):
        """synapse_mask.csvから接続マスクを読み込む。
        元データにおいては0以外はすべて接続（1）として定義する。
        """
        csv_path = Path(__file__).parent / "data" / "c_elegans" / "synapse_mask.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"synapse_mask.csv not found at {csv_path}")

        # synapse_mask.csvを読み込む
        mask_data = np.loadtxt(csv_path, delimiter=",", dtype=np.int8)

        # 形状チェック
        if mask_data.shape != (self.num_neurons, self.num_neurons):
            raise ValueError(
                f"synapse_mask shape {mask_data.shape} does not match "
                f"expected ({self.num_neurons}, {self.num_neurons})"
            )

        # 0以外をすべて1に変換
        mask = (mask_data != 0).astype(np.int8)

        return mask

@CONNECTION_MODELS.register("distance_based")
class DistanceBasedTopology(BaseConnection):
    def generate(self):
        """空間座標間の距離に基づき、ガウス分布で減衰する確率結合を生成"""
        if self.coords is None:
            raise ValueError("DistanceBasedTopology requires spatial coordinates (coords cannot be None).")

        # N x N の距離行列を一括計算
        dist_matrix = squareform(pdist(self.coords))
        
        # 距離に応じた結合確率を計算
        prob_matrix = self.config.max_prob * np.exp(-(dist_matrix**2) / (2 * self.config.sigma**2))
        
        # 確率に基づいてマスクを生成
        mask = self.rng.random((self.num_neurons, self.num_neurons)) < prob_matrix
        
        return mask.astype(np.int8)

@CONNECTION_MODELS.register("prob_based_block")
class BlockRandomTopology(BaseConnection):
    def generate(self):
        """ブロック分割ベースの確率結合を生成する。

        モジュール境界は、NetworkLayout に構造層(複数)が定義されていればそれを用いる
        (層 = ブロック)。層が無い/単一層のときは従来通り config.num_modules で等分する。
        num_modules=1(または単一層)では単一ブロックのランダム結合として振る舞う。
        """
        within_module_connection_prob = self.config.within_module_connection_prob
        between_module_connection_prob = self.config.between_module_connection_prob
        allow_self_connections = self.config.allow_self_connections

        for prob_name, prob_value in (
            ("within_module_connection_prob", within_module_connection_prob),
            ("between_module_connection_prob", between_module_connection_prob),
        ):
            if not isinstance(prob_value, (int, float)) or isinstance(prob_value, bool):
                raise ValueError(f"{prob_name} must be a real number.")
            if not 0.0 <= prob_value <= 1.0:
                raise ValueError(f"{prob_name} must be between 0.0 and 1.0.")

        if not isinstance(allow_self_connections, bool):
            raise ValueError("allow_self_connections must be a boolean.")

        # ニューロンをモジュールに分割するためのインデックス範囲を計算する。
        # NetworkLayout に複数の構造層があれば、その連続ブロックをモジュールとして採用する。
        layers = self.layout.layers() if self.layout is not None else None
        if layers is not None and len(layers) > 1:
            module_ranges: list[tuple[int, int]] = [(l.start, l.stop) for l in layers]
        else:
            num_modules = self.config.num_modules
            if not isinstance(num_modules, int) or isinstance(num_modules, bool):
                raise ValueError("num_modules must be an integer.")
            if num_modules < 1:
                raise ValueError("num_modules must be at least 1.")
            if num_modules > self.num_neurons:
                raise ValueError("num_modules must not exceed num_neurons.")
            module_ranges = []
            for module_idx in range(num_modules):
                start = int(module_idx * self.num_neurons / num_modules)
                end = int((module_idx + 1) * self.num_neurons / num_modules)
                module_ranges.append((start, end))

        num_modules = len(module_ranges)
        mask = np.zeros((self.num_neurons, self.num_neurons), dtype=np.int8)

        # モジュール内の結合を生成
        for start, end in module_ranges:
            block_shape = (end - start, end - start)
            block_mask = self.rng.rand(*block_shape) < within_module_connection_prob
            mask[start:end, start:end] = block_mask.astype(np.int8)

        # モジュール間の結合を生成
        for module_idx in range(num_modules):
            current_module_start, current_module_end = module_ranges[module_idx]
            # module_idx + 1 == num_modulesのとき、IndexErrorになる。よって、%演算子を使うことで、module_idx + 1 == num_modulesのときは、0に戻るようにする。
            next_module_start, next_module_end = module_ranges[(module_idx + 1) % num_modules]

            forward_block_shape = (
                current_module_end - current_module_start,
                next_module_end - next_module_start,
            )
            backward_block_shape = (
                next_module_end - next_module_start,
                current_module_end - current_module_start,
            )

            # 各モジュールを隣接モジュールと双方向に接続する。
            forward_mask = self.rng.rand(*forward_block_shape) < between_module_connection_prob
            backward_mask = self.rng.rand(*backward_block_shape) < between_module_connection_prob

            mask[current_module_start:current_module_end, next_module_start:next_module_end] = forward_mask.astype(np.int8)
            mask[next_module_start:next_module_end, current_module_start:current_module_end] = backward_mask.astype(np.int8)

        if not allow_self_connections:
            np.fill_diagonal(mask, 0)

        return mask
