import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path
from scipy.spatial.distance import cdist, pdist, squareform
from src.core.registry import CONNECTION_MODELS

class BaseConnection(ABC):
    """シナプス結合の有無(マスク)を決定する基底クラス"""

    # 密な (N,N) マスクを作らずに COO を直接生成できるか。True を宣言したクラスは
    # generate_sparse() を実装しなければならない。NetworkBuilder は結合/重み/遅延の
    # 3 段すべてが True のときだけ疎生成経路を選ぶ。
    supports_sparse: bool = False

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

    def generate_sparse(self) -> tuple[np.ndarray, np.ndarray]:
        """結合を COO (rows, cols) で返す。行優先ソート済みであること。

        Returns:
            (rows, cols): それぞれ int32 のグローバル pre/post ID
        """
        raise NotImplementedError(
            f"{type(self).__name__} は疎生成に対応していません (supports_sparse=False)。"
        )

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

@CONNECTION_MODELS.register("gaussian_distance_type")
class GaussianDistanceTypeTopology(BaseConnection):
    """種別別(E→E, E→I, I→E, I→I)のガウス型距離依存結合。

    Beggs & Plenz (2003) 再現用(docs/refs/SNN_PA~1.MD §3)。結合確率は
        P(結合 | 距離 d) = p0_xy * exp(-d^2 / (2 * sigma_xy^2))
    で、送信種別 x(E/I) × 受信種別 y(E/I) の 4 ブロックごとに独立の
    (sigma_xy, p0_xy) を用いる。興奮性/抑制性のグローバルID集合は
    layout.ids_by_mode() から取得する。

    config(フラットなスカラーフィールド):
        sigma_ee/p0_ee, sigma_ei/p0_ei, sigma_ie/p0_ie, sigma_ii/p0_ii  … [um] と確率
        allow_self_connections (bool, 既定 False) … False で対角(オートシナプス)を除去

    大規模ネットワーク向けに generate_sparse() を実装しており、密な (N,N) を作らずに
    COO を直接生成する。乱数ストリームの消費順は密版と完全に一致する(下記参照)。
    """

    supports_sparse = True

    # 疎生成時の 1 行ブロックあたりの一時配列サイズの目安 [bytes]。
    _ROW_BLOCK_BYTES = 128 << 20

    def _validate_inputs(self):
        if self.coords is None:
            raise ValueError(
                "GaussianDistanceTypeTopology requires spatial coordinates (coords cannot be None)."
            )
        if self.layout is None:
            raise ValueError(
                "GaussianDistanceTypeTopology requires a NetworkLayout to resolve E/I populations."
            )

    def generate_sparse(self):
        """行ブロックごとに距離・確率・乱数を作って COO を組み立てる。

        乱数ストリームの同一性:
            RandomState.random_sample は指定形状ぶんの double を Mersenne ストリームから
            逐次消費して C 順に reshape する。よって連続した行ブロック [0,nb), [nb,2nb), ...
            に対して (nb, N) を順に引くことは、(N, N) を一括で引くのとビット単位で同一の
            値が同一の (i, j) に対応する。これを壊さないため:
              - prob == 0 のペアでもドローを省略しない(ブロック全体を無条件にドロー)
              - 自己結合の除去は比較の「後」に行う(密版の fill_diagonal と同じく
                対角のドローは消費して捨てる)
              - 空の id 集合に対する continue は確率の埋め込みだけをスキップし、
                ドローはスキップしない
        """
        self._validate_inputs()

        ids = self.layout.ids_by_mode()
        exc = ids["excitatory"]
        inh = ids["inhibitory"]
        n = self.num_neurons
        coords = np.ascontiguousarray(self.coords, dtype=np.float64)

        is_exc = np.zeros(n, dtype=bool)
        is_exc[exc] = True
        is_inh = np.zeros(n, dtype=bool)
        is_inh[inh] = True
        cols_e = np.nonzero(is_exc)[0]
        cols_i = np.nonzero(is_inh)[0]

        block = max(1, min(n, self._ROW_BLOCK_BYTES // max(1, 8 * n)))
        allow_self = getattr(self.config, "allow_self_connections", False)

        out_rows: list[np.ndarray] = []
        out_cols: list[np.ndarray] = []

        for start in range(0, n, block):
            stop = min(start + block, n)
            dist = cdist(coords[start:stop], coords)
            prob = np.zeros((stop - start, n), dtype=np.float64)

            local_e = np.nonzero(is_exc[start:stop])[0]
            local_i = np.nonzero(is_inh[start:stop])[0]
            for src_local, tgt_global, sigma, p0 in (
                (local_e, cols_e, self.config.sigma_ee, self.config.p0_ee),
                (local_e, cols_i, self.config.sigma_ei, self.config.p0_ei),
                (local_i, cols_e, self.config.sigma_ie, self.config.p0_ie),
                (local_i, cols_i, self.config.sigma_ii, self.config.p0_ii),
            ):
                if src_local.size == 0 or tgt_global.size == 0:
                    continue
                block_idx = np.ix_(src_local, tgt_global)
                d = dist[block_idx]
                prob[block_idx] = p0 * np.exp(-(d ** 2) / (2.0 * sigma ** 2))

            hit = self.rng.random((stop - start, n)) < prob
            if not allow_self:
                # 密版の np.fill_diagonal(mask, 0) と等価。ドローは既に消費済み。
                hit[np.arange(stop - start), np.arange(start, stop)] = False

            rows, cols = np.nonzero(hit)
            if rows.size:
                out_rows.append((rows + start).astype(np.int32))
                out_cols.append(cols.astype(np.int32))

        if not out_rows:
            empty = np.array([], dtype=np.int32)
            return empty, empty
        return np.concatenate(out_rows), np.concatenate(out_cols)

    def generate(self):
        self._validate_inputs()

        ids = self.layout.ids_by_mode()
        exc = ids["excitatory"]
        inh = ids["inhibitory"]

        # N x N の距離行列を一括計算(DistanceBasedTopology と同手法)
        dist_matrix = squareform(pdist(self.coords))

        prob_matrix = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float64)

        # 送信種別(行) × 受信種別(列)の 4 ブロックにそれぞれの (sigma, p0) を適用
        blocks = (
            (exc, exc, self.config.sigma_ee, self.config.p0_ee),
            (exc, inh, self.config.sigma_ei, self.config.p0_ei),
            (inh, exc, self.config.sigma_ie, self.config.p0_ie),
            (inh, inh, self.config.sigma_ii, self.config.p0_ii),
        )
        for src_ids, tgt_ids, sigma, p0 in blocks:
            if len(src_ids) == 0 or len(tgt_ids) == 0:
                continue
            block_idx = np.ix_(src_ids, tgt_ids)
            d = dist_matrix[block_idx]
            prob_matrix[block_idx] = p0 * np.exp(-(d ** 2) / (2.0 * sigma ** 2))

        mask = self.rng.random((self.num_neurons, self.num_neurons)) < prob_matrix

        if not getattr(self.config, "allow_self_connections", False):
            np.fill_diagonal(mask, 0)  # オートシナプス禁止 (MD §8)

        return mask.astype(np.int8)

@CONNECTION_MODELS.register("beggs_plenz")
class BeggsPlenzGaussianTopology(GaussianDistanceTypeTopology):
    """Beggs & Plenz (2003) 再現用の種別別ガウス結合プロファイル。具体値は YAML から読む。"""
    pass

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
