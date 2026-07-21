import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from numbers import Real
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from src.core.registry import WEIGHT_MODELS

class BaseWeight(ABC):
    """シナプスの重みを生成する基底クラス"""

    # 密な (N,N) を作らずに COO 上の 1D 重みを直接生成できるか。
    supports_sparse: bool = False

    def __init__(self, config: Dict[str, Any], num_neurons: int, coords: Optional[np.ndarray], mask: Optional[np.ndarray] = None, rng: np.random.RandomState = None, layout=None):
        self.config = config
        self.num_neurons = num_neurons
        self.coords = coords
        self.mask = mask
        self.rng = rng
        # NetworkLayout。ニューロン種ごとの意図的バイアスや無相関化(シャッフル)を
        # 具象クラス側で実装したい場合に self.layout.items() / ids_by_mode() を参照する。
        self.layout = layout

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: 形状 (num_neurons, num_neurons) の重み行列 (np.float32)
        """
        pass

    def generate_sparse(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """COO (rows, cols) に対応する 1D 重み配列 (np.float32) を返す。"""
        raise NotImplementedError(
            f"{type(self).__name__} は疎生成に対応していません (supports_sparse=False)。"
        )

@WEIGHT_MODELS.register("constant")
class ConstantWeight(BaseWeight):
    def generate(self):
        """結合がある箇所すべてに同じ重みを割り当てる"""
        val = self.config.base_weight
        weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        weights[self.mask != 0] = val
        return weights

@WEIGHT_MODELS.register("constant_zero")
class ConstantZeroWeight(ConstantWeight):
    """論文再現用の初期重み0プロファイル。具体値はYAMLから読む。"""
    pass

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

@WEIGHT_MODELS.register("lognormal_clip")
class LogNormalClipWeight(BaseWeight):
    """対数正規分布 + [w_min, w_max] クリップの初期重み。

    Beggs & Plenz (2003) 再現用(docs/refs/SNN_PA~1.MD §10)。28 DIV の成熟培養網を模し、
    ゼロ初期を避けて `w ~ LogNormal(mu_ln, sigma_ln^2)` を `[w_min, w_max]` に切り捨てる
    (既存 `lognormal_broad` はクリップ無しのため別クラスにする)。custom_Akita の重み `w` は
    正規化 [0, Wmax] 空間なので w_max は Wmax(既定 1.0)に合わせる。

    config(フラットなスカラーフィールド):
        mu_ln    … 台形正規分布(下地の正規分布)の平均。MD §10 は mu_ln = ln(0.7) - sigma_ln^2/2
        sigma_ln … 広がり(MD §10: 0.8〜1.2)
        w_min, w_max … クリップ範囲(= custom_Akita の Wmin/Wmax に対応)
    """

    supports_sparse = True

    def _params(self):
        mu_ln = self.config.mu_ln
        sigma_ln = self.config.sigma_ln
        w_min = self.config.w_min
        w_max = self.config.w_max

        for pname, pval in (("mu_ln", mu_ln), ("sigma_ln", sigma_ln),
                            ("w_min", w_min), ("w_max", w_max)):
            if not isinstance(pval, Real) or isinstance(pval, bool):
                raise ValueError(f"{pname} must be a real number.")
        if sigma_ln < 0.0:
            raise ValueError("sigma_ln must be greater than or equal to 0.0.")
        if w_min > w_max:
            raise ValueError(f"w_min ({w_min}) must not exceed w_max ({w_max}).")
        return float(mu_ln), float(sigma_ln), float(w_min), float(w_max)

    def _sample(self, num_conns: int) -> np.ndarray:
        mu_ln, sigma_ln, w_min, w_max = self._params()
        sampled = self.rng.lognormal(mu_ln, sigma_ln, size=num_conns)
        # MD §10: 切り捨て [w_min, w_max]
        return np.clip(sampled, w_min, w_max).astype(np.float32)

    def generate(self):
        self._params()  # 疎版と同じ検証を先に通す
        weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        idx = (self.mask != 0)
        num_conns = int(np.count_nonzero(idx))
        if num_conns == 0:
            return weights

        # NumPy のブール代入は idx の C 順(= np.nonzero(mask) の順)で埋めるため、
        # 行優先ソート済み COO に対する generate_sparse と同一の割り当てになる。
        weights[idx] = self._sample(num_conns)
        return weights

    def generate_sparse(self, rows, cols):
        num_conns = int(np.asarray(rows).size)
        if num_conns == 0:
            self._params()
            return np.zeros(0, dtype=np.float32)
        return self._sample(num_conns)

@WEIGHT_MODELS.register("beggs_plenz")
class BeggsPlenzLogNormalWeight(LogNormalClipWeight):
    """Beggs & Plenz (2003) 再現用の対数正規初期重みプロファイル。具体値は YAML から読む。"""
    pass

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

@WEIGHT_MODELS.register("C.elegans")
class C_elegansWeight(BaseWeight):
    """C. elegans コネクトームの実重み(化学 + 電気シナプス)を読み込む重みモデル。

    化学(``weight_matrix_chem.csv``)/電気(``weight_matrix_elec.csv``)シナプス行列の
    どちらを使うか、両方使うときの優先、クリップ後の正規化有無を config で指定できる。

    config パラメータ:
      - ``sources``  : ``"chem"`` | ``"elec"`` | ``"both"``(既定 ``"both"``)。使う行列。
      - ``priority`` : ``"chem"`` | ``"elec"``(既定 ``"elec"``)。``both`` で両方に接続が
                       ある要素にどちらの重みを採るか(既定は elec 優先 = gap junction 優先)。
      - ``min_weight`` / ``max_weight`` : 接続重みのクリップ範囲。
      - ``normalize`` : bool(既定 ``True``)。``True`` ならクリップ後に ``max_weight`` で
                        割って正規化(クリップ上限が 1.0 に写像される)。``False`` なら
                        クリップ後の生値([min_weight, max_weight])のまま。

    生成された重みは接続がある箇所(``self.mask != 0`` かつ選択ソースに接続がある要素)に
    限定される。両 CSV は 81×81 で ``ordered_coords.csv`` と同じ NodeID(正準グローバルID)
    順に整列している前提。重みは大きさ(正値)で、興奮性/抑制性の符号は発信 population の
    シナプスモデル側で扱う。
    """

    _CHEM_NAME = "weight_matrix_chem.csv"
    _ELEC_NAME = "weight_matrix_elec.csv"

    @classmethod
    def _data_path(cls, name: str) -> Path:
        return Path(__file__).parent / "data" / "c_elegans" / name

    def _load_matrix(self, name: str) -> np.ndarray:
        path = self._data_path(name)
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at {path}")
        m = np.loadtxt(path, delimiter=",", dtype=np.float64)
        if m.shape != (self.num_neurons, self.num_neurons):
            raise ValueError(
                f"{name} shape {m.shape} does not match "
                f"expected ({self.num_neurons}, {self.num_neurons})"
            )
        return m

    def generate(self):
        min_weight = self.config.min_weight
        max_weight = self.config.max_weight
        sources = getattr(self.config, "sources", "both")
        priority = getattr(self.config, "priority", "elec")
        normalize = getattr(self.config, "normalize", True)

        for pname, pval in (("min_weight", min_weight), ("max_weight", max_weight)):
            if not isinstance(pval, Real) or isinstance(pval, bool):
                raise ValueError(f"{pname} must be a real number.")
        if min_weight > max_weight:
            raise ValueError(
                f"min_weight ({min_weight}) must not exceed max_weight ({max_weight})."
            )
        if not isinstance(normalize, bool):
            raise ValueError("normalize must be a boolean.")
        if normalize and max_weight <= 0:
            raise ValueError(
                f"max_weight ({max_weight}) must be positive when normalize is True."
            )
        if sources not in ("chem", "elec", "both"):
            raise ValueError(f"sources must be 'chem' | 'elec' | 'both' (got {sources!r}).")
        if priority not in ("chem", "elec"):
            raise ValueError(f"priority must be 'chem' | 'elec' (got {priority!r}).")

        # --- ソース選択(chem / elec / both・優先指定) ---
        if sources == "chem":
            weights = self._load_matrix(self._CHEM_NAME).copy()
        elif sources == "elec":
            weights = self._load_matrix(self._ELEC_NAME).copy()
        else:  # both
            chem = self._load_matrix(self._CHEM_NAME)
            elec = self._load_matrix(self._ELEC_NAME)
            if priority == "elec":
                # 化学を基礎に、電気接続がある要素を elec で上書き(gap junction 優先)。
                weights = chem.copy()
                weights[elec != 0] = elec[elec != 0]
            else:  # priority == "chem"
                weights = elec.copy()
                weights[chem != 0] = chem[chem != 0]

        # --- マスク外を 0 に。選択ソースに接続がある要素(=非ゼロ)のみクリップ/正規化。 ---
        #     元データは接続=synapse数>=1 / 非接続=0 なので、非ゼロ判定で「接続あり」を識別できる。
        #     (min_weight>0 でも非接続 0 を持ち上げてしまわないよう conn は非ゼロで取る。)
        weights[self.mask == 0] = 0.0
        conn = weights != 0
        weights[conn] = np.clip(weights[conn], min_weight, max_weight)
        if normalize:
            weights[conn] = weights[conn] / max_weight

        return weights.astype(np.float32)

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
