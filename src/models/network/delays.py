import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from numbers import Real
from src.core.registry import DELAY_MODELS

class BaseDelay(ABC):
    """シナプスの伝播遅延を生成する基底クラス"""

    # True のクラスは「(送信集団, 受信集団) ごとに単一値の遅延」を返すことを約束する。
    # NetworkBuilder はそれを GeNN の軸索遅延 (SynapseGroup 一様の axonal_delay_steps) で
    # 実装する (可塑性モデルが axonal_delay_steps を受け取れる場合のみ)。
    axonal = False

    def __init__(self, config: Dict[str, Any], num_neurons: int, coords: Optional[np.ndarray], rng: np.random.RandomState = None, layout=None):
        self.config = config
        self.num_neurons = num_neurons
        self.coords = coords
        self.rng = rng
        # NetworkLayout。ニューロン種ごとの意図的バイアスや無相関化(シャッフル)を
        # 具象クラス側で実装したい場合に self.layout.ids_by("polarity") などを参照する。
        self.layout = layout

    def describe_axes(self) -> Dict[str, Any]:
        """任意フック: このコンポーネントが定義するカテゴリ/ソート軸を宣言する。

        NetworkBuilder が `generate()` の**直後**に呼び、戻り値を
        `NetworkLayout.add_axis()` へ注入する。生成後に呼ばれるので、自身が計算した
        座標・結合・遅延等から軸を導出してよい(例: 遅延帯でのビン分け)。

        Returns:
            {軸名: 長さ num_neurons の配列}。カテゴリ名(文字列)でもソート用の数値でも
            よい。既定は空 dict = 軸を定義しない。
        """
        return {}

    @abstractmethod
    def generate(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """結合 COO (rows, cols) の各要素に対応する 1D 遅延 [ms] 配列 (np.float32) を返す。

        乱数を使うモデルは rows, cols の並び (行優先) の順に 1 結合 1 値で引く。
        """

@DELAY_MODELS.register("constant")
class ConstantDelay(BaseDelay):
    def generate(self, rows, cols):
        """全結合に同じ遅延 `value` [ms] を割り当てる。"""
        return np.full(np.asarray(rows).size, self.config.value, dtype=np.float32)

@DELAY_MODELS.register("distance_based")
class DistanceBasedDelay(BaseDelay):

    def _params(self):
        """遅延パラメータを検証して (velocity, min_delay, max_delay, synaptic_delay) を返す。"""
        if self.coords is None:
            raise ValueError("DistanceBasedDelay requires spatial coordinates (coords cannot be None).")

        velocity = self.config.velocity
        min_delay = self.config.min_delay
        # 任意の上限。距離が大きい空間(実コネクトーム等)で遅延が uint8 のステップ上限
        # (NetworkBuilder で delay/dt を uint8 化)を超えて折り返すのを防ぐ安全キャップ。
        max_delay = getattr(self.config, "max_delay", None)
        # 任意のシナプス遅延オフセット[ms]。軸索伝導(距離/速度)に加算される固定の
        # シナプス伝達遅延成分。既定 0.0(=従来挙動)。
        synaptic_delay = getattr(self.config, "synaptic_delay", 0.0)
        if not isinstance(synaptic_delay, Real) or isinstance(synaptic_delay, bool):
            raise ValueError("synaptic_delay must be a real number.")
        if synaptic_delay < 0.0:
            raise ValueError("synaptic_delay must be greater than or equal to 0.0.")

        if max_delay is not None:
            if not isinstance(max_delay, Real) or isinstance(max_delay, bool):
                raise ValueError("max_delay must be a real number or None.")
            if max_delay < min_delay:
                raise ValueError(
                    f"max_delay ({max_delay}) must not be less than min_delay ({min_delay})."
                )

        return velocity, min_delay, max_delay, float(synaptic_delay)

    def _delays_from_distance(self, distance):
        velocity, min_delay, max_delay, synaptic_delay = self._params()
        calc = (distance / velocity) + min_delay + synaptic_delay
        if max_delay is not None:
            calc = np.minimum(calc, float(max_delay))
        return calc

    def generate(self, rows, cols):
        self._params()  # coords / パラメータの検証
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        if rows.size == 0:
            return np.zeros(0, dtype=np.float32)
        coords = np.asarray(self.coords, dtype=np.float64)
        distance = np.linalg.norm(coords[rows] - coords[cols], axis=1)
        return self._delays_from_distance(distance).astype(np.float32)


@DELAY_MODELS.register("type_based")
class TypeBasedDelay(BaseDelay):
    """種別(E/I)ペア別の均一遅延。

    `layout.ids_by("polarity")` で興奮性/抑制性のグローバルIDを分け、送信種別 × 受信種別の
    4 ブロック(EE/EI/IE/II)にそれぞれ一定の遅延[ms]を割り当てる。GeNN 上は per-synapse
    遅延 (dendritic) として実装される。空間座標は不要(coords 非依存)。

    config(フラットなスカラーフィールド, 単位 [ms]):
        d_ee (Exc→Exc), d_ei (Exc→Inh), d_ie (Inh→Exc), d_ii (Inh→Inh)
    """

    def generate(self, rows, cols):
        """送信・受信ニューロンの極性 (E/I) の組で 4 値の遅延を引く。E/I 以外のニューロンは 0。"""
        exc, inh, values = self._blocks()
        code = np.full(self.num_neurons, -1, dtype=np.int64)
        code[np.asarray(exc, dtype=np.int64)] = 0
        code[np.asarray(inh, dtype=np.int64)] = 1
        table = np.array([[values[0], values[1]], [values[2], values[3]]], dtype=np.float32)
        src = code[np.asarray(rows, dtype=np.int64)]
        tgt = code[np.asarray(cols, dtype=np.int64)]
        known = (src >= 0) & (tgt >= 0)
        delays = np.zeros(src.size, dtype=np.float32)
        delays[known] = table[src[known], tgt[known]]
        return delays

    def _blocks(self):
        """(興奮性 ID, 抑制性 ID, (d_ee, d_ei, d_ie, d_ii)) を検証して返す。

        値の検証は送信・受信どちらの集合も空でないブロックについてだけ行う。
        """
        if self.layout is None:
            raise ValueError(
                "TypeBasedDelay requires a NetworkLayout to resolve E/I populations."
            )
        ids = self.layout.ids_by("polarity")
        exc = ids["excitatory"]
        inh = ids["inhibitory"]
        values = []
        for src_ids, tgt_ids, value in (
            (exc, exc, self.config.d_ee),
            (exc, inh, self.config.d_ei),
            (inh, exc, self.config.d_ie),
            (inh, inh, self.config.d_ii),
        ):
            if len(src_ids) == 0 or len(tgt_ids) == 0:
                values.append(0.0)
                continue
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError("type_based delays (d_ee/d_ei/d_ie/d_ii) must be real numbers.")
            v = float(value)
            if v < 0.0:
                raise ValueError("type_based delays must be greater than or equal to 0.0.")
            values.append(v)
        return exc, inh, values

@DELAY_MODELS.register("type_based_axonal")
class TypeBasedAxonalDelay(TypeBasedDelay):
    """`type_based` と同じ値を、GeNN の軸索遅延として実装させる種別ペア別遅延。

    polarity は集団ごとに 1 つなので、種別ペアで一様なら (送信集団, 受信集団) でも一様になる。
    """
    axonal = True

@DELAY_MODELS.register("akita_soc")
class AkitaSocTypeDelay(TypeBasedDelay):
    """Akita SoC の種別ペア別遅延を per-synapse 遅延 (dendritic) で実装するプロファイル。
    具体値は YAML から読む。"""
    pass

@DELAY_MODELS.register("akita_soc_axonal")
class AkitaSocAxonalDelay(TypeBasedAxonalDelay):
    """Akita SoC の種別ペア別遅延を軸索遅延で実装するプロファイル。具体値は YAML から読む。"""
    pass

@DELAY_MODELS.register("random")
class RandomDelay(BaseDelay):
    def generate(self, rows, cols):
        """各結合に N(mean, std²) を float32 に丸めて [min, max] で切った遅延を割り当てる。"""
        mean, std, lower, upper = self._params()
        num_connections = int(np.asarray(rows).size)
        if num_connections == 0:
            return np.zeros(0, dtype=np.float32)
        sampled = self.rng.normal(mean, std, size=num_connections).astype(np.float32)
        return np.clip(sampled, lower, upper)

    def _params(self):
        """(mean, std, 下限, 上限) を検証して返す。min / max の None は ±inf。"""
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

        lower = -np.inf if min_delay is None else float(min_delay)
        upper = np.inf if max_delay is None else float(max_delay)
        return float(mean), float(std), lower, upper
