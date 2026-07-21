import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from numbers import Real
from scipy.spatial.distance import pdist, squareform
from src.core.registry import DELAY_MODELS

class BaseDelay(ABC):
    """シナプスの伝播遅延を生成する基底クラス"""
    def __init__(self, config: Dict[str, Any], num_neurons: int, coords: Optional[np.ndarray], mask: np.ndarray, rng: np.random.RandomState, layout=None):
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

        # 距離行列の計算
        dist_matrix = squareform(pdist(self.coords))

        # 距離 / 速度(軸索伝導) + 最小遅延 + シナプス遅延オフセット
        calc_delays = (dist_matrix / velocity) + min_delay + float(synaptic_delay)

        if max_delay is not None:
            if not isinstance(max_delay, Real) or isinstance(max_delay, bool):
                raise ValueError("max_delay must be a real number or None.")
            if max_delay < min_delay:
                raise ValueError(
                    f"max_delay ({max_delay}) must not be less than min_delay ({min_delay})."
                )
            calc_delays = np.minimum(calc_delays, float(max_delay))

        # 結合がない箇所の遅延は0にする（メモリ効率とバグ防止のため）
        delays = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        idx = (self.mask != 0)
        delays[idx] = calc_delays[idx]

        return delays

@DELAY_MODELS.register("type_based")
class TypeBasedDelay(BaseDelay):
    """種別(E/I)ペア別の均一遅延。

    `layout.ids_by_mode()` で興奮性/抑制性のグローバルIDを分け、送信種別 × 受信種別の
    4 ブロック(EE/EI/IE/II)にそれぞれ一定の遅延[ms]を割り当てる。`delay_by_target`
    (= NetworkBuilder の axonal 経路)を使わずに per-(src,tgt)-type 遅延を network.delay
    経由(非axonal の per-synapse dc 経路)で与えるための手段。空間座標は不要(coords 非依存)。

    config(フラットなスカラーフィールド, 単位 [ms]):
        d_ee (Exc→Exc), d_ei (Exc→Inh), d_ie (Inh→Exc), d_ii (Inh→Inh)
    """

    def generate(self):
        if self.layout is None:
            raise ValueError(
                "TypeBasedDelay requires a NetworkLayout to resolve E/I populations."
            )

        ids = self.layout.ids_by_mode()
        exc = ids["excitatory"]
        inh = ids["inhibitory"]

        delays = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)

        # 送信種別(行) × 受信種別(列)の 4 ブロックにそれぞれの遅延を割り当てる。
        # global_delays は [source, target] 配向(NetworkBuilder が np.ix_(src, tgt) で切出す)。
        blocks = (
            (exc, exc, self.config.d_ee),
            (exc, inh, self.config.d_ei),
            (inh, exc, self.config.d_ie),
            (inh, inh, self.config.d_ii),
        )
        for src_ids, tgt_ids, value in blocks:
            if len(src_ids) == 0 or len(tgt_ids) == 0:
                continue
            v = float(value)
            if not isinstance(value, Real) or isinstance(value, bool):
                raise ValueError("type_based delays (d_ee/d_ei/d_ie/d_ii) must be real numbers.")
            if v < 0.0:
                raise ValueError("type_based delays must be greater than or equal to 0.0.")
            delays[np.ix_(src_ids, tgt_ids)] = v

        # 結合がない箇所の遅延は0にする(既存 delay モデルと同じ規約)。
        delays[self.mask == 0] = 0.0
        return delays

@DELAY_MODELS.register("akita_soc")
class AkitaSocTypeDelay(TypeBasedDelay):
    """akita_soc_delay.yaml の delay_by_target と同一の遅延を network.delay 経由で再現する
    種別ペア別均一遅延プロファイル。具体値は YAML から読む。"""
    pass

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
