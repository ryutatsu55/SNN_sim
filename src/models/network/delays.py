import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from scipy.spatial.distance import pdist, squareform
from src.core.registry import DELAY_MODELS


class BaseDelay(ABC):
    """シナプスの伝播遅延を生成する基底クラス"""

    def __init__(
        self,
        config,  # ComponentConfig (Pydantic) または Dict のどちらも受け付ける
        num_neurons: int,
        coords: Optional[np.ndarray],
        mask: np.ndarray,
        rng: np.random.RandomState,
    ):
        # PydanticモデルとDictの両方に対応するため辞書に統一する
        if isinstance(config, dict):
            self._cfg = config
        else:
            self._cfg = config.model_dump()

        self.num_neurons = num_neurons
        self.coords = coords
        self.mask = mask
        self.rng = rng

    def _get(self, *keys, default=None):
        """
        ネストされたキーを安全に取得するヘルパー。
        例: self._get("empirical", "mean", default=4.0)
        """
        val = self._cfg
        for key in keys:
            if not isinstance(val, dict):
                return default
            val = val.get(key, default)
        return val

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: 形状 (num_neurons, num_neurons) の遅延行列 (np.float32)
        """
        pass

    def _apply_inter_module(
        self, delays: np.ndarray, num_modules: int, inter_module_value: float
    ) -> np.ndarray:
        """
        異なるモジュール間の接続に固定遅延を上書きする。(NumPyによる高速化版)
        """
        n = self.num_neurons
        module_size = n // num_modules

        r_module = np.arange(n)[:, None] // module_size  # (n, 1)
        c_module = np.arange(n)[None, :] // module_size  # (1, n)

        inter_mask = (r_module != c_module) & (self.mask != 0)
        delays[inter_mask] = inter_module_value

        return delays

    def _apply_inter_module_if_enabled(self, delays: np.ndarray) -> np.ndarray:
        """inter_moduleが有効な場合のみ上書き処理を実行"""
        if not self._get("inter_module", "enabled", default=False):
            return delays

        num_modules = self._get("network", "num_modules", default=4)
        value       = self._get("inter_module", "value", default=3.0)

        return self._apply_inter_module(delays, num_modules, value)


@DELAY_MODELS.register("empirical")
class EmpiricalDelay(BaseDelay):
    """正規分布に基づく伝播遅延（クリッピングあり）"""

    def generate(self) -> np.ndarray:
        mean      = self._get("empirical", "mean", default=4.0)
        std       = self._get("empirical", "std",  default=1.0)
        min_delay = self._get("empirical", "min",  default=1.0)
        max_delay = self._get("empirical", "max",  default=7.0)

        raw     = self.rng.randn(self.num_neurons, self.num_neurons) * std + mean
        clipped = np.clip(raw, min_delay, max_delay)

        delays = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        delays[self.mask != 0] = clipped[self.mask != 0]

        return self._apply_inter_module_if_enabled(delays)


@DELAY_MODELS.register("constant")
class ConstantDelay(BaseDelay):
    """全シナプスで共通の伝播遅延"""

    def generate(self) -> np.ndarray:
        value = self._get("constant", "value", default=1.0)

        delays = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        delays[self.mask != 0] = value

        return self._apply_inter_module_if_enabled(delays)


@DELAY_MODELS.register("distance_based")
class DistanceBasedDelay(BaseDelay):
    """物理的な距離を伝播速度で割って遅延を決定する"""

    def generate(self) -> np.ndarray:
        if self.coords is None:
            raise ValueError(
                "DistanceBasedDelay requires spatial coordinates "
                "(coords cannot be None)."
            )

        velocity  = self._get("distance_based", "velocity",  default=1.0)
        min_delay = self._get("distance_based", "min_delay", default=0.1)

        dist_matrix = squareform(pdist(self.coords))
        calc_delays = (dist_matrix / velocity) + min_delay

        delays = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        delays[self.mask != 0] = calc_delays[self.mask != 0]

        return self._apply_inter_module_if_enabled(delays)


def create_delay_model(
    config,
    num_neurons: int,
    coords: Optional[np.ndarray],
    mask: np.ndarray,
    rng: np.random.RandomState,
) -> BaseDelay:
    """
    YAMLのdelay_typeに基づいて対応するDelayクラスを返すファクトリ関数。

    Usage:
        model = create_delay_model(config, num_neurons, coords, mask, rng)
        delay_matrix = model.generate()
    """
    # PydanticモデルとDictの両方に対応
    if isinstance(config, dict):
        delay_type = config.get("delay_type", "empirical")
    else:
        delay_type = getattr(config, "delay_type", "empirical")

    model_class = DELAY_MODELS.get(delay_type)

    if model_class is None:
        available = list(DELAY_MODELS.keys())
        raise ValueError(
            f"Unknown delay_type: '{delay_type}'. "
            f"Available options: {available}"
        )

    return model_class(config, num_neurons, coords, mask, rng)