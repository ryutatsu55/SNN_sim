"""保存済みCOOを既存NetworkBuilderへ接続する登録コンポーネント。"""

import numpy as np

from src.core.network_snapshot import arrival_to_dendritic_ms, load_snapshot
from src.core.registry import (
    CONNECTION_MODELS,
    DELAY_MODELS,
    SPATIAL_MODELS,
    WEIGHT_MODELS,
)
from .connectors import BaseConnection
from .delays import BaseDelay
from .space import BaseSpace
from .weights import BaseWeight


def _read(component):
    arrays, meta, fingerprint = load_snapshot(component.config.path)
    if meta["n"] != component.num_neurons:
        raise ValueError("保存配線の細胞数が設定と一致しません")
    if getattr(component.config, "fingerprint", fingerprint) != fingerprint:
        raise ValueError("要求した配線識別子と一致しません")
    if component.layout is not None:
        groups = component.layout.ids_by("polarity")
        if not np.array_equal(
            np.sort(groups["excitatory"]), np.arange(meta["n_exc"])
        ) or not np.array_equal(
            np.sort(groups["inhibitory"]), np.arange(meta["n_exc"], meta["n"])
        ):
            raise ValueError("保存配線と実行時のE/I割当が一致しません")
    return arrays


@SPATIAL_MODELS.register("saved_snapshot")
class SnapshotSpace(BaseSpace):
    def generate(self):
        return _read(self)["coords"]


@CONNECTION_MODELS.register("saved_snapshot")
class SnapshotConnection(BaseConnection):
    def generate(self):
        arrays = _read(self)
        return arrays["rows"], arrays["cols"]


@WEIGHT_MODELS.register("baseline_pair_weights")
class BaselinePairWeights(BaseWeight):
    def generate(self, rows, cols):
        is_inhibitory = np.zeros(self.num_neurons, dtype=bool)
        is_inhibitory[self.layout.ids_by("polarity")["inhibitory"]] = True
        pair = is_inhibitory[rows].astype(int) * 2 + is_inhibitory[cols]
        weights = np.array(
            [self.config.ee, self.config.ei, self.config.ie, self.config.ii]
        )
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights[:2] < 0)
            or np.any(weights[2:] > 0)
        ):
            raise ValueError("E/Iの結合重みの符号または有限性が不正です")
        return weights[pair]


@DELAY_MODELS.register("saved_arrival_delay")
class SnapshotDelay(BaseDelay):
    def generate(self, rows, cols):
        arrays = _read(self)
        if not np.array_equal(rows, arrays["rows"]) or not np.array_equal(
            cols, arrays["cols"]
        ):
            raise ValueError("遅延とCOOの並びが一致しません")
        return arrival_to_dendritic_ms(arrays["arrival_ms"], self.config.dt)
