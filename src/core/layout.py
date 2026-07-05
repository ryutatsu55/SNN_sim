"""ニューロンのグローバルインデックス割り当てを司る NetworkLayout。

旧来の group_info(dict) を置き換える。config のニューロン宣言順に**連番**で
グローバルインデックスを割り当てる決定論的レイアウトを表現し、
global↔local の変換や興奮性/抑制性の分類といった共通処理を集約する。

割り当ては RandomState を一切消費しないため、GeNN のビルドなしに
`NetworkLayout.from_config(config)` だけで同一のレイアウトを再現できる。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np


@dataclass(frozen=True)
class GroupSpec:
    """1ニューロングループが占めるグローバルインデックス区間の記述子。"""

    name: str
    start: int
    num: int
    params: Any  # config.neurons[name] (Pydantic NeuronConfig)

    @property
    def stop(self) -> int:
        return self.start + self.num

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def global_indices(self) -> np.ndarray:
        return np.arange(self.start, self.stop)

    @property
    def mode(self) -> str:
        return getattr(self.params, "mode", None) or ""


class NetworkLayout:
    """config のニューロン順に連番で割り当てられたグローバルインデックスのレイアウト。"""

    def __init__(self, groups: List[GroupSpec]):
        self._groups: Dict[str, GroupSpec] = {g.name: g for g in groups}
        self._order: List[str] = [g.name for g in groups]
        self._total: int = sum(g.num for g in groups)

    @classmethod
    def from_config(cls, config) -> "NetworkLayout":
        """AppConfig の neurons を宣言順に走査し、連番でレイアウトを構築する。"""
        groups: List[GroupSpec] = []
        offset = 0
        for name, params in config.neurons.items():
            num = params.num
            groups.append(GroupSpec(name=name, start=offset, num=num, params=params))
            offset += num
        return cls(groups)

    # --- 基本アクセサ ---
    @property
    def total_neurons(self) -> int:
        return self._total

    def names(self) -> List[str]:
        return list(self._order)

    def items(self) -> Iterator[Tuple[str, GroupSpec]]:
        for name in self._order:
            yield name, self._groups[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._order)

    def __contains__(self, name: str) -> bool:
        return name in self._groups

    def __getitem__(self, name: str) -> GroupSpec:
        return self._groups[name]

    def global_indices(self, name: str) -> np.ndarray:
        return self._groups[name].global_indices

    def slice_of(self, name: str) -> slice:
        return self._groups[name].slice

    def num(self, name: str) -> int:
        return self._groups[name].num

    # --- mode(興奮性/抑制性)分類。旧 get_group_ids を置換 ---
    def mode_of(self, name: str) -> str:
        return self._groups[name].mode

    def ids_by_mode(self) -> Dict[str, np.ndarray]:
        """mode 接頭辞で興奮性/抑制性のグローバルインデックスを分類して返す。

        戻り値: {"excitatory": np.ndarray, "inhibitory": np.ndarray}(昇順)。
        mode は "excitatory" / "excitatory_b8" 等のバリアントを prefix 一致で許容する。
        """
        excitatory: List[np.ndarray] = []
        inhibitory: List[np.ndarray] = []
        for name in self._order:
            g = self._groups[name]
            if g.mode.startswith("excitatory"):
                excitatory.append(g.global_indices)
            elif g.mode.startswith("inhibitory"):
                inhibitory.append(g.global_indices)
        exc = np.concatenate(excitatory) if excitatory else np.array([], dtype=np.int64)
        inh = np.concatenate(inhibitory) if inhibitory else np.array([], dtype=np.int64)
        return {"excitatory": np.sort(exc), "inhibitory": np.sort(inh)}

    # --- global↔local 変換(simulator / base_loader の重複を集約)---
    def split_global_to_local(self, global_arr: np.ndarray) -> Dict[str, np.ndarray]:
        """(total_neurons,) のグローバル配列を Population 毎の配列に分割する。"""
        if len(global_arr) != self._total:
            raise ValueError(
                f"Input data size {len(global_arr)} does not match total neurons {self._total}"
            )
        return {name: global_arr[g.slice] for name, g in self.items()}

    def merge_local_to_global(
        self, local_dict: Dict[str, np.ndarray], dtype=np.float32
    ) -> np.ndarray:
        """Population 毎の配列を (total_neurons,) のグローバル配列に結合する。"""
        global_data = np.zeros(self._total, dtype=dtype)
        for name, local_data in local_dict.items():
            global_data[self._groups[name].slice] = local_data
        return global_data

    def local_to_global(self, name: str, local_ids: np.ndarray) -> np.ndarray:
        """Population ローカルインデックスをグローバルインデックスに変換する(連番なので start+local)。"""
        return self._groups[name].start + np.asarray(local_ids)
