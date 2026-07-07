"""ニューロンのグローバルインデックス割り当てを司る NetworkLayout (v2)。

旧来の group_info(dict) を置き換える。**正準グローバルID空間 0..N-1** を唯一の基準とし、
そこに 2つの直交する分割(=軸)を貼る:

- 構造層 (layer): グローバルID空間で**連続**するブロック。行列のブロック構造を規定し、解析軸①。
- 種別/population (type=E/I): 各グローバルIDに割り当てられるラベル。GeNN population を決め、解析軸②。
  `assignment="sequential"`(既定)は config 順の連番、`assignment="random"` は全体比率で
  ランダムに散らばる(seed 決定論的)。散在してもメンバーは**昇順ソート**で保持する
  (GeNN の発信ニューロン順格納と一致させ pull_synapse の順序前提を保つため)。

行列・座標はすべて正準グローバルID空間で生成される(層でブロック対角が自然に表現される)。
GeNN 配線や解析は「population のグローバルID集合」を `np.ix_`/fancy-index で切り出す。

割り当ては GeNN のビルドなしに `NetworkLayout.from_config(config)` だけで再現でき、
`assignment="random"` でも `config.simulation.seed` から決定論的に同一のレイアウトになる。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# random 割当用のシード派生オフセット。行列生成用 RNG(NetworkBuilder.self.rng)とは
# 別ストリームにし、E/I 割当と行列値の相関を避けるために seed に加算する定数。
_ASSIGN_SEED_OFFSET = 104729


@dataclass(frozen=True)
class PopulationSpec:
    """1つの GeNN population(=種別 type)が占めるグローバルID集合の記述子。

    global_indices は昇順ソート済み。連番割当なら連続、ランダム割当なら散在する。
    population ローカル index i は global_indices[i] に対応する。
    """

    name: str
    params: Any  # config.neurons[name] (Pydantic NeuronConfig)
    global_indices: np.ndarray

    @property
    def num(self) -> int:
        return len(self.global_indices)

    @property
    def mode(self) -> str:
        return getattr(self.params, "mode", None) or ""


@dataclass(frozen=True)
class LayerSpec:
    """1つの構造層(グローバルID空間で連続するブロック)の記述子。"""

    name: str
    start: int
    num: int

    @property
    def stop(self) -> int:
        return self.start + self.num

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)

    @property
    def global_indices(self) -> np.ndarray:
        return np.arange(self.start, self.stop)


@dataclass(frozen=True)
class LayoutPlan:
    """空間コンポーネント等がデータ/自身の config から導出しうるレイアウト構成案。

    両フィールドとも任意(None)。指定されたものだけが「明示的な config.layout が無い
    場合のフォールバック」として採用される(config.layout が常に優先)。

    - layers: 構造層 [(層名, ニューロン数), ...](グローバルID空間で連続する順)。
    - assignment: population(E/I)割当方式 "sequential" | "random"。
    """

    layers: Optional[List[Tuple[str, int]]] = None
    assignment: Optional[str] = None


class NetworkLayout:
    """正準グローバルID空間の上に population(type)軸と構造層(layer)軸を持つレイアウト。"""

    def __init__(self, populations: List[PopulationSpec], layers: List[LayerSpec]):
        self._pops: Dict[str, PopulationSpec] = {p.name: p for p in populations}
        self._pop_order: List[str] = [p.name for p in populations]
        self._layers: List[LayerSpec] = list(layers)
        self._layer_map: Dict[str, LayerSpec] = {l.name: l for l in layers}
        self._total: int = sum(p.num for p in populations)

        # --- ラベル表(長さ N。柔軟な再構築のため)---
        self._labels: Dict[str, np.ndarray] = {
            "population": np.empty(self._total, dtype=object),
            "mode": np.empty(self._total, dtype=object),
            "layer": np.empty(self._total, dtype=object),
        }
        for p in populations:
            self._labels["population"][p.global_indices] = p.name
            self._labels["mode"][p.global_indices] = p.mode
        for l in layers:
            self._labels["layer"][l.slice] = l.name

    # ==================================================================
    # 構築
    # ==================================================================
    @classmethod
    def from_config(cls, config) -> "NetworkLayout":
        """AppConfig から population(type)と構造層を決定論的に構築する。

        層(layers)と割当(assignment)の解決は **フィールド単位** で次の優先順:
          1. 明示的な config.layout の値
          2. 空間コンポーネントが提供するプラン(`BaseSpace.describe_layout`)
          3. 既定(assignment="sequential"、単一層 "all")

        空間コンポーネント経由でもデータ/自身の config から決定論的に導出されるため、
        GeNN ビルド無しの再構築性は保たれる(データファイルが必要な場合はそれを読む)。
        """
        pop_names = list(config.neurons.keys())
        counts = [config.neurons[name].num for name in pop_names]
        total = sum(counts)

        layout_cfg = getattr(config, "layout", None)
        cfg_layers = getattr(layout_cfg, "layers", None) if layout_cfg is not None else None
        cfg_assignment = getattr(layout_cfg, "assignment", None) if layout_cfg is not None else None

        # 空間コンポーネントが層/割当を提供する場合のプラン(config.layout が優先)。
        plan = cls._space_layout_plan(config, total)

        # --- 構造層 (layer) の解決 ---
        if cfg_layers:
            layer_pairs: Optional[List[Tuple[str, int]]] = [(lc.name, lc.num) for lc in cfg_layers]
        elif plan is not None and plan.layers:
            layer_pairs = list(plan.layers)
        else:
            layer_pairs = None

        layers: List[LayerSpec] = []
        if layer_pairs:
            offset = 0
            for name, num in layer_pairs:
                layers.append(LayerSpec(name=name, start=offset, num=int(num)))
                offset += int(num)
            if offset != total:
                raise ValueError(
                    f"layout.layers の合計 {offset} が総ニューロン数 {total} と一致しません。"
                )
        else:
            # 層未指定 → 全ニューロンを単一層とみなす
            layers.append(LayerSpec(name="all", start=0, num=total))

        # --- 種別 (type=population) 割当の解決 ---
        assignment = cfg_assignment
        if assignment is None and plan is not None:
            assignment = plan.assignment
        if assignment is None:
            assignment = "sequential"

        if assignment == "sequential":
            membership: Dict[str, np.ndarray] = {}
            offset = 0
            for name, c in zip(pop_names, counts):
                membership[name] = np.arange(offset, offset + c)
                offset += c
        elif assignment == "random":
            base_seed = getattr(config.simulation, "seed", None)
            assign_seed = None if base_seed is None else (int(base_seed) + _ASSIGN_SEED_OFFSET) % (2 ** 32)
            rng = np.random.RandomState(assign_seed)
            perm = rng.permutation(total)
            membership = {}
            offset = 0
            for name, c in zip(pop_names, counts):
                membership[name] = np.sort(perm[offset:offset + c])
                offset += c
        else:
            raise ValueError(f"未知の layout.assignment: {assignment!r} (sequential | random)")

        populations = [
            PopulationSpec(name=name, params=config.neurons[name], global_indices=membership[name])
            for name in pop_names
        ]
        return cls(populations, layers)

    @staticmethod
    def _space_layout_plan(config, total: int) -> Optional[LayoutPlan]:
        """空間コンポーネントが提供する層/割当プランを取得する(無ければ None)。

        空間モデルが未登録(モジュール未 import)の場合は None を返し、config.layout /
        既定へフォールバックする。これによりネットワークモジュールを import せずに
        `from_config` を呼ぶ再構築経路(既存の replot 等)の後方互換を保つ。
        """
        space_cfg = getattr(getattr(config, "network", None), "space", None)
        if space_cfg is None:
            return None
        # 遅延 import で循環を避ける(registry は low-level なので実質問題ないが明示的に)。
        from src.core.registry import SPATIAL_MODELS
        try:
            space_class = SPATIAL_MODELS.get(space_cfg.profile_name)
        except KeyError:
            # 空間モデル未登録 → プラン無し扱い(フォールバック)
            return None
        describe = getattr(space_class, "describe_layout", None)
        if describe is None:
            return None
        plan = describe(space_cfg, total)
        if plan is not None and not isinstance(plan, LayoutPlan):
            raise TypeError(
                f"{space_class.__name__}.describe_layout は LayoutPlan か None を返す必要が"
                f"あります (got {type(plan).__name__})。"
            )
        return plan

    # ==================================================================
    # population(type)軸 — 従来 API(散在集合に一般化)
    # ==================================================================
    @property
    def total_neurons(self) -> int:
        return self._total

    def names(self) -> List[str]:
        """population(=type)名の一覧(config 宣言順)。"""
        return list(self._pop_order)

    def items(self) -> Iterator[Tuple[str, PopulationSpec]]:
        for name in self._pop_order:
            yield name, self._pops[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._pop_order)

    def __contains__(self, name: str) -> bool:
        return name in self._pops

    def __getitem__(self, name: str) -> PopulationSpec:
        return self._pops[name]

    def global_indices(self, name: str) -> np.ndarray:
        """population のグローバルID集合(昇順ソート済み。連番/散在いずれも)。"""
        return self._pops[name].global_indices

    def num(self, name: str) -> int:
        return self._pops[name].num

    def mode_of(self, name: str) -> str:
        return self._pops[name].mode

    # ==================================================================
    # 構造層(layer)軸 — 新規
    # ==================================================================
    def layers(self) -> List[LayerSpec]:
        return list(self._layers)

    def layer_names(self) -> List[str]:
        return [l.name for l in self._layers]

    def layer_slice(self, name: str) -> slice:
        return self._layer_map[name].slice

    def layer_ids(self, name: str) -> np.ndarray:
        return self._layer_map[name].global_indices

    def ids_by_layer(self) -> Dict[str, np.ndarray]:
        """{層名: グローバルID配列}(連続)。層単位の解析に使う。"""
        return {l.name: l.global_indices for l in self._layers}

    # ==================================================================
    # ラベル / 柔軟グルーピング(解析用)
    # ==================================================================
    def labels(self) -> Dict[str, np.ndarray]:
        """{"population","mode","layer"} の各長さ N のラベル配列。"""
        return self._labels

    def ids_where(self, **filters: Any) -> np.ndarray:
        """指定ラベルがすべて一致するグローバルIDを昇順で返す。

        例: ids_where(layer="L2", mode="excitatory")
        """
        mask = np.ones(self._total, dtype=bool)
        for key, value in filters.items():
            if key not in self._labels:
                raise KeyError(f"未知のラベル: {key!r} (利用可能: {list(self._labels)})")
            mask &= (self._labels[key] == value)
        return np.nonzero(mask)[0]

    def ids_by_mode(self) -> Dict[str, np.ndarray]:
        """mode 接頭辞で興奮性/抑制性のグローバルIDを分類して返す。

        戻り値: {"excitatory": np.ndarray, "inhibitory": np.ndarray}(昇順)。
        mode は "excitatory" / "excitatory_b8" 等のバリアントを prefix 一致で許容する。
        """
        excitatory: List[np.ndarray] = []
        inhibitory: List[np.ndarray] = []
        for name in self._pop_order:
            p = self._pops[name]
            if p.mode.startswith("excitatory"):
                excitatory.append(p.global_indices)
            elif p.mode.startswith("inhibitory"):
                inhibitory.append(p.global_indices)
        exc = np.concatenate(excitatory) if excitatory else np.array([], dtype=np.int64)
        inh = np.concatenate(inhibitory) if inhibitory else np.array([], dtype=np.int64)
        return {"excitatory": np.sort(exc), "inhibitory": np.sort(inh)}

    # ==================================================================
    # global↔local 変換(散在集合対応の fancy-index。simulator/base_loader が委譲)
    # ==================================================================
    def split_global_to_local(self, global_arr: np.ndarray) -> Dict[str, np.ndarray]:
        """(total_neurons,) のグローバル配列を population 毎の配列に分割する。"""
        if len(global_arr) != self._total:
            raise ValueError(
                f"Input data size {len(global_arr)} does not match total neurons {self._total}"
            )
        return {name: global_arr[p.global_indices] for name, p in self.items()}

    def merge_local_to_global(
        self, local_dict: Dict[str, np.ndarray], dtype=np.float32
    ) -> np.ndarray:
        """population 毎の配列を (total_neurons,) のグローバル配列に結合する。"""
        global_data = np.zeros(self._total, dtype=dtype)
        for name, local_data in local_dict.items():
            global_data[self._pops[name].global_indices] = local_data
        return global_data

    def local_to_global(self, name: str, local_ids: np.ndarray) -> np.ndarray:
        """population ローカルインデックスをグローバルIDに変換する。

        local i は global_indices[i](昇順ソート済み)に対応する。
        """
        return self._pops[name].global_indices[np.asarray(local_ids)]
