"""損傷 (切断) セレクタ。

**4 種類すべてが「M 本のシナプスに対する bool マスク」を返す**、という 1 点だけで
統一してある。ハブ基準もモジュール間もブリッジも、結局は「どのシナプスを消すか」でしか
ないので、合成 (and / or) が自明になり、マニフェストへの記録も 1 本の配列で済む。

CLI の spec 文法は `NAME:key=value,key=value`。未知のキーは即エラーにする ——
`top` を `topk` と書いて黙って全件切る、が最悪の失敗なので。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.utils.analysis.axons import synapse_crossed_parts
from src.utils.analysis.connectivity import bridge_part_indices
from src.utils.analysis.graph import (betweenness_centrality, degree_table,
                                      module_participation)


@dataclass(frozen=True)
class LesionContext:
    """セレクタが判断に使ってよいもの。すべて**切断前**の状態を指す。"""
    coo: Any                      # GlobalCOO
    layout: Any                   # NetworkLayout
    area: Any = None              # ダックタイピング (parts / part_names / part_allows_soma)
    geometry: Any = None          # AxonGeometry (pre/post は coo と位置一致していること)
    coords: np.ndarray | None = None


@dataclass(frozen=True)
class LesionSelection:
    cut: np.ndarray               # bool (M,)
    label: str                    # ディレクトリ名用の短い識別子
    description: str              # 人が読む 1 行
    detail: dict = field(default_factory=dict)   # 選択根拠。lesion.json へそのまま入る


class HubSelector:
    """ハブニューロンを選び、その入出力シナプスを切る。"""

    METRICS = ("out_degree", "in_degree", "degree", "out_strength", "in_strength",
               "betweenness", "participation", "within_module_z")

    def __init__(self, metric: str = "participation", top: int = 1,
                 direction: str = "out", axis: str = "module"):
        if metric not in self.METRICS:
            raise ValueError(f"未知の hub metric: {metric!r} (使えるのは {self.METRICS})")
        if direction not in ("out", "in", "both"):
            raise ValueError(f"未知の direction: {direction!r} (out | in | both)")
        self.metric, self.top, self.direction, self.axis = metric, int(top), direction, axis

    def select(self, ctx: LesionContext) -> LesionSelection:
        coo = ctx.coo
        n = int(coo.shape[0])
        row = np.asarray(coo.row, dtype=np.int64)
        col = np.asarray(coo.col, dtype=np.int64)

        if self.metric in ("participation", "within_module_z"):
            labels = ctx.layout.labels(self.axis)
            values = module_participation(row, col, n, labels)[
                "participation" if self.metric == "participation" else "within_module_z"]
        elif self.metric == "betweenness":
            values, _exact = betweenness_centrality(row, col, n)
        else:
            values = degree_table(row, col, n, weights=coo.weights)[self.metric]

        chosen = np.argsort(values)[::-1][:self.top]
        mask = np.zeros(row.size, dtype=bool)
        if self.direction in ("out", "both"):
            mask |= np.isin(row, chosen)
        if self.direction in ("in", "both"):
            mask |= np.isin(col, chosen)
        return LesionSelection(
            cut=mask,
            label=f"hub-{self.metric}-top{self.top}-{self.direction}",
            description=(f"{self.metric} 上位 {self.top} ニューロンの"
                         f"{ {'out':'出力','in':'入力','both':'入出力'}[self.direction] }シナプス"),
            detail={"metric": self.metric, "direction": self.direction,
                    "neuron_ids": chosen.tolist(),
                    "values": [float(values[i]) for i in chosen]},
        )


class SynapseSelector:
    """(pre, post) を明示指定して切る。"""

    def __init__(self, pairs):
        self.pairs = [(int(a), int(b)) for a, b in pairs]

    def select(self, ctx: LesionContext) -> LesionSelection:
        coo = ctx.coo
        n = int(coo.shape[1])
        keys = np.asarray(coo.row, dtype=np.int64) * n + np.asarray(coo.col, dtype=np.int64)
        order = np.argsort(keys)
        sorted_keys = keys[order]

        mask = np.zeros(keys.size, dtype=bool)
        missing = []
        for pre, post in self.pairs:
            key = pre * n + post
            pos = int(np.searchsorted(sorted_keys, key))
            if pos < sorted_keys.size and sorted_keys[pos] == key:
                mask[order[pos]] = True
            else:
                missing.append((pre, post))
        if missing:
            # 存在しないペアを黙って無視すると「何も切らなかった」が静かに通る。
            raise ValueError(f"指定されたシナプスが存在しません: {missing[:10]}")
        return LesionSelection(
            cut=mask, label=f"synapses-{len(self.pairs)}",
            description=f"明示指定した {len(self.pairs)} 本のシナプス",
            detail={"pairs": self.pairs},
        )


class BetweenGroupSelector:
    """指定軸で異なるグループに属する pre/post を持つシナプスを切る (トポロジー的定義)。"""

    def __init__(self, axis: str = "module", pairs=None):
        self.axis = axis
        self.pairs = [tuple(sorted(p)) for p in pairs] if pairs else None

    def select(self, ctx: LesionContext) -> LesionSelection:
        labels = np.asarray(ctx.layout.labels(self.axis))
        src = labels[np.asarray(ctx.coo.row, dtype=np.int64)]
        tgt = labels[np.asarray(ctx.coo.col, dtype=np.int64)]
        mask = src != tgt
        if self.pairs is not None:
            wanted = np.zeros(mask.size, dtype=bool)
            for a, b in self.pairs:
                wanted |= ((src == a) & (tgt == b)) | ((src == b) & (tgt == a))
            mask &= wanted
        pair_label = "all" if self.pairs is None else "+".join(f"{a}-{b}" for a, b in self.pairs)
        return LesionSelection(
            cut=mask, label=f"between-{self.axis}-{pair_label}",
            description=f"{self.axis} 軸で異なるグループをまたぐシナプス ({pair_label})",
            detail={"axis": self.axis, "pairs": self.pairs},
        )


class BridgeSelector:
    """**軸索が実際に通った**ブリッジで切る (幾何的定義)。

    `BetweenGroupSelector` とは別物である点に注意。モジュール内で完結しつつブリッジに
    出入りする軸索が存在するので、両者の本数は一致しない (実測: ブリッジ通過 1069 本に
    対しモジュール間は別勘定)。どちらを「モジュール間切断」と呼ぶかは主張に直結するので、
    lesion.json には**両方の本数**を残すこと。
    """

    KINDS = ("any", "inter_cluster", "intra_cluster")

    def __init__(self, parts=None, kind: str = "any", samples: int | None = None):
        if kind not in self.KINDS:
            raise ValueError(f"未知の bridge kind: {kind!r} (使えるのは {self.KINDS})")
        self.parts = list(parts) if parts else None
        self.kind = kind
        self.samples = samples

    def _target_indices(self, area) -> np.ndarray:
        indices = bridge_part_indices(area)
        names = list(area.part_names)
        if self.parts is not None:
            unknown = [p for p in self.parts if p not in names]
            if unknown:
                raise ValueError(f"part 名が見つかりません: {unknown} (part_names={names})")
            wanted = set(self.parts)
            return np.array([i for i in indices if names[int(i)] in wanted], dtype=np.int64)
        # 接頭辞での分類は hierarchical_modular_grid 固有の命名なので、汎用側
        # (analysis/axons.py) ではなくここ (実験側) で判定する。
        if self.kind == "inter_cluster":
            return np.array([i for i in indices if str(names[int(i)]).startswith("BX")],
                            dtype=np.int64)
        if self.kind == "intra_cluster":
            return np.array([i for i in indices if str(names[int(i)]).startswith("BC")],
                            dtype=np.int64)
        return indices

    def select(self, ctx: LesionContext) -> LesionSelection:
        if ctx.geometry is None:
            raise ValueError(
                "ブリッジ切断には軸索幾何が要ります (connection: axon_growth 系の run のみ)。"
            )
        if ctx.area is None:
            raise ValueError("ブリッジ切断には area が要ります。")
        targets = self._target_indices(ctx.area)
        if targets.size == 0:
            raise ValueError(f"条件に合うブリッジ part がありません (kind={self.kind}, parts={self.parts})")

        kwargs = {} if self.samples is None else {"samples": self.samples}
        crossed = synapse_crossed_parts(ctx.geometry, ctx.area, targets, **kwargs)
        mask = crossed.any(axis=1)
        names = list(ctx.area.part_names)
        per_part = {str(names[int(p)]): int(crossed[:, j].sum()) for j, p in enumerate(targets)}
        label = f"bridge-{self.kind}" if self.parts is None else "bridge-" + "+".join(self.parts)
        return LesionSelection(
            cut=mask, label=label,
            description=f"軸索が通ったブリッジ ({self.kind}) の結合 {int(mask.sum())} 本",
            detail={"kind": self.kind, "parts": [str(names[int(p)]) for p in targets],
                    "per_part_counts": per_part},
        )


_SPEC_PATTERN = re.compile(r"^(?P<name>[a-z_]+)(?::(?P<args>.*))?$")


def _parse_args(text: str | None) -> dict[str, str]:
    if not text:
        return {}
    out = {}
    for item in text.split(","):
        if "=" not in item:
            raise ValueError(f"spec の引数は key=value の形にしてください: {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def parse_cut_spec(spec: str):
    """`"bridge:kind=inter_cluster"` のような CLI 文字列をセレクタにする。"""
    match = _SPEC_PATTERN.match(spec.strip())
    if match is None:
        raise ValueError(f"切断 spec を解釈できません: {spec!r} (形式: NAME:key=value,...)")
    name = match.group("name")
    args = _parse_args(match.group("args"))

    def take(allowed):
        unknown = set(args) - set(allowed)
        if unknown:
            raise ValueError(f"{name} に未知のキー: {sorted(unknown)} (使えるのは {sorted(allowed)})")

    if name == "hub":
        take({"metric", "top", "direction", "axis"})
        return HubSelector(metric=args.get("metric", "participation"),
                           top=int(args.get("top", 1)),
                           direction=args.get("direction", "out"),
                           axis=args.get("axis", "module"))
    if name == "bridge":
        take({"kind", "parts", "samples"})
        return BridgeSelector(parts=args["parts"].split("+") if "parts" in args else None,
                              kind=args.get("kind", "any"),
                              samples=int(args["samples"]) if "samples" in args else None)
    if name == "between":
        take({"axis", "pairs"})
        pairs = None
        if "pairs" in args:
            pairs = [tuple(p.split("-", 1)) for p in args["pairs"].split("+")]
        return BetweenGroupSelector(axis=args.get("axis", "module"), pairs=pairs)
    if name == "synapses":
        take({"pairs", "file"})
        if "file" in args:
            import csv
            with open(args["file"], newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            return SynapseSelector([(r["pre"], r["post"]) for r in rows])
        return SynapseSelector([tuple(p.split("-", 1)) for p in args["pairs"].split("+")])
    raise ValueError(f"未知のセレクタ名: {name!r} (hub | bridge | between | synapses)")


def combine(selections, mode: str = "or") -> LesionSelection:
    """複数のセレクタ結果を合成する。"""
    if not selections:
        raise ValueError("セレクタが 1 つも指定されていません。")
    if len(selections) == 1:
        return selections[0]
    if mode not in ("or", "and"):
        raise ValueError(f"未知の合成モード: {mode!r} (or | and)")
    cut = selections[0].cut.copy()
    for sel in selections[1:]:
        cut = (cut | sel.cut) if mode == "or" else (cut & sel.cut)
    return LesionSelection(
        cut=cut,
        label=f"_{mode}_".join(s.label for s in selections),
        description=f" {mode} ".join(s.description for s in selections),
        detail={"combine": mode,
                "parts": [{"label": s.label, "num_cut": int(s.cut.sum()), **s.detail}
                          for s in selections]},
    )
