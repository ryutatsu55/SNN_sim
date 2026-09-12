"""任意の軸でグループ分けした**群間結合確率** — 層内 (対角) と層間 (非対角)。

`weights.py` が per-synapse の値を E/I の 4 ブロックへ分けるのに対し、こちらは**結合が
存在するかどうか**を数える。しかも軸を固定しない: `layout` の任意の軸 (`module` /
`layer` / `polarity` / …) でグループ分けし、送信グループ × 受信グループの K×K 表を作る。
`plot_connection_mask_coarse` が絵にしているものを、セルではなく**軸の値そのもの**で
集計して数値で返す版にあたる。

    p[i, j] = (i から j への実結合数) / (i から j へ**ありうる**ペア数)

「ありうるペア数」が要点で、グループの大きさが違っても比較できるのはこれで割るから。
非対角は `n_i · n_j`、対角は自己結合を除いて `n_i · (n_i − 1)`
(`include_self_pairs=True` で `n_i²` に切り替わる)。

**方向つき**であることに注意 — `row` が送信、`col` が受信なので `p[i, j] != p[j, i]` が
普通に起きる。層内 = 対角、層間 = 非対角。

後半 (`bridge_hop_matrix` / `hop_connection_probability`) は、その K×K 表を**経由する
ブリッジの本数**でまとめ直す。K×K を対角 / 非対角に割ると「同一モジュール」と「それ以外」
の 2 つにしかならないが、モジュール構造では非対角の中身が一様でない — 隣のモジュールと
対角のモジュールでは通るブリッジの本数が違う。エリアの part の重なりからモジュール隣接
グラフを組み、最短ブリッジ本数ごとにプールし直したのがホップ別の表。

入力は他の解析と同じく **COO (row, col)** のみ。値の配列は要らない (結合の有無だけを見る)
ので、`NetworkBuilder.global_coo()` の row/col をそのまま渡せる。エリアはダックタイピング
で読む (`parts` / `part_names` / `part_allows_soma`) ので、ここは `src/models` も
matplotlib も import しない (`src/utils/analysis` の規約)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupConnectivity:
    """軸でグループ分けした群間結合確率の集計結果。

    Attributes:
        axis: グループ分けに使った軸名。
        names: グループ名 (K 個、軸の値でソート済み)。
        sizes: (K,) 各グループのニューロン数。
        counts: (K, K) 実結合数。`counts[i, j]` は i → j。
        possible: (K, K) ありうるペア数。
        probability: (K, K) `counts / possible`。ペアが 0 の要素は NaN。
    """
    axis: str
    names: list[str]
    sizes: np.ndarray
    counts: np.ndarray
    possible: np.ndarray
    probability: np.ndarray

    # ------------------------------------------------------------------ 層内 / 層間

    @property
    def within(self) -> np.ndarray:
        """(K,) 各グループの**層内**結合確率 (対角)。"""
        return np.diag(self.probability)

    @property
    def between_mask(self) -> np.ndarray:
        """(K, K) 非対角 (= 層間) を示す bool 行列。"""
        return ~np.eye(len(self.names), dtype=bool)

    @property
    def within_probability(self) -> float:
        """層内をプールした結合確率 = Σ対角counts / Σ対角possible。

        **各グループの確率の単純平均ではない。** グループの大きさが揃わないのが普通
        (一様サンプリングなら偶然に任される) で、単純平均だと小さいグループが過剰に
        効いてしまう。「層内のペアを 1 つ引いたとき結合している確率」が欲しい量なので、
        分子と分母をそれぞれ足してから割る。
        """
        return _ratio(np.trace(self.counts), np.trace(self.possible))

    @property
    def between_probability(self) -> float:
        """層間をプールした結合確率。"""
        mask = self.between_mask
        return _ratio(self.counts[mask].sum(), self.possible[mask].sum())

    @property
    def segregation(self) -> float:
        """層内 / 層間の比。大きいほどモジュール構造がはっきりしている。

        層間が 0 なら inf (完全に分離した部分ネットワーク)。両方 0 なら NaN。
        """
        return _ratio(self.within_probability, self.between_probability)


def _ratio(numerator: float, denominator: float) -> float:
    """0 除算を NaN / inf として素直に返す (エラーにしない)。"""
    numerator, denominator = float(numerator), float(denominator)
    if denominator == 0.0:
        return np.nan if numerator == 0.0 else np.inf
    return numerator / denominator


def group_connection_probability(
    row: np.ndarray,
    col: np.ndarray,
    layout,
    axis: str,
    *,
    include_self_pairs: bool = False,
) -> GroupConnectivity:
    """軸でグループ分けした群間結合確率 (K×K) を求める。計算量は O(nnz + K²)。

    Args:
        row, col: 各結合の送信/受信グローバルID (1D)。値の配列は要らない。
        layout: NetworkLayout。`axis` のラベルを読む。
        axis: グループ分けに使う軸名 (`module` / `layer` / `polarity` など)。
        include_self_pairs: 対角の分母に自己ペア (i→i) を含めるか。既定 False。
            ほとんどのコネクタが `allow_self_connections=False` なので、含めると
            **到達しえないペアで割る**ことになり層内確率が過小に出る。

    Raises:
        KeyError / ValueError: `axis` が layout に無い場合 (NetworkLayout 側が送出)。
    """
    labels = layout.labels(axis)
    # **ソートは軸のネイティブ dtype のまま行う。** 先に str へ落とすと数値軸で
    # "10" < "9" の辞書順になり、グループの対応が静かにずれる。文字列化は表示だけ。
    values = np.asarray(layout.values(axis, order="sorted"))
    names = [str(v) for v in values]
    k = len(names)

    # ラベル -> 0..K-1。values はソート済みなので searchsorted で引ける。
    group_of_id = np.searchsorted(values, labels)

    src = np.asarray(row, dtype=np.int64)
    tgt = np.asarray(col, dtype=np.int64)
    if src.size != tgt.size:
        raise ValueError("row / col の長さが一致しません。")

    counts = np.zeros((k, k), dtype=np.int64)
    if src.size:
        np.add.at(counts, (group_of_id[src], group_of_id[tgt]), 1)

    sizes = np.bincount(group_of_id, minlength=k).astype(np.int64)
    possible = np.outer(sizes, sizes)
    if not include_self_pairs:
        # 自己ペア (i→i) は各ニューロンに 1 つずつ。対角ブロックからだけ引く。
        possible = possible - np.diag(sizes)

    probability = np.divide(
        counts.astype(np.float64), possible.astype(np.float64),
        out=np.full((k, k), np.nan), where=possible > 0,
    )
    return GroupConnectivity(axis=axis, names=names, sizes=sizes,
                             counts=counts, possible=possible, probability=probability)


def format_group_connection_probability(
    result: GroupConnectivity,
    *,
    precision: int = 4,
) -> str:
    """`GroupConnectivity` を端末に出せる表へ整形する (matplotlib 非依存)。

    行が送信グループ、列が受信グループ。対角が層内、非対角が層間。
    最後にプールした層内/層間の確率と、その比 (segregation) を添える。
    """
    names = result.names
    longest = max([len(n) for n in names] + [1])
    width = max(longest, len("source \\ target"))
    # 列幅は「数値の桁」と「グループ名の長さ」の両方を満たす必要がある。名前だけで決めると
    # 数値が詰まり、桁だけで決めると polarity のような長い名前で列が潰れてくっつく。
    cell = max(precision + 3, longest + 2)

    lines = [
        f"Connection probability by '{result.axis}' axis  "
        f"(source -> target, {int(result.counts.sum())} synapses)",
        "  " + "source \\ target".ljust(width) + "".join(n.rjust(cell) for n in names)
        + "   n",
    ]
    for i, name in enumerate(names):
        row = "".join(
            ("  -".rjust(cell) if result.possible[i, j] == 0
             else f"{result.probability[i, j]:.{precision}f}".rjust(cell))
            for j in range(len(names))
        )
        lines.append("  " + name.ljust(width) + row + f"   {int(result.sizes[i])}")

    summary = [
        (f"within-{result.axis} (diagonal, pooled)", f"{result.within_probability:.{precision}f}"),
        (f"between-{result.axis} (off-diagonal)", f"{result.between_probability:.{precision}f}"),
        ("ratio within / between", f"{result.segregation:.2f}"),
    ]
    label_width = max(len(label) for label, _ in summary)
    lines.append("  " + "-" * (width + cell * len(names) + 4))
    lines.extend(f"  {label.ljust(label_width)} : {value}" for label, value in summary)
    return "\n".join(lines)


# ---------------------------------------------------------------- ブリッジ経由のホップ数

@dataclass(frozen=True)
class HopConnectivity:
    """グループ対を**経由するブリッジの本数**でまとめ直した結合確率。

    `GroupConnectivity` の K×K 表を、`hops[i, j]` (= i のモジュールから j のモジュールへ
    行くのに最低何本のブリッジを通るか) の値ごとにプールしたもの。分子も分母も足してから
    割るので、`within_probability` と同じ意味で「そのホップ数のペアを 1 つ引いたとき結合
    している確率」になる。`hops == 0` は同一モジュール内なので `within_probability` に一致する。

    Attributes:
        axis: 元になった軸名。
        names: グループ名 (K 個、`GroupConnectivity.names` と同じ並び)。
        hops: (K, K) グループ間の最短ブリッジ本数。`-1` は到達不能 (連結していない)。
        levels: (L,) 現れたホップ数 (昇順。到達不能 `-1` があれば末尾)。
        group_pairs: (L,) そのホップ数に属する**順序つきグループ対**の数。
        counts: (L,) 実結合数。
        possible: (L,) ありうるニューロン対の数。
        probability: (L,) `counts / possible`。
    """
    axis: str
    names: list[str]
    hops: np.ndarray
    levels: np.ndarray
    group_pairs: np.ndarray
    counts: np.ndarray
    possible: np.ndarray
    probability: np.ndarray


def bridge_part_indices(area) -> np.ndarray:
    """area の part のうち **ブリッジ** であるものの index を返す。

    ブリッジの定義はこのリポジトリでは 1 つだけ —— **`allow_soma: false` の part**。
    part 名の接頭辞 (`B` / `BC` / `BX`) は `modular_grid` 系に固有の命名なので、
    汎用の判定に使ってはいけない。この関数が定義の唯一の置き場所で、
    `bridge_hop_matrix()` と `src/utils/analysis/axons.py` の両方がここを見る。

    Raises:
        ValueError: area が part を持たない、または `allow_soma: false` の part が
            1 つも無い場合 (= モジュールとブリッジを区別できない)。
    """
    part_names = list(getattr(area, "part_names", []))
    if not part_names:
        raise ValueError(
            "ブリッジは composite area (parts を持つ領域) でしか区別できません。"
        )
    allows = list(getattr(area, "part_allows_soma", [True] * len(part_names)))
    indices = np.array([i for i, ok in enumerate(allows) if not ok], dtype=np.int64)
    if indices.size == 0:
        raise ValueError(
            "soma を置けない part (allow_soma: false) が 1 つもないので、モジュールと"
            " ブリッジを区別できません。modular_grid 系なら soma_in_bridge: false に"
            " してください。"
        )
    return indices


def part_overlap_graph(area, *, samples: int = 33) -> np.ndarray:
    """composite area の part 同士が**重なっているか**の (P, P) bool 行列。

    part は「境界箱が交わり、その交差箱の格子点のうち少なくとも 1 点が両方の `contains()`
    を満たす」とき重なっているとみなす。判定を交差箱の中だけで行うのが要点で、
    ブリッジがモジュールへ食い込む幅 (`bridge_overlap`) がどれだけ細くても、格子は
    その帯の中に敷かれるので取りこぼさない (格子密度が形の大きさに依存しない)。

    `area` はダックタイピングで読む (`parts` と、各 part の `bounds` / `contains`)。
    `src/utils/analysis` は `src/models` を import しない。
    """
    parts = list(getattr(area, "parts", []))
    p = len(parts)
    adjacency = np.zeros((p, p), dtype=bool)
    bounds = [np.asarray(part.bounds, dtype=np.float64) for part in parts]
    for i in range(p):
        for j in range(i + 1, p):
            lo = np.maximum(bounds[i][0], bounds[j][0])
            hi = np.minimum(bounds[i][1], bounds[j][1])
            if np.any(hi < lo):
                continue
            xs = np.linspace(lo[0], hi[0], samples)
            ys = np.linspace(lo[1], hi[1], samples)
            grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
            hit = bool(np.any(parts[i].contains(grid) & parts[j].contains(grid)))
            adjacency[i, j] = adjacency[j, i] = hit
    return adjacency


def bridge_hop_matrix(area, names, *, samples: int = 33) -> np.ndarray:
    """グループ間の**最短ブリッジ本数** (K, K) を area の part 構造から求める。

    part を「soma を置ける part (= モジュール)」と「置けない part (= ブリッジ)」に分け、
    重なりグラフの上で**ブリッジを何本通るか**を最小化する 0-1 BFS を回す。隣接
    モジュールは 1、対角モジュールは 2、同一モジュールは 0。連結していない相手は `-1`。

    Args:
        area: composite area (`parts` / `part_names` / `part_allows_soma` を読む)。
        names: グループ名の並び。`GroupConnectivity.names` をそのまま渡す。
        samples: 重なり判定の 1 辺あたり格子点数。

    Raises:
        ValueError: area が part を持たない、`allow_soma: false` の part が 1 つも無い
            (= ブリッジと呼べる part が無いので「経由した本数」が定義できない)、
            または `names` に part 名でないものが混ざっている場合。
    """
    part_names = list(getattr(area, "part_names", []))
    is_bridge = np.zeros(len(part_names), dtype=bool)
    is_bridge[bridge_part_indices(area)] = True

    index_of = {name: i for i, name in enumerate(part_names)}
    unknown = [n for n in names if n not in index_of]
    if unknown:
        raise ValueError(
            f"軸の値が part 名と対応しません: {', '.join(map(str, unknown))}。"
            f" part_names = {part_names}"
        )

    adjacency = part_overlap_graph(area, samples=samples)
    cost = is_bridge.astype(np.int64)          # part へ入るコスト (ブリッジだけ 1)
    neighbours = [np.nonzero(adjacency[u])[0] for u in range(len(part_names))]

    k = len(names)
    hops = np.full((k, k), -1, dtype=np.int64)
    for i, name in enumerate(names):
        dist = _zero_one_bfs(index_of[name], neighbours, cost)
        hops[i] = [dist[index_of[n]] for n in names]
    return hops


def _zero_one_bfs(source: int, neighbours, cost: np.ndarray) -> np.ndarray:
    """コスト 0/1 の辺 (ここでは「入る part がブリッジか」) 上の最短距離。未到達は -1。"""
    from collections import deque

    dist = np.full(len(cost), -1, dtype=np.int64)
    dist[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in neighbours[u]:
            candidate = dist[u] + cost[v]
            if dist[v] < 0 or candidate < dist[v]:
                dist[v] = candidate
                (queue.appendleft if cost[v] == 0 else queue.append)(v)
    return dist


def hop_connection_probability(
    result: GroupConnectivity,
    hops: np.ndarray,
) -> HopConnectivity:
    """`GroupConnectivity` をホップ数ごとにプールし直す。計算量は O(K²)。

    `group_connection_probability` が既に K×K の分子 (`counts`) と分母 (`possible`) を
    持っているので、ここは `hops` の値で足し合わせるだけ。**方向は潰れる** —
    `hops` は対称なので i→j と j→i が同じ段に入る。
    """
    hops = np.asarray(hops, dtype=np.int64)
    if hops.shape != result.counts.shape:
        raise ValueError(
            f"hops の形 {hops.shape} が結合確率の表 {result.counts.shape} と一致しません。"
        )
    # 到達不能 (-1) は「ホップ数が大きい」ではなく別枠なので、昇順の末尾へ回す。
    levels = sorted(set(int(v) for v in np.unique(hops)), key=lambda v: (v < 0, v))
    counts, possible, pairs = [], [], []
    for level in levels:
        mask = hops == level
        counts.append(int(result.counts[mask].sum()))
        possible.append(int(result.possible[mask].sum()))
        pairs.append(int(mask.sum()))
    counts = np.array(counts, dtype=np.int64)
    possible = np.array(possible, dtype=np.int64)
    probability = np.array(
        [_ratio(c, p) for c, p in zip(counts, possible)], dtype=np.float64
    )
    return HopConnectivity(
        axis=result.axis, names=list(result.names), hops=hops,
        levels=np.array(levels, dtype=np.int64),
        group_pairs=np.array(pairs, dtype=np.int64),
        counts=counts, possible=possible, probability=probability,
    )


def format_hop_connection_probability(
    result: HopConnectivity,
    *,
    precision: int = 4,
) -> str:
    """`HopConnectivity` を端末に出せる表へ整形する (matplotlib 非依存)。

    1 行が「ブリッジ n 本ぶん離れたグループ対」。`pairs` はニューロン対の総数 (= 分母)、
    `synapses` が実結合数 (= 分子)。`n/a` は経路が無い (連結していない) 対。
    """
    header = ("hops", "group pairs", "pairs", "synapses", "p")
    rows = []
    for i, level in enumerate(result.levels):
        rows.append((
            "n/a" if level < 0 else str(int(level)),
            str(int(result.group_pairs[i])),
            str(int(result.possible[i])),
            str(int(result.counts[i])),
            f"{result.probability[i]:.{precision}f}",
        ))
    widths = [max(len(h), *(len(r[c]) for r in rows)) for c, h in enumerate(header)]
    lines = [
        f"Connection probability by bridge hops ('{result.axis}' axis, "
        f"{len(result.names)} groups)",
        "  " + "  ".join(h.rjust(w) for h, w in zip(header, widths)),
        "  " + "  ".join("-" * w for w in widths),
    ]
    lines += ["  " + "  ".join(v.rjust(w) for v, w in zip(row, widths)) for row in rows]
    return "\n".join(lines)
