"""グラフ理論によるネットワーク構造の指標。

`connectivity.py` が「群と群のあいだにどれだけ結合があるか」を見るのに対し、
こちらは**個々のニューロンがネットワークの中でどんな位置にいるか** = ハブ性を見る。
損傷実験でどのニューロンを切るかを決め、切った後の再編成を測るのに使う。

層の契約は `connectivity.py` と同じ: 入力は COO (`row`, `col`, 任意で `weights`) と
`num_neurons`、モジュール指標だけ長さ N のラベル配列を追加で取る。matplotlib も
`src/models` も import しない。**`NetworkLayout` すら取らない** ——
ここで要るのは軸名ではなくラベル配列そのものなので、依存を 1 段減らしてある。

betweenness とクラスタ係数は `networkx` を使う。`requirements.txt` に
`networkx==3.6.1` がピン留めされている一方 `src/` からの import は従来ゼロだったので、
**直接依存になるのはこのファイルだけ**に閉じてある。participation coefficient と
within-module z-score は networkx にも無い (bctpy の領分) ので numpy で書く。
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# betweenness を厳密に解く上限ニューロン数。Brandes は O(N*E) なので、これを超えたら
# ピボットをサンプリングした推定値に落とす (黙って何時間も走るのを避ける)。
DEFAULT_MAX_EXACT_BETWEENNESS = 2000
DEFAULT_BETWEENNESS_PIVOTS = 200


def degree_table(row, col, num_neurons: int, weights=None) -> dict[str, np.ndarray]:
    """ニューロンごとの次数と strength (重み付き次数)。

    Returns:
        `out_degree` / `in_degree` / `degree` と、weights を渡した場合は
        `out_strength` / `in_strength`。いずれも長さ num_neurons。
    """
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    out_degree = np.bincount(row, minlength=num_neurons)[:num_neurons].astype(np.float64)
    in_degree = np.bincount(col, minlength=num_neurons)[:num_neurons].astype(np.float64)
    table = {
        "out_degree": out_degree,
        "in_degree": in_degree,
        "degree": out_degree + in_degree,
    }
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        table["out_strength"] = np.bincount(row, weights=w, minlength=num_neurons)[:num_neurons]
        table["in_strength"] = np.bincount(col, weights=w, minlength=num_neurons)[:num_neurons]
    return table


def module_participation(row, col, num_neurons: int, labels) -> dict[str, np.ndarray]:
    """participation coefficient P と within-module degree z-score。

    Guimerà & Amaral (2005) のハブ分類の 2 軸。モジュラーネットワークで
    「ハブかどうか」と「モジュール内のハブか、モジュールをまたぐハブか」を分ける
    標準的な指標で、**ブリッジ損傷の効果を予測するのに直接効く**
    (connector hub を切ればモジュール間が切れる)。

        P_i = 1 - Σ_m (k_im / k_i)^2      0 = 1 モジュールに閉じる, 1 に近い = 均等に分散
        z_i = (k_i^within - mean) / sd    自分のモジュール内での次数の標準化

    次数 0 のニューロンは P=0 / z=0 とする (NaN にしない —— 「孤立している」ことと
    「計算できない」ことを混同させないため)。モジュール内で次数がすべて同じ (sd=0) の
    ときも z=0 とする。

    Args:
        labels: 長さ num_neurons のモジュールラベル配列 (文字列でも数値でも可)。

    Returns:
        `participation` / `within_module_z` / `module_index` (長さ num_neurons)。
    """
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    labels = np.asarray(labels)
    if labels.size != num_neurons:
        raise ValueError(f"labels の長さ {labels.size} が num_neurons {num_neurons} と違います。")

    uniques, module_index = np.unique(labels, return_inverse=True)
    num_modules = uniques.size

    # 無向として扱う (どちらの向きの結合もそのニューロンをモジュールに繋いでいる)
    ends_self = np.concatenate([row, col])
    ends_other = np.concatenate([col, row])
    degree = np.bincount(ends_self, minlength=num_neurons)[:num_neurons].astype(np.float64)

    per_module = np.zeros((num_neurons, num_modules), dtype=np.float64)
    np.add.at(per_module, (ends_self, module_index[ends_other]), 1.0)

    participation = np.zeros(num_neurons, dtype=np.float64)
    connected = degree > 0
    ratios = per_module[connected] / degree[connected][:, None]
    participation[connected] = 1.0 - np.sum(ratios ** 2, axis=1)

    within = per_module[np.arange(num_neurons), module_index]
    z = np.zeros(num_neurons, dtype=np.float64)
    for m in range(num_modules):
        members = module_index == m
        vals = within[members]
        sd = float(np.std(vals))
        if sd > 0.0:
            z[members] = (vals - float(np.mean(vals))) / sd
    return {"participation": participation, "within_module_z": z, "module_index": module_index}


def classify_roles(participation, within_module_z, *, hub_z: float = 2.5,
                   provincial_p: float = 0.62, connector_p: float = 0.80) -> np.ndarray:
    """Guimerà-Amaral のハブ役割ラベル (文字列配列)。

    z >= hub_z がハブ。ハブのうち P <= provincial_p が provincial (モジュール内ハブ)、
    connector_p までが connector (モジュールをまたぐハブ)、それ以上が kinless。
    非ハブは P で peripheral / non_hub_connector に分ける。
    """
    p = np.asarray(participation, dtype=np.float64)
    z = np.asarray(within_module_z, dtype=np.float64)
    roles = np.full(p.size, "peripheral", dtype=object)
    is_hub = z >= hub_z
    roles[~is_hub & (p > provincial_p)] = "non_hub_connector"
    roles[is_hub & (p <= provincial_p)] = "provincial_hub"
    roles[is_hub & (p > provincial_p) & (p <= connector_p)] = "connector_hub"
    roles[is_hub & (p > connector_p)] = "kinless_hub"
    return roles.astype(str)


def _to_networkx(row, col, num_neurons: int, directed: bool):
    import networkx as nx
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(num_neurons))
    graph.add_edges_from(zip(np.asarray(row).tolist(), np.asarray(col).tolist()))
    return graph


def betweenness_centrality(row, col, num_neurons: int, *,
                           max_exact: int = DEFAULT_MAX_EXACT_BETWEENNESS,
                           pivots: int = DEFAULT_BETWEENNESS_PIVOTS,
                           seed: int = 0) -> tuple[np.ndarray, bool]:
    """媒介中心性 (有向グラフ、正規化済み)。

    N が `max_exact` を超えたらピボットをサンプリングした**推定値**に落とす。
    黙って何時間も走るより、推定値であることを返り値で明示するほうがよい。

    Returns:
        (values: 長さ num_neurons, exact: bool)
    """
    import networkx as nx
    graph = _to_networkx(row, col, num_neurons, directed=True)
    exact = num_neurons <= max_exact
    scores = nx.betweenness_centrality(
        graph, k=None if exact else min(pivots, num_neurons), seed=None if exact else seed
    )
    values = np.zeros(num_neurons, dtype=np.float64)
    for node, value in scores.items():
        values[int(node)] = float(value)
    return values, exact


def clustering_coefficients(row, col, num_neurons: int) -> np.ndarray:
    """無向・単純グラフとして見たクラスタ係数 (Watts-Strogatz)。"""
    import networkx as nx
    graph = _to_networkx(row, col, num_neurons, directed=False)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    scores = nx.clustering(graph)
    values = np.zeros(num_neurons, dtype=np.float64)
    for node, value in scores.items():
        values[int(node)] = float(value)
    return values


def component_sizes(row, col, num_neurons: int) -> dict[str, float]:
    """連結成分。**損傷でネットワークが割れたか**を最も直接に示す量。"""
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    adjacency = coo_matrix(
        (np.ones(row.size, dtype=np.int8), (row, col)), shape=(num_neurons, num_neurons)
    ).tocsr()
    out = {}
    for label, connection in (("weak", "weak"), ("strong", "strong")):
        count, membership = connected_components(adjacency, directed=True, connection=connection)
        sizes = np.bincount(membership)
        out[f"num_{label}_components"] = float(count)
        out[f"largest_{label}_component"] = float(sizes.max()) if sizes.size else 0.0
        out[f"largest_{label}_fraction"] = (
            float(sizes.max()) / num_neurons if num_neurons and sizes.size else float("nan")
        )
    return out


def graph_metrics(row, col, num_neurons: int, *, weights=None, labels=None,
                  include_betweenness: bool = True,
                  include_clustering: bool = True,
                  hub_z: float = 2.5,
                  max_exact: int = DEFAULT_MAX_EXACT_BETWEENNESS) -> dict[str, float]:
    """1 時点ぶんのグラフ指標を平坦な dict で返す (structure.csv の 1 行分)。

    重い指標 (betweenness / clustering) は個別に切れる。回復観察は記録点が多くなるので、
    大きな N では切って回すことを想定している。
    """
    degrees = degree_table(row, col, num_neurons, weights=weights)
    out_degree = degrees["out_degree"]
    metrics: dict[str, float] = {
        "num_synapses": float(np.asarray(row).size),
        "out_degree_mean": float(np.mean(out_degree)),
        "out_degree_sd": float(np.std(out_degree)),
        "out_degree_max": float(np.max(out_degree)) if out_degree.size else float("nan"),
        "in_degree_mean": float(np.mean(degrees["in_degree"])),
        "in_degree_sd": float(np.std(degrees["in_degree"])),
        "in_degree_max": float(np.max(degrees["in_degree"])) if out_degree.size else float("nan"),
        "unconnected_fraction": float(np.mean(degrees["degree"] == 0)),
    }
    if weights is not None:
        metrics["out_strength_mean"] = float(np.mean(degrees["out_strength"]))
        metrics["in_strength_mean"] = float(np.mean(degrees["in_strength"]))
    metrics.update(component_sizes(row, col, num_neurons))

    if labels is not None:
        part = module_participation(row, col, num_neurons, labels)
        roles = classify_roles(part["participation"], part["within_module_z"], hub_z=hub_z)
        metrics["participation_mean"] = float(np.mean(part["participation"]))
        metrics["participation_max"] = float(np.max(part["participation"]))
        metrics["within_module_z_max"] = float(np.max(part["within_module_z"]))
        for role in ("provincial_hub", "connector_hub", "kinless_hub"):
            metrics[f"num_{role}"] = float(np.count_nonzero(roles == role))

    if include_betweenness:
        values, exact = betweenness_centrality(row, col, num_neurons, max_exact=max_exact)
        metrics["betweenness_mean"] = float(np.mean(values))
        metrics["betweenness_max"] = float(np.max(values)) if values.size else float("nan")
        metrics["betweenness_exact"] = float(exact)
    if include_clustering:
        metrics["clustering_mean"] = float(np.mean(clustering_coefficients(row, col, num_neurons)))
    return metrics
