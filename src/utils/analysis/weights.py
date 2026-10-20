"""COO 上の per-synapse 量の E/I ブロック分解と、重みのブロック別統計。

分解の対象は重みに限らない。各シナプスに 1 つずつ値が付いた 1D 配列であれば何でもよく
(遅延・距離・重み)、`block_masks()` はそれらすべてに使われる。統計 (`weight_block_metrics`
など) だけが重み固有。

送信側 × 受信側の極性で結合を EE / EI / IE / II の 4 ブロックへ分ける。入力は常に
**COO (row, col, 値の 1D 配列)** で、密な (N, N) は受け取らない — ビルド以降の受け渡しは
COO 一本、という全体の規約に従う (`NetworkBuilder.global_coo()`)。COO は実結合しか
持たないので、「結合の無い箇所の 0 が統計に混ざる」問題は構造的に起きず、
かつての `connection_mask` 引数も要らない。極性は常に `layout.ids_by("polarity")` から取る。

    block_values(weights, row, col, layout) ─┬─▶ weight_block_metrics(blocks, wmax)
                                             └─▶ summarize_values(blocks[name])

統計の語彙が 2 つあるのは意図的で、それぞれ別の CSV の列に直結している:

- `weight_block_metrics` … metrics.csv の `weight_*` 列 (平均・飽和率)
- `summarize_values`     … weight_block_metrics.csv の列 (平均・最大・各しきい値以上の割合)
"""
from __future__ import annotations

import numpy as np

# 送信種別 × 受信種別のブロック名 (描画・集計で順序を揃えるための正準順)
BLOCK_ORDER = ("EE", "EI", "IE", "II")

# 全結合をまとめた擬似ブロックの名前。分解結果では常に先頭に来る。
ALL_BLOCK = "ALL"


# ======================================================================================
# ブロック分解
# ======================================================================================

def excitatory_flags(layout, total_neurons: int) -> np.ndarray:
    """グローバルID -> 興奮性なら True の bool 配列を返す。"""
    ids = layout.ids_by("polarity")
    flags = np.zeros(total_neurons, dtype=bool)
    flags[np.asarray(ids.get("excitatory", []), dtype=np.int64)] = True
    return flags


def block_masks(row: np.ndarray, col: np.ndarray, is_exc: np.ndarray) -> dict[str, np.ndarray]:
    """各シナプスが EE / EI / IE / II のどれかを示すブールマスクを返す (COO 用)。"""
    src_exc = is_exc[np.asarray(row, dtype=np.int64)]
    tgt_exc = is_exc[np.asarray(col, dtype=np.int64)]
    return {
        "EE": src_exc & tgt_exc,
        "EI": src_exc & ~tgt_exc,
        "IE": ~src_exc & tgt_exc,
        "II": ~src_exc & ~tgt_exc,
    }


def synapse_distances(coords: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """各シナプスの「細胞体から細胞体までの直線距離」[um] を返す (COO と index 整合)。

    XY 平面への投影で測る。空間モデルが 3 列返す場合でも、結合確率も遅延もこの図も
    2D 平面上の距離で定義されているため。軸索の実際の経路長ではないことに注意
    (`axon_growth` の経路長は `AxonGrowthTopology` の `axon_length` 軸が持つ)。
    """
    coords = np.asarray(coords, dtype=np.float64)
    src = coords[np.asarray(row, dtype=np.int64), :2]
    tgt = coords[np.asarray(col, dtype=np.int64), :2]
    return np.linalg.norm(src - tgt, axis=1)


def block_values(
    weights: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    layout,
) -> dict[str, np.ndarray]:
    """COO 形式の重みを {ブロック名: 1D の値配列} へ分解する。計算量は O(nnz)。

    返るのは `ALL` + BLOCK_ORDER の 5 キー。`ALL` は全結合をまとめた擬似ブロック。

    Args:
        weights: 各シナプスの重み (1D)
        row, col: 各シナプスの送信/受信グローバルID (1D, weights と index 整合)
        layout: NetworkLayout (E/I の分類は polarity 軸から取る)
    """
    values = np.asarray(weights)
    src = np.asarray(row, dtype=np.int64)
    tgt = np.asarray(col, dtype=np.int64)
    if not (values.size == src.size == tgt.size):
        raise ValueError("weights / row / col の長さが一致しません。")

    if values.size == 0:
        empty = values.reshape(-1)
        return {name: empty for name in (ALL_BLOCK, *BLOCK_ORDER)}

    is_exc = excitatory_flags(layout, layout.total_neurons)
    masks = block_masks(src, tgt, is_exc)
    blocks = {ALL_BLOCK: values.reshape(-1)}
    for name in BLOCK_ORDER:
        blocks[name] = values[masks[name]]
    return blocks


# ======================================================================================
# 統計
# ======================================================================================

def summarize_values(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "mean": np.nan,
            "max": np.nan,
            "nonzero_fraction": np.nan,
            "ge_0p5_fraction": np.nan,
            "ge_0p9_fraction": np.nan,
            "at_1_fraction": np.nan,
        }
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "nonzero_fraction": float(np.mean(values > 0.0)),
        "ge_0p5_fraction": float(np.mean(values >= 0.5)),
        "ge_0p9_fraction": float(np.mean(values >= 0.9)),
        "at_1_fraction": float(np.mean(values >= 0.999)),
    }


def _weight_block_stats(block: np.ndarray, wmax: float, at_max_tolerance: float) -> dict[str, float]:
    block = np.asarray(block, dtype=np.float64)
    if block.size == 0:
        return {
            "mean": np.nan,
            "nonzero_fraction": np.nan,
            "at_max_fraction": np.nan,
        }
    return {
        "mean": float(np.mean(block)),
        "nonzero_fraction": float(np.mean(block > 0.0)),
        "at_max_fraction": float(np.mean(block >= (wmax - at_max_tolerance))),
    }


def weight_block_metrics(
    blocks: dict[str, np.ndarray],
    wmax: float,
    at_max_tolerance: float = 1e-3,
) -> dict[str, float]:
    """分解済みのブロックから metrics.csv 用の `weight_*` 列を作る。

    `blocks` は `block_values` の返り値。
    """
    all_stats = _weight_block_stats(blocks[ALL_BLOCK], wmax=wmax, at_max_tolerance=at_max_tolerance)
    metrics = {
        "weight_mean": all_stats["mean"],
        "weight_nonzero_fraction": all_stats["nonzero_fraction"],
        "weight_at_max_fraction": all_stats["at_max_fraction"],
    }
    for name in BLOCK_ORDER:
        stats = _weight_block_stats(blocks[name], wmax=wmax, at_max_tolerance=at_max_tolerance)
        key = name.lower()
        metrics[f"weight_{key}_mean"] = stats["mean"]
        metrics[f"weight_{key}_at_max_fraction"] = stats["at_max_fraction"]
    return metrics


def compute_block_metrics(
    hour: float,
    weights: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    layout,
    previous_weights: np.ndarray | None = None,
) -> list[dict[str, float | str]]:
    """weight_block_metrics.csv 用の行 (1 記録時刻 × 5 ブロック) を作る。

    `weights` / `previous_weights` は同じ (row, col) 上の値ベクトル。結合構造は
    シミュレーション中に変わらないので、記録間で row/col を取り直す必要はない。
    """
    blocks = block_values(weights, row, col, layout)
    if previous_weights is None:
        deltas = None
    else:
        # 「ブロックを取ってから差を取る」と「差を取ってからブロックを取る」は同値。
        deltas = block_values(np.asarray(weights) - np.asarray(previous_weights),
                              row, col, layout)

    rows: list[dict[str, float | str]] = []
    for name, values in blocks.items():
        entry: dict[str, float | str] = {"hour": hour, "block": name.lower()}
        entry.update(summarize_values(values))
        entry["mean_delta_from_previous"] = (
            np.nan if deltas is None else float(np.mean(deltas[name]))
        )
        rows.append(entry)
    return rows
