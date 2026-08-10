"""E/I ブロック分解と、重みのブロック別統計。

送信側 × 受信側の極性で結合を EE / EI / IE / II の 4 ブロックに分ける操作は、密な (N, N)
行列を持つ場合と COO (row, col) を持つ場合の 2 通りある。**分解と統計は分離**してあり、
どちらの持ち方でも分解の結果は同じ `{ブロック名: 1D の値配列}` になるので、下流の統計は
形式を意識しない。極性は常に `layout.ids_by("polarity")` から取る。

    密: block_values(weights, layout, connection_mask) ─┐
                                                        ├─▶ weight_block_metrics(blocks, wmax)
    疎: block_values_coo(weights, row, col, layout) ────┘   summarize_values(blocks[name])

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


def block_id_pairs(layout) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{ブロック名: (送信側グローバルID, 受信側グローバルID)} を返す (密行列用)。"""
    ids = layout.ids_by("polarity")
    exc = np.asarray(ids.get("excitatory", []), dtype=np.int32)
    inh = np.asarray(ids.get("inhibitory", []), dtype=np.int32)
    everything = np.arange(layout.total_neurons, dtype=np.int32)
    return {
        ALL_BLOCK: (everything, everything),
        "EE": (exc, exc),
        "EI": (exc, inh),
        "IE": (inh, exc),
        "II": (inh, inh),
    }


def block_values(
    weights: np.ndarray,
    layout,
    connection_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """密な重み行列を {ブロック名: 1D の値配列} へ分解する。

    `connection_mask` を渡すと、結合が存在する要素だけを残す (全結合でない構成では、
    結合の無い箇所の 0 が統計に混ざるのを防ぐ)。

    dtype は変換しない。float32 の行列を渡せば float32 のまま返るので、平均の累積精度は
    呼び出し側の統計関数の責任になる (`weight_block_metrics` は float64 へ上げる)。
    """
    matrix = np.asarray(weights)
    mask = None if connection_mask is None else np.asarray(connection_mask) != 0

    blocks: dict[str, np.ndarray] = {}
    for name, (src_ids, tgt_ids) in block_id_pairs(layout).items():
        if name == ALL_BLOCK:
            # 全体は np.ix_ で取り直さずに行列そのものを使う (N×N のコピーを避ける)。
            block = matrix
            block_mask = mask
        else:
            block = matrix[np.ix_(src_ids, tgt_ids)]
            block_mask = None if mask is None else mask[np.ix_(src_ids, tgt_ids)]
        blocks[name] = block.reshape(-1) if block_mask is None else block[block_mask]
    return blocks


def block_values_coo(
    weights: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    layout,
) -> dict[str, np.ndarray]:
    """COO 形式の重みを {ブロック名: 1D の値配列} へ分解する。

    `block_values` の疎版で、計算量は O(nnz)。COO は実結合のみを持つため、密版に
    `connection_mask` を渡したのと等価な結果になる (結合が無い箇所の 0 は最初から
    含まれない)。密な (N,N) を確保できない大規模ネットワーク用。

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

    `blocks` は `block_values` または `block_values_coo` の返り値。どちらを渡しても
    返すキーは同じなので、疎/密で列が変わることはない。
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
    layout,
    connection_mask: np.ndarray | None = None,
    previous_weights: np.ndarray | None = None,
) -> list[dict[str, float | str]]:
    """weight_block_metrics.csv 用の行 (1 記録時刻 × 5 ブロック) を作る。"""
    blocks = block_values(weights, layout, connection_mask=connection_mask)
    if previous_weights is None:
        deltas = None
    else:
        # 「ブロックを取ってから差を取る」と「差を取ってからブロックを取る」は同値。
        deltas = block_values(np.asarray(weights) - np.asarray(previous_weights),
                              layout, connection_mask=connection_mask)

    rows: list[dict[str, float | str]] = []
    for name, values in blocks.items():
        row: dict[str, float | str] = {"hour": hour, "block": name.lower()}
        row.update(summarize_values(values))
        row["mean_delta_from_previous"] = (
            np.nan if deltas is None else float(np.mean(deltas[name]))
        )
        rows.append(row)
    return rows
