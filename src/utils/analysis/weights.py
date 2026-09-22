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


# ======================================================================================
# 保存済みの重みを別の COO へ載せ直す
#
# **位置ではなく (pre, post) で対応づける。** 理由は 3 つ:
#
# 1. **本数が違う場合がある。** 損傷後のネットワークは損傷前の部分集合なので、位置では
#    そもそも並ばない (`align_subset_to_coo`)。これがいちばん本質的な理由。
# 2. **古い run がありうる。** COO の並びをプロジェクト全体で 1 つに揃える前に作られた
#    記録は GeNN の格納順 (シナプス集団ごとのブロック連結) で書かれている。本数が同じで
#    どちらも正しいので、位置で対応づけると気付かないまま別のシナプスに重みが乗る。
# 3. 引き当てに失敗したとき「別のネットワークだ」と言い切れる。
#
# ここにあるのは、これが**純粋な COO の索引計算**だから。損傷実験が最初の利用者だが、
# 「保存した重みを別の結合集合へ載せる」こと自体は実験に依らない
# (`scripts/lesion/analysis/restore.py` に置いていた頃は、図がそれを import するために
#  `figures/ → analysis/` の辺ができていた)。
# ======================================================================================

class WeightRestoreError(RuntimeError):
    """重みを復元できない = 親 run と別のネットワークを再ビルドしてしまっている。"""


def align_subset_to_coo(saved_row, saved_col, saved_values,
                        target_row, target_col, n_cols: int) -> np.ndarray:
    """保存済み COO の値を、その**部分集合**である `(target_row, target_col)` へ引き当てる。

    `align_saved_to_coo` との違いは本数の一致を要求しないこと。損傷後のネットワーク
    (切断で減ったシナプス) に、損傷前に測った値を並べ直すのに使う —— 例えば
    baseline probe (Phase 1、全シナプス) の重みを、生き残ったシナプスの軌跡として
    post probe と同じ土俵に載せるとき。

    Raises:
        WeightRestoreError: 値の長さが合わない / 保存側に重複がある /
            target のシナプスが保存側に無い (= 部分集合ではない)。
    """
    # **int64 へ上げてから掛ける。** int32 のままだと N が大きいときに row*N+col が
    # 2^31 を超えて折り返り、別のペアが同じキーになって「一致した」ふりをする。
    key_saved = np.asarray(saved_row, dtype=np.int64) * n_cols + np.asarray(saved_col, dtype=np.int64)
    key_target = np.asarray(target_row, dtype=np.int64) * n_cols + np.asarray(target_col, dtype=np.int64)
    values = np.asarray(saved_values, dtype=np.float64).reshape(-1)

    if values.size != key_saved.size:
        raise WeightRestoreError(
            f"結合情報 ({key_saved.size} 本) と値 ({values.size} 本) の長さが一致しません。"
            " 対になっていないファイルを渡しています。"
        )

    order = np.argsort(key_saved, kind="stable")
    sorted_keys = key_saved[order]
    if sorted_keys.size > 1:
        duplicated = np.diff(sorted_keys) == 0
        if np.any(duplicated):
            examples = sorted_keys[:-1][duplicated][:5]
            raise WeightRestoreError(
                f"保存側に同じ (pre, post) が複数あります (キー例: {examples.tolist()})。"
            )

    pos = np.searchsorted(sorted_keys, key_target)
    np.clip(pos, 0, max(sorted_keys.size - 1, 0), out=pos)
    missing = (sorted_keys[pos] != key_target) if sorted_keys.size else np.ones(
        key_target.size, dtype=bool)
    if np.any(missing):
        idx = np.flatnonzero(missing)[:5]
        examples = list(zip(np.asarray(target_row)[idx].tolist(),
                            np.asarray(target_col)[idx].tolist()))
        raise WeightRestoreError(
            f"引き当て先のシナプスのうち {int(missing.sum())} 本が保存側にありません"
            f" (例: {examples})。部分集合になっていません。"
        )
    return values[order][pos]


def align_saved_to_coo(saved_row, saved_col, saved_values, coo) -> np.ndarray:
    """保存済み COO の値を、再ビルドした `coo` の並びへ並べ替える。

    Args:
        saved_row, saved_col: `connectivity.npz` の row / col。
        saved_values: `weights_{h}h.npz` の `data` (row/col と index 整合)。
        coo: 再ビルドした `GlobalCOO`。

    Returns:
        `coo.row` / `coo.col` と index 整合した float64 の重み。

    Raises:
        WeightRestoreError: 本数が違う / 値の長さが合わない / 保存側に重複がある /
            再ビルドしたシナプスが保存側に無い。いずれも「同じネットワークではない」の証拠。
    """
    saved_size = np.asarray(saved_row).size
    target_size = np.asarray(coo.row).size
    if saved_size != target_size:
        raise WeightRestoreError(
            f"シナプス本数が違います: 保存 {saved_size} 本 / 再ビルド {target_size} 本。"
            " 親 run の config.yaml をそのまま渡していますか"
            " (seed / backend / layout.assignment / network.sparse のどれかが変わると"
            " 別のネットワーク実現になります)。"
        )
    # 長さ一致 + 保存側に重複なし + 全キーが見つかる、で集合として同一が従う。
    return align_subset_to_coo(saved_row, saved_col, saved_values,
                               coo.row, coo.col, int(coo.shape[1]))
