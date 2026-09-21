"""親 run (`scripts/develop/` の出力) からネットワークを引き継ぐ。

引き継ぎは「同じ config を同じ seed で再ビルドし、保存済みの重みを流し込む」の 2 段。
結合構造そのものは再生成すれば完全に一致する (seed / backend / assignment / sparse が
`config.yaml` に実値で記録されているため) ので、保存が要るのは**重みだけ**。

**位置ではなく (pre, post) で対応づける**のがこのモジュールの一貫した方針。理由は 3 つ:

1. **本数が違う場合がある。** 切断後のネットワークは損傷前の部分集合なので、位置では
   そもそも並ばない (`align_subset_to_coo`)。これがいちばん本質的な理由。
2. **古い親 run がありうる。** `simulator.synapse_connectivity_coo()` が並びを
   `NetworkBuilder.global_coo()` へ揃える前に作られた run は、GeNN の格納順
   (シナプス集団ごとのブロック連結) で書かれている。本数が同じでどちらも正しいので、
   位置で対応づけると気付かないまま別のシナプスに重みが乗る。
3. 引き当てに失敗したとき「親 run と別のネットワークを再ビルドした」と言い切れる。

軸索幾何の検証 (`verify_axon_geometry` / `verify_geometry_alignment`) は並び順とは
無関係で、「同じ軸索が同じ経路を伸びたか」を結合集合の一致より強く確かめるためのもの。
"""
from __future__ import annotations

import numpy as np


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


def verify_axon_geometry(rebuilt, saved) -> dict[str, bool]:
    """再ビルドした軸索幾何が親 run の記録と一致するかを検証する。

    結合集合の一致より**強い**検証。同じ軸索が同じ経路を伸び、同じセグメントで
    同じ相手に触れたことまで確認できる。ここが通れば、ブリッジ通過判定を
    親 run の幾何ではなく再ビルドした幾何で行っても同じ答えになる。

    Raises:
        WeightRestoreError: 一致しない場合。
    """
    checks = {}
    for name in ("pre", "post", "contact_seg", "offsets", "seg_owner"):
        checks[name] = bool(np.array_equal(
            np.asarray(getattr(rebuilt, name)), np.asarray(getattr(saved, name))
        ))
    for name in ("seg_start", "seg_end", "contact_t"):
        checks[name] = bool(np.allclose(
            np.asarray(getattr(rebuilt, name)), np.asarray(getattr(saved, name))
        ))
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        raise WeightRestoreError(
            f"再ビルドした軸索幾何が親 run と一致しません (不一致: {', '.join(bad)})。"
            " 同じ config / seed から同じ軸索が再現できていません。"
        )
    return checks


def verify_geometry_alignment(geometry, coo) -> None:
    """軸索幾何の pre/post が COO と**位置一致**していることを確認する。

    `AxonGrowthTopology.generate_sparse()` は (pre, post) 昇順で返し、`global_coo()` も
    行優先ソート済みなので、両者は位置一致するはず。破れているとブリッジ通過判定が
    黙って別のシナプスに付くので、ここで止める。
    """
    if not (np.array_equal(np.asarray(geometry.pre, dtype=np.int64),
                           np.asarray(coo.row, dtype=np.int64))
            and np.array_equal(np.asarray(geometry.post, dtype=np.int64),
                               np.asarray(coo.col, dtype=np.int64))):
        raise WeightRestoreError(
            "軸索幾何の pre/post が global_coo() と位置一致していません。"
            " このままではブリッジ通過判定が別のシナプスに付きます。"
        )
