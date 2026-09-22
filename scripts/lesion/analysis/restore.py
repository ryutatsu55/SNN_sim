"""再ビルドしたネットワークが親 run と同じものかを検証する。

引き継ぎは「同じ config を同じ seed で再ビルドし、保存済みの重みを流し込む」の 2 段。
結合構造そのものは再生成すれば完全に一致する (seed / backend / assignment / sparse が
`config.yaml` に実値で記録されているため) ので、保存が要るのは**重みだけ**。

重みを載せ直す計算そのもの (`align_subset_to_coo` / `align_saved_to_coo`) は
**純粋な COO の索引計算**なので `src/utils/analysis/weights.py` にある。ここに残るのは
損傷実験に固有の検証 —— 「同じ軸索が同じ経路を伸びたか」を結合集合の一致より強く
確かめるもの。
"""
from __future__ import annotations

import numpy as np

from src.utils.analysis.weights import WeightRestoreError


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
