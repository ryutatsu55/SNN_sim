"""軸索の折れ線から読む量。

`AxonGeometry` (`axon_geometry.npz`) が持つ「どの軸索がどこを通ったか」を、
**領域の part 単位**の通過情報へ変換する。損傷実験で「このブリッジを通った結合を
一斉に切る」を成立させるのがこのモジュールの唯一の目的。

`connectivity.py` と同じ層の契約を守る:
- matplotlib も `src/models` も import しない
- geometry と area は**ダックタイピング**で受ける
  (geometry: `pre` / `post` / `offsets` / `seg_start` / `seg_end` / `contact_seg` / `contact_t`、
   area: `parts` / `part_names` / `part_allows_soma`、各 part は `contains`)
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.utils.analysis.connectivity import bridge_part_indices

# 1 セグメントを何点でサンプルして part との交差を見るか。
# 実測 (axon_growth: seg 100 um / bridge 50 um、axon_growth_fine: seg 10 um / bridge 20 um)
# では 32 点で両方とも収束する (16 点だと粗いほうで 1069 本中 1 本取りこぼす)。
# **新しい幾何を使うときは同じ収束確認をすること** —— 必要なのは
# 「セグメント長 / (samples-1) << part の最小幅」。
DEFAULT_SEGMENT_SAMPLES = 32


def synapse_path_segments(geometry, index: int) -> tuple[np.ndarray, np.ndarray]:
    """シナプス `index` を作った軸索経路 (soma -> 接触点) の線分列を返す。

    範囲は `offsets[pre] .. contact_seg` で、**最終セグメントは `contact_t` で切る**。
    軸索全体を使ってはいけない: 接触より後に伸びた部分は、その結合とは無関係な別の
    part を通っていることがあり、それを拾うと「通っていないブリッジ」で切られる。

    Returns:
        (starts, ends) いずれも (k, 2) float64。
    """
    pre = int(geometry.pre[index])
    lo = int(geometry.offsets[pre])
    hi = int(geometry.contact_seg[index])
    starts = np.asarray(geometry.seg_start[lo:hi + 1], dtype=np.float64).copy()
    ends = np.asarray(geometry.seg_end[lo:hi + 1], dtype=np.float64).copy()
    if starts.shape[0] == 0:
        return starts, ends
    t = float(geometry.contact_t[index])
    ends[-1] = starts[-1] + t * (ends[-1] - starts[-1])
    return starts, ends


def synapse_crossed_parts(
    geometry,
    area,
    part_indices=None,
    *,
    samples: int = DEFAULT_SEGMENT_SAMPLES,
) -> np.ndarray:
    """各シナプスが各 part を通ったかの (M, P) bool 行列。

    判定は**線分のサンプリング**で行う。端点だけを見てはいけない —— 1 セグメント
    (既定 100 um) はブリッジ幅 (20-50 um) より長いので、端点判定では細いブリッジを
    またいでしまう (`BaseArea.segment_inside` が同じ理由でサンプリングしている)。

    Args:
        geometry: `AxonGeometry` 互換。`pre` / `post` は COO と**位置一致している前提**。
        area: composite area。`parts` を読む。
        part_indices: 調べる part の index。None ならブリッジ part すべて
            (`connectivity.bridge_part_indices`)。
        samples: 1 セグメントあたりのサンプル点数。

    Returns:
        (M, P) bool。列の並びは `part_indices` の並び。
    """
    if part_indices is None:
        part_indices = bridge_part_indices(area)
    part_indices = np.asarray(part_indices, dtype=np.int64)
    parts = list(area.parts)

    num_synapses = int(np.asarray(geometry.pre).size)
    crossed = np.zeros((num_synapses, part_indices.size), dtype=bool)
    if num_synapses == 0 or part_indices.size == 0:
        return crossed

    weights = np.linspace(0.0, 1.0, int(samples))[:, None, None]
    for m in range(num_synapses):
        starts, ends = synapse_path_segments(geometry, m)
        if starts.shape[0] == 0:
            continue
        # (samples, k, 2) -> (samples*k, 2)。1 シナプスぶんをまとめて 1 回で問い合わせる。
        points = (starts[None, :, :] + weights * (ends - starts)[None, :, :]).reshape(-1, 2)
        for j, part_index in enumerate(part_indices):
            crossed[m, j] = bool(np.any(parts[int(part_index)].contains(points)))
    return crossed


def crossing_summary(crossed: np.ndarray, part_indices, part_names) -> dict[str, int]:
    """part 名 -> その part を通ったシナプス数。`lesion.json` へそのまま入れられる形。"""
    names = list(part_names)
    return {
        str(names[int(p)]): int(crossed[:, j].sum())
        for j, p in enumerate(np.asarray(part_indices, dtype=np.int64))
    }


def subset_geometry(geometry, keep):
    """シナプスを絞った軸索幾何の**ビュー**を返す (元は変更しない)。

    切断後のネットワークを `axon_network()` で描くのに要る。あの図は
    `geometry.pre` / `post` / `contact_seg` / `contact_t` からシナプスの経路を引くので、
    切断前の幾何をそのまま渡すと**切ったはずの結合まで描かれる**。

    絞るのはシナプス側の 4 本だけで、**軸索そのもの (`seg_*` / `offsets` / `seg_owner`) は
    触らない**。この実験で除去したのはシナプスであって軸索ではないので、下敷きに描かれる
    軸索の形は切断前後で変わらないのが正しい。`contact_seg` はセグメントのグローバル
    index なので、セグメント配列を保つ限り添字は有効なまま。

    Args:
        geometry: `AxonGeometry` 互換。
        keep: 残すシナプスの bool マスク (長さ M) か index 配列。

    Returns:
        `AxonGeometry` と同じ属性を持つ `SimpleNamespace`
        (`src/models` を import しないための duck type)。
    """
    keep = np.asarray(keep)
    # seg_owner は `axon_network()` が読まないので必須にしない (ダックタイプの最小契約を
    # 消費側に合わせる)。持っていればそのまま引き継ぐ。
    return SimpleNamespace(
        seg_start=geometry.seg_start,
        seg_end=geometry.seg_end,
        seg_owner=getattr(geometry, "seg_owner", None),
        offsets=geometry.offsets,
        pre=np.asarray(geometry.pre)[keep],
        post=np.asarray(geometry.post)[keep],
        contact_seg=np.asarray(geometry.contact_seg)[keep],
        contact_t=np.asarray(geometry.contact_t)[keep],
    )
