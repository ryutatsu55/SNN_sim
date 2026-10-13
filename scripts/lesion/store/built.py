"""切断直後 (= この run の build 直後、`Built`) の記録を書く。**何を書くかの一覧はここだけ。**

    data/layout_axes.npz       外部軸 (親 run から読み戻したもの)。直列化は `NetworkLayout`
    data/coords.npz            soma の座標。`no_space` の run では書かない
    data/connectivity_pre.npz  切断**前**の結合 (全シナプス)。ベースラインの窓が使う
    data/connectivity.npz      切断**後**の結合 (生き残り)
    data/delays.npz            切断後の遅延 [ms] (connectivity と index 整合)
    data/weights_initial.npz   切断直後の重み = 親 run から復元した値の残存ぶん (同上)
    data/axon_geometry.npz     切断後の軸索幾何。axon_growth 系だけ。直列化は `AxonGeometry`

`config.yaml` と切断の記録 (`lesion.json` / `lesion_cut.npz`) は
ここに含めない —— 前者は build の前にランチャが書く。後者は「何を切ったか」の記録で、
ネットワークの形ではないので `run_one.py` が書く。

**自前のスキーマを持つオブジェクトは自分で直列化し、ただの配列は `formats.py` が直列化する**
—— ここは「何を・どこへ」を決めるだけ。読む側は `views.py` (`Series` と `Built`)。
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from src.core.layout import AXES_NAME
from src.models.network.connectors import AXONS_NAME

from scripts.lesion.store import paths
from scripts.lesion.store.formats import (COORDS_NAME, DELAYS_NAME, INITIAL_WEIGHTS_NAME,
                                          POST_CONNECTIVITY_NAME, PRE_CONNECTIVITY_NAME,
                                          save_connectivity, save_coords, save_delays,
                                          save_weight_values)


def write_built(run_dir: Path, builder, layout, *, pre_coo, keep,
                       geometry=None) -> None:
    """切断後の `builder` から、build 直後の記録を全部書く。

    Args:
        builder: 切断と重みの復元を済ませた `build()` 後のもの。`global_coo()` が切断後の結合。
        layout: 親 run の外部軸を読み戻した layout。
        pre_coo: 切断**前**の結合 (下見の `global_coo()`)。
        keep: `pre_coo` のうち残したシナプスの bool マスク。
        geometry: 切断**前**の軸索幾何 (`AxonGeometry`)。持たないコネクタでは None。
            **ここで切断後へ絞って書く** —— コネクタの幾何は `replace_global_coo()` の
            影響を受けないので、そのまま書くと切ったはずの結合まで残る。
    """
    layout.save_axes(paths.data_path(run_dir, AXES_NAME))

    # 「ファイルが無い = 空間を持たない run」を読む側の判定にしている。
    coords = builder.global_coords
    if coords is not None and np.all(np.isfinite(coords)):
        save_coords(paths.data_path(run_dir, COORDS_NAME), coords)

    save_connectivity(paths.data_path(run_dir, PRE_CONNECTIVITY_NAME),
                      pre_coo.row, pre_coo.col, np.asarray(pre_coo.shape, dtype=np.int64))

    post = builder.global_coo()
    save_connectivity(paths.data_path(run_dir, POST_CONNECTIVITY_NAME),
                      post.row, post.col, np.asarray(post.shape, dtype=np.int64))
    save_delays(paths.data_path(run_dir, DELAYS_NAME), post.delays)
    save_weight_values(paths.data_path(run_dir, INITIAL_WEIGHTS_NAME), post.weights)
    print(f"  Saved connectivity: {pre_coo.row.size} -> {post.num_synapses} synapses"
          f" ({PRE_CONNECTIVITY_NAME} / {POST_CONNECTIVITY_NAME} + {DELAYS_NAME},"
          f" {INITIAL_WEIGHTS_NAME})")

    if geometry is not None:
        keep = np.asarray(keep, dtype=bool)
        # 絞るのはシナプス側の 4 本だけ。軸索そのものは切っていない
        # (`src/utils/analysis/axons.py::subset_geometry` と同じ線引き)。
        post_geometry = dataclasses.replace(
            geometry,
            pre=np.asarray(geometry.pre)[keep], post=np.asarray(geometry.post)[keep],
            contact_seg=np.asarray(geometry.contact_seg)[keep],
            contact_t=np.asarray(geometry.contact_t)[keep])
        post_geometry.save(paths.data_path(run_dir, AXONS_NAME))
        print(f"  Saved post-lesion axon geometry: {post_geometry.pre.size} contacts"
              f" -> {AXONS_NAME}")
