"""build 直後の時点 (`Built`) の記録を書く。**何を書くかの一覧はここだけ。**

    data/layout_axes.npz      外部軸 (layer / module …)。直列化は `NetworkLayout`
    data/coords.npz           soma の座標。`no_space` の run では書かない
    data/axon_geometry.npz    軸索の折れ線。axon_growth 系だけ。直列化は `AxonGeometry`
    data/connectivity.npz     結合 (row / col / shape)
    data/delays.npz           遅延 [ms] (connectivity と index 整合)
    data/weights_initial.npz  build 直後の重み (同上)

`config.yaml` はここに含めない (build の前にランチャが書く)。

**自前のスキーマを持つオブジェクトは自分で直列化し、ただの配列は `formats.py` が直列化する**
—— ここは「何を・どこへ」を決めるだけ。

読む側は `views.py` (`Series` と、build 直後の時点を見る `Built`)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.layout import AXES_NAME
from src.models.network.connectors import AXONS_NAME

from scripts.akita_soc.store import paths
from scripts.akita_soc.store.formats import (CONNECTIVITY_NAME, COORDS_NAME, DELAYS_NAME,
                                           INITIAL_WEIGHTS_NAME, save_connectivity,
                                           save_coords, save_delays, save_weight_values)


def write_built(run_dir: Path, builder) -> None:
    """ビルド済みの `builder` から build 直後の記録を全部書く。

    `builder` は `build()` を通した後のもの。GeNN の setup は要らない —— 結合は
    `builder.global_coo()` が唯一の入口 (ルート `CLAUDE.md` Hard Rule 4)。
    """
    builder.layout.save_axes(paths.data_path(run_dir, AXES_NAME))

    # 「ファイルが無い = 空間を持たない run」を読む側の判定にしている。
    coords = builder.global_coords
    if coords is not None and np.all(np.isfinite(coords)):
        save_coords(paths.data_path(run_dir, COORDS_NAME), coords)

    geometry = getattr(builder.connection, "axon_geometry", lambda: None)()
    if geometry is not None:
        geometry.save(paths.data_path(run_dir, AXONS_NAME))
        print(f"  Saved axon geometry: {geometry.seg_owner.size} segments -> {AXONS_NAME}")

    coo = builder.global_coo()
    save_connectivity(paths.data_path(run_dir, CONNECTIVITY_NAME), coo.row, coo.col,
                      np.asarray(coo.shape, dtype=np.int64))
    save_delays(paths.data_path(run_dir, DELAYS_NAME), coo.delays)
    save_weight_values(paths.data_path(run_dir, INITIAL_WEIGHTS_NAME), coo.weights)
    print(f"  Saved connectivity: {coo.num_synapses} synapses -> {CONNECTIVITY_NAME}"
          f" (+ {DELAYS_NAME}, {INITIAL_WEIGHTS_NAME})")
