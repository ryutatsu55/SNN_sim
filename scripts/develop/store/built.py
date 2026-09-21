"""build 直後のネットワークを読む口。

シミュレーションの結果ではなく**構築されたネットワークそのもの**を見る段階。
ここでしか存在しない情報 (エリアの実体・軸索の幾何・初期重み) を扱う。

**これがあるおかげで構造図は `NetworkBuilder` を知らなくて済む。** 以前は図の側が
`builder.global_coo()` / `builder.area` / `builder.global_coords` /
`builder.connection.axon_geometry()` を直に叩いていて、図の層が `src/core` の内部 API に
結びついていた。

もう 1 つの役目は `replot.py` との共通化。再解析は構造図を描くために**再ビルドする**ので、
「生きた builder から」と「再ビルドした builder から」の 2 経路ができる。同じ型に
まとめておかないと、本番と再解析で別々の橋渡しが育つ (`series.py` と同じ理屈)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.tools import runview
from scripts.tools.runview import Coo, MissingData, Wiring


class Built(runview.Built):
    """build を通した `NetworkBuilder` の見え方。"""

    def __init__(self, run_dir: Path, config, layout, coo: Coo,
                 coords=None, area=None, geometry=None):
        super().__init__(run_dir, config, layout)
        self._coo = coo
        self._coords = coords
        self._area = area
        self._geometry = geometry

    @classmethod
    def from_builder(cls, builder, run_dir: Path, *, geometry=None) -> "Built":
        """**ビルド済みの** NetworkBuilder から作る。

        Args:
            builder: `build()` か `_generate_global_matrices()` を通した後のもの。
                **ここで作り直してはならない** —— config の seed が未指定なら
                `resolve()` のたびに別の seed が引かれ、図が「実際に走らせた
                ネットワーク」と別物になる。
            geometry: 軸索幾何。None ならコネクタから取る (持たないものは None)。
        """
        # 疎/密の違いは builder が吸収済み。ここから先は COO しか見ない (Hard Rule 4)。
        coo = builder.global_coo()
        if geometry is None:
            geometry = getattr(builder.connection, "axon_geometry", lambda: None)()
        return cls(
            run_dir=Path(run_dir),
            config=builder.config,
            layout=builder.layout,
            coo=Coo(row=coo.row, col=coo.col, weights=coo.weights,
                    delays=coo.delays, shape=coo.shape),
            coords=builder.global_coords,
            area=builder.area,
            geometry=geometry,
        )

    # --- 契約の実装 ---------------------------------------------------------------

    def wiring(self) -> Wiring:
        return self._coo.wiring()

    def coo(self) -> Coo:
        return self._coo

    def weights(self) -> np.ndarray:
        return self._coo.weights

    def coords(self) -> np.ndarray:
        """soma の座標。`no_space` の run は持たない。

        「無い」の判定は None だけでなく**非有限値も含む**。`no_space` は座標を
        `inf` で埋めるので、None しか見ないと座標軸が壊れた図が出る。
        """
        coords = self._coords
        if coords is None:
            raise MissingData("coords", "空間を持たない config です")
        coords = np.asarray(coords, dtype=np.float64)
        if not np.all(np.isfinite(coords)):
            raise MissingData("coords", "座標に有限でない値があります (no_space)")
        return coords

    def area(self):
        """ニューロンが置かれる領域。**無界のエリアは持たないものとして扱う。**

        描きようがないので、呼ぶ側が `is_bounded` を見て分岐するのではなく、
        ここで「無い」と言う。判定が 1 か所に集まる。
        """
        area = self._area
        if area is None:
            raise MissingData("area")
        if not area.is_bounded:
            raise MissingData(
                "area", f"{self.config.network.area.profile_name} は無界です")
        return area

    def geometry(self):
        """軸索の折れ線。`axon_growth` 系のコネクタだけが持つ。"""
        if self._geometry is None:
            raise MissingData(
                "geometry",
                f"connection ({self.config.network.connection.profile_name})"
                f" が軸索の幾何を持ちません")
        return self._geometry

    # --- 再ビルドの検証 -------------------------------------------------------------

    def verify_against(self, wiring: Wiring) -> None:
        """記録済みの `connectivity.npz` と**同じ結合集合**であることを確かめる。

        `replot.py` が構造図のために**再ビルド**したときに呼ぶ。ネットワークの生成は
        `NetworkBuilder.__init__` の `np.random.RandomState(config.simulation.seed)` だけに
        依存する (backend は GeNN のデバイス RNG = スパイク列にしか効かない) ので、
        同じ config からは同じ結合が出るはず —— **その「はず」を毎回確かめる。**

        **比べるのは結合の集合。** どちらも行優先ソート済みなので位置比較でも足りるが、
        この関数が答えるのは「同じネットワークか」であって「同じ並びか」ではないので、
        並び順に依存しない形にしてある。

        (以前は `builder.global_coo()` が行優先、`connectivity.npz` が GeNN の格納順、
        という 2 系統があり、位置比較だと同一ネットワークで誤検知した。並びは
        `simulator.synapse_connectivity_coo()` 側で揃えたので、いまは一致する。)
        """
        mine = self.wiring()
        if mine.shape != wiring.shape:
            raise SystemExit(
                f"再ビルドしたネットワークの大きさ {mine.shape} が記録の {wiring.shape} と"
                " 一致しません。config.yaml がその run のものか確認してください。"
            )
        if mine.row.size != wiring.row.size:
            raise SystemExit(
                f"再ビルドしたシナプス数 {mine.row.size} が記録の {wiring.row.size} と"
                " 一致しません。同じ seed から別のネットワークが出ているので、"
                " 構造図は描けません。"
            )
        if not _same_edge_set(mine, wiring):
            raise SystemExit(
                "再ビルドした結合が記録の connectivity.npz と一致しません"
                " (本数は同じですが、繋がっている相手が違います)。"
                " 同じ seed から別のネットワークが出ているので、構造図は描けません。"
            )


def _same_edge_set(a: Wiring, b: Wiring) -> bool:
    """(row, col) の集合として等しいか。**並び順は無視する。**"""
    return (np.array_equal(a.row[np.lexsort((a.col, a.row))],
                           b.row[np.lexsort((b.col, b.row))])
            and np.array_equal(a.col[np.lexsort((a.col, a.row))],
                               b.col[np.lexsort((b.col, b.row))]))
