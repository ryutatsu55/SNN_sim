"""実験出力の読み出し契約。

解析・描画関数が「どの実験のデータか」ではなく**この契約**に対して書かれる。同じ関数が
実験をまたいで動き、共有層と実験固有層の間の移動がファイル 1 つの移動で済む。

    RunView          1 つの run を、ある時点から見たもの
    ├── Built        build 直後。まだ回していない
    │   └── BuiltNetwork   ビルド済み NetworkBuilder から作る唯一の実装
    ├── Window       記録窓 1 つ
    │   └── MemoryWindow   記録を持たない道具のための実装
    └── Series       run 全体

## 約束

- **全実験共通。分岐させない。** 実験ごとに違うのは実装 (どのファイルをどう読むか)
  であって、名前と形ではない。実装は各実験の `store/`。
- 基底が持つデータは `run_dir` / `config` / `layout` の 3 つだけ。それ以外は
  **「その名前で呼べる」ことだけを決めたアクセサ**で、既定は `MissingData` を投げる。
  実験は持っているものだけを override する。
- **スパイクもトレースも時刻は窓の先頭を 0 とするローカル [ms]。** 絶対時刻は返さない。
- `Window.hour` は基準時刻からの経過 [h]。**識別子ではなく数値**で、図の横軸になる。
  基準を何に置くかは実験が決める (develop は run の開始、lesion は切断時刻)。
- **`src/models` も matplotlib も NetworkBuilder も import しない。** area も軸索
  ジオメトリも builder も、すべてダックタイピングで受ける。

設計の背景は `docs/architecture/runview_contract.md`。
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class MissingData(Exception):
    """この run はそのデータを持たない、という正常な状態。**バグではない。**

    `no_space` の config に座標が無い、トレースを採っていない run に V/I が無い、など。
    受けるのは `guard()`。
    """

    def __init__(self, what: str, detail: str | None = None):
        self.what = what
        super().__init__(f"{what} がありません" + (f" ({detail})" if detail else ""))


def optional(getter, default=None):
    """**あれば使う**ものを読む。無ければ `default`。

    使ってよいのは「その図の本体ではないが、あれば重ねたい」ものだけ。図の本体が
    要求するデータに使うと、中身が抜けた図が黙って出る。

        area = optional(built.area)        # あれば境界線を重ねる
        coords = built.coords()            # 無ければこの図は描けない -> MissingData
    """
    try:
        return getter()
    except MissingData:
        return default


def guard(label: str, fn, *args) -> None:
    """出力 1 つを実行する。**失敗しても残りを止めない。**

    `MissingData` は `skip`、それ以外は `Warning` として報告するので、ログ上で
    「この run はそのデータを持たない」と「バグで落ちた」が区別できる。

    通してよいのは**出力を作る処理だけ**。記録を書く処理をここへ通すと、失敗が
    警告 1 行になって流れる。
    """
    try:
        fn(*args)
    except MissingData as missing:
        print(f"  skip {label}: {missing}")
    except Exception as error:
        print(f"  Warning: {label} generation failed: {error}")


# ======================================================================================
# 読み出した値の形。**フィールド名は npz のキー名と揃える。**
# ======================================================================================

class Wiring(NamedTuple):
    """どこに結合があるか。**run を通して不変。**

    重みは持たない。重みは時点ごとに変わるので `weights()` / `coo()` の担当。
    """
    row: np.ndarray
    col: np.ndarray
    shape: tuple[int, int]

    @property
    def num_synapses(self) -> int:
        return int(self.row.size)


class Coo(NamedTuple):
    """ある時点の結合。`Wiring` + その時点の重み (と遅延)。

    時点が定まる view だけが持つ。`Series` は `wiring()` だけを返す。
    """
    row: np.ndarray
    col: np.ndarray
    weights: np.ndarray
    delays: np.ndarray | None
    shape: tuple[int, int]

    @property
    def num_synapses(self) -> int:
        return int(self.row.size)

    def wiring(self) -> Wiring:
        return Wiring(row=self.row, col=self.col, shape=self.shape)


class Spikes(NamedTuple):
    """1 記録窓ぶんのスパイク。

    **`times` は窓の先頭を 0 とするローカル時刻 [ms]。** 絶対時刻が要るときは
    `view.record_start_ms` を足す。
    """
    times: np.ndarray
    ids: np.ndarray


class Trace(NamedTuple):
    """単一ニューロンの膜電位トレース。`spike_times` も窓の先頭を 0 とするローカル時刻。"""
    V: np.ndarray
    I: np.ndarray
    dt: float
    neuron_id: int
    window_s: float
    spike_times: np.ndarray
    spike_ids: np.ndarray


# ======================================================================================
# view
# ======================================================================================

class RunView:
    """1 つの run を、ある時点から見たもの。持つデータは `run_dir` / `config` / `layout`。"""

    def __init__(self, run_dir: Path, config, layout):
        self.run_dir = Path(run_dir)
        self.config = config
        self.layout = layout

    @property
    def total_neurons(self) -> int:
        return int(self.layout.total_neurons)

    # --- アクセサ。既定は全部 MissingData。実験は持っているものだけ override する ---

    def wiring(self) -> Wiring:
        """どこに結合があるか (run 不変)。"""
        raise MissingData("wiring")

    def coo(self) -> Coo:
        """その時点の結合 (重み込み)。"""
        raise MissingData("coo")

    def weights(self) -> np.ndarray:
        """その時点の重み (1D, `wiring()` と index 整合)。"""
        raise MissingData("weights")

    def coords(self) -> np.ndarray:
        """soma の座標 (N, 2)。`no_space` の run は持たない。"""
        raise MissingData("coords")

    def area(self):
        """ニューロンが置かれる領域。無界のエリアは持たないものとして扱う。"""
        raise MissingData("area")

    def geometry(self):
        """軸索の折れ線。`axon_growth` 系のコネクタだけが持つ。"""
        raise MissingData("geometry")

    def spikes(self) -> Spikes:
        """スパイク列 (ローカル時刻)。"""
        raise MissingData("spikes")

    def trace(self) -> Trace:
        """膜電位トレース。採取した run だけが持つ。"""
        raise MissingData("trace")

    def metrics(self):
        """指標の表 (`pd.DataFrame`)。"""
        raise MissingData("metrics")


class Built(RunView):
    """build 直後。シミュレーションの結果ではなく**ネットワークそのもの**。

    ここにしかない情報 (エリアの実体・軸索の幾何・初期重み) を扱う。
    """


class Window(RunView):
    """記録窓 1 つ。`hour` は基準時刻からの経過 [h] (モジュール docstring の約束を参照)。"""

    def __init__(self, run_dir: Path, config, layout, *, hour: float,
                 record_start_ms: float):
        super().__init__(run_dir, config, layout)
        self.hour = float(hour)
        self.record_start_ms = float(record_start_ms)


class Series(RunView):
    """run 全体。記録窓が時刻順に並ぶ。"""

    @property
    def windows(self) -> tuple[Window, ...]:
        raise MissingData("windows")

    @property
    def hours(self) -> np.ndarray:
        return np.array([window.hour for window in self.windows], dtype=np.float64)


class BuiltNetwork(Built):
    """ビルド済み `NetworkBuilder` を `Built` として見る。

    契約のうち**実装まで全実験で共通な唯一のもの**。構造図は builder の内部 API を
    知らずに済み、本番の builder と再解析で再ビルドした builder が同じ型になる。
    """

    def __init__(self, run_dir: Path, config, layout, coo: Coo,
                 coords=None, area=None, geometry=None):
        super().__init__(run_dir, config, layout)
        self._coo = coo
        self._coords = coords
        self._area = area
        self._geometry = geometry

    @classmethod
    def from_builder(cls, builder, run_dir: Path, *, geometry=None) -> "BuiltNetwork":
        """ビルド済みの NetworkBuilder から作る。

        Args:
            builder: `build()` か `_generate_global_matrices()` を通した後のもの。
                **ここで作り直さないこと** (同じ config でも seed 未指定なら別の
                ネットワークになる)。
            geometry: 軸索幾何。None ならコネクタから取る。**結合を差し替えた builder
                では必ず渡すこと** —— コネクタの幾何は差し替えを反映しない。
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
        """soma の座標 (N, 2)。`no_space` の run は持たない。

        None だけでなく**非有限値も「無い」**として扱う (`no_space` は `inf` で埋める)。
        """
        coords = self._coords
        if coords is None:
            raise MissingData("coords", "空間を持たない config です")
        coords = np.asarray(coords, dtype=np.float64)
        if not np.all(np.isfinite(coords)):
            raise MissingData("coords", "座標に有限でない値があります (no_space)")
        return coords

    def area(self):
        """ニューロンが置かれる領域。**無界のエリアは持たないものとして扱う。**"""
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
        """記録済みの結合と**同じ結合集合**であることを確かめ、違えば止める。

        再解析が構造図のために再ビルドしたときに呼ぶ。比べるのは集合なので、
        答えるのは「同じネットワークか」であって「同じ並びか」ではない。
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
        if not same_edge_set(mine, wiring):
            raise SystemExit(
                "再ビルドした結合が記録の connectivity.npz と一致しません"
                " (本数は同じですが、繋がっている相手が違います)。"
                " 同じ seed から別のネットワークが出ているので、構造図は描けません。"
            )


def same_edge_set(a: Wiring, b: Wiring) -> bool:
    """(row, col) の集合として等しいか。並び順は無視する。"""
    return (np.array_equal(a.row[np.lexsort((a.col, a.row))],
                           b.row[np.lexsort((b.col, b.row))])
            and np.array_equal(a.col[np.lexsort((a.col, a.row))],
                               b.col[np.lexsort((b.col, b.row))]))


class MemoryWindow(Window):
    """メモリ上の配列をそのまま契約に載せる窓。

    **記録を持たない道具 (`scripts/tools/`) 専用。** run ディレクトリも記録ファイルも
    無いので、読み直す先が無い。`hour` の既定 0.0 は「記録窓が 1 つで時間軸を持たない」。

    **実験では使わない。** 実験は一度 npz に書いてから読み直すことで、本番と再解析の
    描画経路を 1 本に保っている。
    """

    def __init__(self, run_dir: Path, config, layout, *, hour: float = 0.0,
                 record_start_ms: float = 0.0, spikes: Spikes | None = None,
                 trace: Trace | None = None, coo: Coo | None = None, coords=None):
        super().__init__(run_dir, config, layout, hour=hour,
                         record_start_ms=record_start_ms)
        self._spikes = spikes
        self._trace = trace
        self._coo = coo
        self._coords = coords

    @property
    def record_window_ms(self) -> float:
        return float(self.config.task.duration)

    def spikes(self) -> Spikes:
        if self._spikes is None:
            raise MissingData("spikes")
        return self._spikes

    def trace(self) -> Trace:
        if self._trace is None:
            raise MissingData("trace")
        return self._trace

    def coo(self) -> Coo:
        if self._coo is None:
            raise MissingData("coo")
        return self._coo

    def wiring(self) -> Wiring:
        return self.coo().wiring()

    def weights(self) -> np.ndarray:
        return self.coo().weights

    def coords(self) -> np.ndarray:
        if self._coords is None:
            raise MissingData("coords", "空間を持たない config です")
        coords = np.asarray(self._coords, dtype=np.float64)
        if not np.all(np.isfinite(coords)):
            raise MissingData("coords", "座標に有限でない値があります (no_space)")
        return coords
