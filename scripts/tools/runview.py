"""実験出力の読み出し契約。

解析・描画関数が「どの実験のデータか」ではなく **この契約**に対して書かれるようにする
ための取り決め。おかげで同じ関数が実験をまたいで動き、共有層と実験固有層の間の移動が
ファイル 1 つの移動で済む。

設計の背景と決定の経緯は `docs/runview_contract.md`。

**ここは全実験共通。分岐させない。** 実験ごとに違うのは *実装* (どのファイルをどう読むか)
であって、*名前と形* ではない。

    RunView          1 つの run を、ある時点から見たもの
    ├── Built        build 直後。まだ回していない
    ├── Window       記録窓 1 つ
    └── Series       run 全体

基底が持つデータは `run_dir` / `config` / `layout` の 3 つだけ。それ以外は
**「その名前で呼べる」ことだけを決めたアクセサ**で、既定は `MissingData` を投げる。
実験は持っているものだけを override すればよい。
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np


class MissingData(Exception):
    """この run はそのデータを持たない。**バグではない。**

    `no_space` の config に座標が無い、トレースを採っていない run に V/I が無い、
    といった正常な状態を表す。呼び出し側 (登録簿の `guard()`) がこれを捕まえて
    「無いから出なかった」と「バグで落ちた」を区別する。
    """

    def __init__(self, what: str, detail: str | None = None):
        self.what = what
        super().__init__(f"{what} がありません" + (f" ({detail})" if detail else ""))


def optional(getter, default=None):
    """**あれば使う**ものを読む。無ければ `default`。

    「その図の本体ではないが、あれば重ねたい」もの専用 (ネットワーク図に重ねるエリアの
    境界線など)。図の本体が要求するデータにこれを使ってはいけない —— 無いまま描いて
    しまうと、**中身が抜けた図が黙って出る。**

        area = optional(built.area)        # あれば境界線を重ねる
        coords = built.coords()            # 無ければこの図は描けない -> MissingData
    """
    try:
        return getter()
    except MissingData:
        return default


# ======================================================================================
# 読み出した値の形
#
# **フィールド名は npz のキー名と揃える。** 読む側と書く側で名前が食い違わないようにする。
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

    **時点が定まる view だけが持つ。** `Series` は「いつの重みか」が決まらないので
    持たない (`wiring()` だけを返す)。
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

    **`times` は記録窓の先頭を 0 とするローカル時刻 [ms]。** 絶対時刻が要るときは
    `view.record_start_ms` を足す。絶対時刻を既定にすると、アバランチ分割は同じでも
    burstiness のビン割りが静かにずれる (どこも例外を出さない)。
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
    """1 つの run を、ある時点から見たもの。

    **持つデータは 3 つだけ。** 実験ごとに変わるものを基底に入れない。
    """

    def __init__(self, run_dir: Path, config, layout):
        self.run_dir = Path(run_dir)
        self.config = config
        self.layout = layout

    @property
    def total_neurons(self) -> int:
        return int(self.layout.total_neurons)

    # --- アクセサ -----------------------------------------------------------------
    #
    # 既定は全部 MissingData。実験は持っているものだけ override する。
    # **図の側は分岐を書かない** —— 無いものを要求して、投げられたまま素通しする。

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
    """build 直後。**シミュレーションの結果ではなくネットワークそのもの。**

    ここでしか存在しない情報 (エリアの実体・軸索の幾何・初期重み) を扱う。
    """


class Window(RunView):
    """記録窓 1 つ。

    `hour` は **基準時刻からの経過時間 [h]**。基準を何に置くかは実験が決める
    (develop は run の開始、lesion は切断時刻)。

    **数値であって識別子ではない。** 記録時刻は等間隔とは限らず、点の間隔そのものが
    図の横軸になる。
    """

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
