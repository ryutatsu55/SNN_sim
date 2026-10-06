"""lesion の run を読む。`Built` (切断直後)、`Window` (probe 1 つ)、`Series` (run 全体)。

`src/utils/runview.py` の契約を、この実験のファイル名規約 (`records.py`) の上に
実装したもの。**ここは読み出しだけ。加工はしない。** 「run 全体を通した計算」
(重み軌跡・発火レートの時系列) は、それを使う図のファイルにある。

## `hour` の基準は**切断時刻**

契約の `Window.hour` は「基準時刻からの経過時間 [h]」で、基準を何に置くかは実験が決める。
develop は run の開始、**lesion は切断の瞬間**。したがって:

- baseline (切断前) の窓は **負の hour** を持つ
- post (切断後) の窓は 0 以上

0.0 を baseline に使わないのは、post の最初の probe と重なって「切断の瞬間」が図でも
CSV でも判別できなくなるため。baseline は窓の幅ぶん手前に置く。

**baseline はこの run が測ったものではない。** 親 run の記録時刻の窓をそのまま置いた
もので、窓幅も親のもの (`task.record_window_ms` とは違いうる)。

## シナプス本数が run の途中で変わる

損傷実験は**同じ run の中で結合が変わる**唯一の実験。`Window.wiring()` はその窓の
phase に応じて切断前 / 切断後の結合を返す。`Series.wiring()` が返すのは
**切断後 (生き残り)** —— run 全体を代表するのはそちらなので。

2 つをまたいで重みを比べる図は `src/utils/analysis/weights.align_subset_to_coo()` で引き当てる。

## `Built` は**切断直後**

この run の「build 直後」は、親のネットワークを組み直して切り、親の重みを流し込んだ
瞬間。`Built` はその時点の切断後のネットワークを見る (重みは `weights_initial.npz`)。

**記録が無いときの `MissingData` は理由を 2 つに分ける。** その run が本来持たない
(`no_space` に座標が無い) のか、記録を足す前に走った run なのか。どちらかは config から
決まるので、ファイルの有無から推測はしない。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config_manager import CONFIG_NAME, ConfigManager
from src.core.layout import NetworkLayout
from src.core.layout import AXES_NAME

from src.utils import runview
from src.utils.analysis.avalanche import mean_iei_ms
from src.utils.runview import Coo, MissingData, Spikes, Trace, Wiring

from scripts.lesion.store import paths
from scripts.lesion.store.records import (BASELINE_INDEX, COORDS_NAME, DELAYS_NAME,
                                          INITIAL_WEIGHTS_NAME, METRICS_NAME, RTE_NAME,
                                          MS_PER_HOUR,
                                          PHASE_BASELINE, SPIKES, TRACE, WEIGHTS,
                                          discover_probes,
                                          load_coords, load_cut, load_connectivity,
                                          load_delays, load_manifest, load_metrics,
                                          load_spikes, load_trace,
                                          load_weight_values,
                                          post_connectivity_path, pre_connectivity_path,
                                          probe_filename, read_window_meta)


def _not_recorded(what: str, path: Path) -> MissingData:
    """記録を足す前に走った run であることを示す `MissingData`。"""
    return MissingData(what, f"この記録を足す前に走った run です ({path.name} が無い)。"
                             "再実行すれば出ます")


def _load_value_vector(path: Path, wiring: Wiring, loader) -> np.ndarray:
    """シナプスごとの値ベクトルを読み、**wiring と本数が合うことを確かめる。**

    合わないまま下流へ流すと、ブロック分けが 1 本ずつずれた図と指標になる
    (どこも例外を出さない)。
    """
    values = loader(path)
    if values.size != wiring.row.size:
        raise ValueError(
            f"{path} の値 {values.size} 本が結合 {wiring.row.size} 本と一致しません。"
        )
    return values


class Built(runview.Built):
    """切断直後の記録。**切断後**のネットワークを、可塑性が 1 ステップも走る前に見る。

    `Series.built()` が作る。結合・座標・領域・軸索は `Series` から借りる。
    """

    def __init__(self, series: "Series"):
        super().__init__(series.run_dir, series.config, series.layout)
        self._series = series

    def wiring(self) -> Wiring:
        return self._series.wiring()

    def coords(self) -> np.ndarray:
        return self._series.coords()

    def area(self):
        return self._series.area()

    def geometry(self):
        return self._series.geometry()

    def weights(self) -> np.ndarray:
        """切断直後の重み (1D, 切断後の wiring と index 整合)。"""
        path = paths.data_path(self.run_dir, INITIAL_WEIGHTS_NAME)
        if not path.exists():
            raise _not_recorded("initial weights", path)
        return _load_value_vector(path, self.wiring(), load_weight_values)

    def coo(self) -> Coo:
        wiring = self.wiring()
        return Coo(row=wiring.row, col=wiring.col, weights=self.weights(),
                   delays=self._series.delays(), shape=wiring.shape)


class Window(runview.Window):
    """probe 1 つぶんの記録。

    `hour` は**切断からの経過時間**。baseline は負。`phase` を持つのはこの実験だけで、
    契約には無い —— 図のタイトルと「どちらの結合を見るか」に使う。
    """

    def __init__(self, series: "Series", *, index: int, hour: float,
                 record_start_ms: float, record_window_ms: float, phase: str,
                 spikes_path: Path, weights_path: Path, trace_path: Path | None):
        super().__init__(series.run_dir, series.config, series.layout,
                         hour=hour, record_start_ms=record_start_ms)
        self._series = series
        self.index = index
        self.phase = phase
        self._record_window_ms = record_window_ms
        self.spikes_path = spikes_path
        self.weights_path = weights_path
        self.trace_path = trace_path

    @property
    def record_window_ms(self) -> float:
        """**この窓の**長さ。baseline と post で違いうるので npz から読む。"""
        return self._record_window_ms

    @property
    def is_baseline(self) -> bool:
        """切断**前**に測った窓か。

        図が `phase` の文字列と `records` の定数を突き合わせずに済むようにここで畳む
        (`figures/` が `store/` を import しないで済む)。
        """
        return self.phase == PHASE_BASELINE

    @property
    def label(self) -> str:
        """図のタイトルに使う短い説明。`baseline (-0.17 h)` / `post (+3.00 h)`。"""
        return f"{self.phase} ({self.hour:+.2f} h)"

    def wiring(self) -> Wiring:
        """**この窓の時点の**結合。切断前と切断後で違う。"""
        return (self._series.wiring_pre() if self.is_baseline
                else self._series.wiring())

    def coords(self) -> np.ndarray:
        """soma の座標。**切断で動かない**ので series が持つものをそのまま返す。"""
        return self._series.coords()

    def area(self):
        """ニューロンが置かれる領域。**run を通して不変**なので series が持つものを返す。"""
        return self._series.area()

    def geometry(self):
        """**切断後**の軸索幾何。series が持つものを返す (baseline の窓でも切断後)。"""
        return self._series.geometry()

    def avalanche_bin_ms(self) -> float:
        """アバランチのビン幅。**run 全体で固定**なので series が持つものを返す。"""
        return self._series.avalanche_bin_ms()

    def weights(self) -> np.ndarray:
        """重みの値ベクトル。**wiring と本数が合うことをここで確かめる。**

        合わないまま下流へ流すと、ブロック分けが 1 本ずつずれた図と指標になる
        (どこも例外を出さない)。
        """
        return _load_value_vector(self.weights_path, self.wiring(), load_weight_values)

    def coo(self) -> Coo:
        wiring = self.wiring()
        return Coo(row=wiring.row, col=wiring.col, weights=self.weights(),
                   delays=None, shape=wiring.shape)

    def spikes(self) -> Spikes:
        return load_spikes(self.spikes_path)

    def trace(self) -> Trace:
        """膜電位トレース。**採っていない probe では `MissingData`。**

        採るかどうかは `task.trace_neuron`。ベースラインは親 run が採っていれば
        引き継ぐので、post だけにある / baseline だけにある、のどちらも起こりうる。
        """
        if self.trace_path is None:
            raise MissingData("trace", "この probe は膜電位トレースを持ちません")
        return load_trace(self.trace_path)


class Series(runview.Series):
    """完走した run 1 つ。`open_run()` が作る。"""

    def __init__(self, run_dir: Path, config, layout, wiring: Wiring, wiring_pre: Wiring,
                 windows: tuple[Window, ...] = ()):
        super().__init__(run_dir, config, layout)
        self._wiring = wiring
        self._wiring_pre = wiring_pre
        self._windows = windows
        # area は config から組み直すので 1 度だけ作って持つ (`area()` 参照)。
        self._area = None
        # 軸索の幾何は大きいので 1 度だけ読んで持つ。
        self._geometry = None
        # アバランチのビンは baseline の窓で決めて固定する (`avalanche_bin_ms()` 参照)。
        self._avalanche_bin_ms = None

    @property
    def windows(self) -> tuple[Window, ...]:
        return self._windows

    def wiring(self) -> Wiring:
        """**切断後**の結合。run 全体を代表するのはこちら。"""
        return self._wiring

    def wiring_pre(self) -> Wiring:
        """切断**前**の結合 (全シナプス)。baseline の probe だけがこれを使う。"""
        return self._wiring_pre

    def coords(self) -> np.ndarray:
        """soma の座標。**`no_space` の run は持たない** (ファイルごと無い)。

        結合と違って切断の前後で 1 つしかない。切るのはシナプスであって
        ニューロンの位置ではないため。
        """
        path = paths.data_path(self.run_dir, COORDS_NAME)
        if not path.exists():
            if self.config.network.area.profile_name == "no_space":
                raise MissingData("coords", "空間を持たない config です")
            raise _not_recorded("coords", path)
        return load_coords(path)

    def geometry(self):
        """**切断後**の軸索幾何 (`AxonGeometry`)。axon_growth 系のコネクタの run だけが持つ。

        切ったのはシナプスなので、軸索の折れ線は切断前と同じで、接触点だけが絞られている。
        """
        if self._geometry is None:
            from src.core.registry import CONNECTION_MODELS
            from src.models.network.connectors import AXONS_NAME, AxonGeometry
            name = self.config.network.connection.profile_name
            path = paths.data_path(self.run_dir, AXONS_NAME)
            if not path.exists():
                if not hasattr(CONNECTION_MODELS.get(name), "axon_geometry"):
                    raise MissingData("geometry", f"connection ({name}) が軸索の幾何を持ちません")
                raise _not_recorded("geometry", path)
            self._geometry = AxonGeometry.load(path)
        return self._geometry

    def delays(self) -> np.ndarray:
        """切断後の伝播遅延 [ms] (1D, `wiring()` と index 整合)。

        契約のアクセサではない。`Coo.delays` を埋めるために `Built` が使う。
        """
        path = paths.data_path(self.run_dir, DELAYS_NAME)
        if not path.exists():
            raise _not_recorded("delays", path)
        return _load_value_vector(path, self.wiring(), load_delays)

    def area(self):
        """ニューロンが置かれる領域。**config から組み直す** (記録には残っていない)。

        `BaseArea` は RNG を保持も消費もしない (ルート `CLAUDE.md` の Hard Rule 6) ので、
        読み出しの途中で作っても run の再現性には影響しない。**無界のエリアと
        `no_space` は「持たない」**として扱う —— 契約 (`src/utils/runview.py`) と同じ。
        """
        if self._area is None:
            from src.core.registry import AREA_MODELS
            import src.models.network.area  # noqa: F401  @register を走らせる
            profile = self.config.network.area
            area_class = AREA_MODELS.get(profile.profile_name)
            self._area = area_class(profile, self.layout.total_neurons, layout=self.layout)
        if not self._area.is_bounded:
            raise MissingData("area", "無界のエリアです")
        return self._area

    def avalanche_bin_ms(self) -> float:
        """アバランチ検出のビン幅 [ms]。**ベースラインの窓で決めて run 全体で固定する。**

        Beggs-Plenz の既定は「その窓の平均イベント間隔」だが、損傷実験でそれを使うと
        **発火率が落ちるとビンが伸び、それだけで σ_bp が上がって見える**
        (実測で 0.496 → 0.529。同じビンで測ると 0.496 → 0.441 と符号が逆)。
        切断の前後を比べるのが目的なので、ビンは切断前に固定する。

        ベースラインを持たない run では測れない —— 比較の基準そのものが無い。
        """
        if self._avalanche_bin_ms is None:
            path = paths.data_path(self.run_dir, probe_filename(SPIKES, BASELINE_INDEX))
            if not path.exists():
                raise MissingData(
                    "avalanche bin",
                    f"{path.name} がありません。アバランチのビンは"
                    " ベースラインの窓で決めるので、それが無いと測れません。")
            self._avalanche_bin_ms = float(mean_iei_ms(load_spikes(path).times))
        return self._avalanche_bin_ms

    def metrics(self) -> pd.DataFrame:
        """`metrics.csv` を読む。**呼ばれた時点で読む** (開いた時点ではない)。

        再解析は「開く → 指標を書き直す → 図を描く」の順に進むので、先に読むと図が
        前回の値を描く。
        """
        path = paths.data_path(self.run_dir, METRICS_NAME)
        if not path.exists():
            raise MissingData(METRICS_NAME, f"{path} がありません")
        return load_metrics(path)

    def rte(self) -> pd.DataFrame:
        """`rte.csv` を読む。`metrics()` と同じく**呼ばれた時点で読む。**"""
        path = paths.data_path(self.run_dir, RTE_NAME)
        if not path.exists():
            raise MissingData(RTE_NAME, f"{path} がありません")
        return load_metrics(path)

    def manifest(self) -> dict:
        """`lesion.json`。切断で何が起きたかの記録。"""
        try:
            return load_manifest(paths.data_dir(self.run_dir))
        except FileNotFoundError as error:
            raise MissingData("lesion.json", str(error)) from None

    def cut(self) -> dict:
        """切断されたシナプス 1 本ごとの素性 (`lesion_cut.npz`)。"""
        try:
            return load_cut(paths.data_dir(self.run_dir))
        except FileNotFoundError as error:
            raise MissingData("lesion_cut.npz", str(error)) from None

    def built(self) -> Built:
        """切断直後の時点から見た view。構造図 (`report/structure.py`) に渡す。"""
        return Built(self)

    # --- 走りながら使う口 ---------------------------------------------------------

    def window(self, index: int) -> Window:
        """**いま書いたばかりの** probe を指す `Window` を作る。

        `run_one.py` が「書いてから読み直す」ための入口。
        """
        data_dir = paths.data_dir(self.run_dir)
        return _make_window(self, data_dir, index,
                            data_dir / probe_filename(SPIKES, index))


def open_run(run_dir: str | Path, *, require_windows: bool = True) -> Series:
    """run ディレクトリを開く。config を読み、layout を復元し、probe を番号順に並べる。

    結合構造は 2 本 (切断前・切断後) ともここで 1 回だけ読む。

    Args:
        require_windows: probe が 1 つも無いときに落とすか。`run_one.py` は**これから
            記録を書く**ところで開くので `False` を渡す。

    **layout が復元できなければ落とす。** 無いと E/I 列も module 別の指標も作れず、
    本番より列の少ない `metrics.csv` で上書きしてしまう。
    """
    run_dir = Path(run_dir)
    config_path = run_dir / CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} がありません。"
            " build を通っていない run か、run ディレクトリではありません。"
        )
    config = ConfigManager().load_resolved(config_path)

    layout = NetworkLayout.from_config(config)
    # 外部軸 (layer / module …) を持たない config では save_axes() が何も書かないので、
    # 「ファイルが無い」は正常。
    axes_path = paths.data_path(run_dir, AXES_NAME)
    if axes_path.exists():
        layout.load_axes_file(axes_path)

    data_dir = paths.data_dir(run_dir)
    wiring = load_connectivity(post_connectivity_path(data_dir))
    wiring_pre = load_connectivity(pre_connectivity_path(data_dir))
    if layout.total_neurons != wiring.shape[0]:
        # config から復元した layout と記録された結合の大きさが食い違う。この先どの図も
        # 「ずれた行列」を黙って描くので、読んだ時点で落とす。
        raise ValueError(
            f"layout の total_neurons={layout.total_neurons} が "
            f"connectivity の {wiring.shape[0]} と一致しません: {run_dir}"
        )
    series = Series(run_dir, config, layout, wiring, wiring_pre)
    series._windows = _discover_windows(series, data_dir)
    if require_windows and not series.windows:
        raise FileNotFoundError(
            f"{probe_filename(SPIKES, 0)} 形式の記録がありません: {data_dir}")
    return series


def _discover_windows(series: Series, data_dir: Path) -> tuple[Window, ...]:
    """probe を並べる。**並べ替えの鍵は index ではなく時刻。**

    baseline が負の時刻を持つので、index 順と時刻順はたまたま一致しているだけ。
    図の横軸は時刻なので、時刻で並べておく。
    """
    windows = [_make_window(series, data_dir, item.index, item.path)
               for item in discover_probes(data_dir, SPIKES)]
    return tuple(sorted(windows, key=lambda window: window.hour))


def _make_window(series: Series, data_dir: Path, index: int, spikes_path: Path) -> Window:
    start_ms, window_ms, phase = read_window_meta(spikes_path)
    weights_path = data_dir / probe_filename(WEIGHTS, index)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"{spikes_path.name} に対応する {weights_path.name} がありません。"
        )
    # トレースは任意。**無いのが既定**なので、欠けていても落とさない。
    trace_path = data_dir / probe_filename(TRACE, index)
    return Window(series, index=index, hour=start_ms / MS_PER_HOUR,
                  record_start_ms=start_ms, record_window_ms=window_ms, phase=phase,
                  spikes_path=spikes_path, weights_path=weights_path,
                  trace_path=trace_path if trace_path.exists() else None)
