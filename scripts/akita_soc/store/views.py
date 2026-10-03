"""akita_soc の run を読む。`Built` (build 直後)、`Window` (記録窓 1 つ)、`Series` (run 全体)。

`src/utils/runview.py` の契約 (全実験共通の形) を、この実験のファイル名規約
(`records.py`) の上に実装したもの。**ここは読み出しだけ。加工はしない。**

run を通して不変なもの (結合・座標・領域・軸索・遅延) は `Series` が持ち、`Built` と
`Window` はそれを借りる。両者の違いは「どの時点の重みか」だけ。

**記録が無いときの `MissingData` は理由を 2 つに分ける。** その run が本来持たない
(`no_space` に座標が無い) のか、記録を足す前に走った run なのか。どちらかは config から
決まるので、ファイルの有無から推測はしない。

run 全体を通した計算 (重み軌跡・発火レートの時系列) は、それを使う図のファイルにある
(`figures/fig2c.py` / `figures/fig2d.py` / `figures/weight_matrix.py`)。

**時刻の正は npz が持つ。** ファイル名の `{hour:g}` は有効数字 6 桁なので、record_hours が
非整数だと往復で元に戻らない。`Window.hour` は npz の `record_start_ms` から作る。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config_manager import CONFIG_NAME, ConfigManager
from src.core.layout import NetworkLayout
from src.core.layout import AXES_NAME

from src.utils import runview
from src.utils.runview import Coo, MissingData, Spikes, Trace, Wiring

from scripts.akita_soc.store import paths
from scripts.akita_soc.store.records import (COORDS_NAME, DELAYS_NAME, INITIAL_WEIGHTS_NAME,
                                           METRICS_NAME, MS_PER_HOUR, SPIKES, TRACE, WEIGHTS,
                                           discover_records, load_connectivity, load_coords,
                                           load_delays, load_metrics, load_spikes, load_trace,
                                           load_weight_values, read_record_start_ms,
                                           record_filename)


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
            f"{path} の値 {values.size} 本が connectivity の {wiring.row.size} 本と一致しません。"
        )
    return values


class Built(runview.Built):
    """build 直後の記録。**まだ 1 ステップも回していない**ネットワークそのもの。

    `Window` との違いは重みだけ (`weights_initial.npz` = build 直後の値)。
    `Series.built()` が作る。
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
        """build 直後の重み (1D, wiring と index 整合)。"""
        path = paths.data_path(self.run_dir, INITIAL_WEIGHTS_NAME)
        if not path.exists():
            raise _not_recorded("initial weights", path)
        return _load_value_vector(path, self.wiring(), load_weight_values)

    def coo(self) -> Coo:
        wiring = self.wiring()
        return Coo(row=wiring.row, col=wiring.col, weights=self.weights(),
                   delays=self._series.delays(), shape=wiring.shape)


class Window(runview.Window):
    """1 記録時刻ぶんの記録。

    `hour` は npz が持つ原点から作った値なので、`metrics.csv` の `hour` 列と
    ビット単位で一致する (どちらも同じ `record_start_ms` から割っている)。
    """

    def __init__(self, series: "Series", *, hour: float, record_start_ms: float,
                 spikes_path: Path, weights_path: Path, trace_path: Path | None):
        super().__init__(series.run_dir, series.config, series.layout,
                         hour=hour, record_start_ms=record_start_ms)
        self._series = series
        self.spikes_path = spikes_path
        self.weights_path = weights_path
        self.trace_path = trace_path

    @property
    def record_window_ms(self) -> float:
        return float(self.config.task.record_window_ms)

    def wiring(self) -> Wiring:
        return self._series.wiring()

    def coords(self) -> np.ndarray:
        """soma の座標。run を通して不変なので series が持つものをそのまま返す。"""
        return self._series.coords()

    def area(self):
        """ニューロンが置かれる領域。**run を通して不変**なので series が持つものを返す。"""
        return self._series.area()

    def geometry(self):
        """軸索の折れ線。**run を通して不変**なので series が持つものを返す。"""
        return self._series.geometry()

    def weights(self) -> np.ndarray:
        """その記録時刻の重み (1D)。**wiring と本数が合うことをここで確かめる。**"""
        return _load_value_vector(self.weights_path, self.wiring(), load_weight_values)

    def coo(self) -> Coo:
        wiring = self.wiring()
        return Coo(row=wiring.row, col=wiring.col, weights=self.weights(),
                   delays=None, shape=wiring.shape)

    def spikes(self) -> Spikes:
        return load_spikes(self.spikes_path)

    def trace(self) -> Trace:
        if self.trace_path is None:
            raise MissingData("trace", "この run は膜電位トレースを採っていません")
        return load_trace(self.trace_path)


class Series(runview.Series):
    """完走した run 1 つ。`open_run()` が作る。"""

    def __init__(self, run_dir: Path, config, layout, wiring: Wiring,
                 windows: tuple[Window, ...] = ()):
        super().__init__(run_dir, config, layout)
        self._wiring = wiring
        self._windows = windows
        # area は config から組み直すので 1 度だけ作って持つ (`area()` 参照)。
        self._area = None
        # 軸索の幾何は大きいので 1 度だけ読んで持つ。
        self._geometry = None

    @property
    def windows(self) -> tuple[Window, ...]:
        return self._windows

    def wiring(self) -> Wiring:
        return self._wiring

    def coords(self) -> np.ndarray:
        """soma の座標。**`no_space` の run は持たない** (ファイルごと無い)。

        結合構造と同じく run を通して不変なので、窓はこれを借りる。
        """
        path = paths.data_path(self.run_dir, COORDS_NAME)
        if not path.exists():
            if self.config.network.area.profile_name == "no_space":
                raise MissingData("coords", "空間を持たない config です")
            raise _not_recorded("coords", path)
        return load_coords(path)

    def geometry(self):
        """軸索の折れ線 (`AxonGeometry`)。**axon_growth 系のコネクタの run だけが持つ。**"""
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
        """伝播遅延 [ms] (1D, wiring と index 整合)。run を通して不変。

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

    def metrics(self) -> pd.DataFrame:
        """`metrics.csv` を読む。**呼ばれた時点で読む** (開いた時点ではない)。

        再解析は「開く → 指標を書き直す → 図を描く」の順に進むので、先に読むと図が
        前回の値を描く。
        """
        path = paths.data_path(self.run_dir, METRICS_NAME)
        if not path.exists():
            raise MissingData(METRICS_NAME, f"{path} がありません")
        return load_metrics(path)

    def built(self) -> Built:
        """build 直後の時点から見た view。構造図 (`report/structure.py`) に渡す。"""
        return Built(self)

    # --- 走りながら使う口 ---------------------------------------------------------

    def window(self, hour: float) -> Window:
        """**いま書いたばかりの**記録を指す `Window` を作る。

        `run_one.py` が「書いてから読み直す」ための入口。
        """
        data_dir = paths.data_dir(self.run_dir)
        spikes_path = data_dir / record_filename(SPIKES, hour)
        start_ms = read_record_start_ms(spikes_path)
        return _make_window(self, data_dir, spikes_path, start_ms)


def open_run(run_dir: str | Path, *, require_windows: bool = True) -> Series:
    """run ディレクトリを開く。config を読み、layout を復元し、記録を時刻順に並べる。

    結合構造は run を通して不変なのでここで 1 回だけ読む。build の記録
    (`build_record.write_build_record()`) を書いた直後から開ける。
    **layout が復元できなければ落とす** (E/I 列の無い `metrics.csv` を作らないため)。

    Args:
        require_windows: 記録が 1 つも無いときに落とすか。`run_one.py` は**これから
            記録を書く**ところで開くので `False` を渡す。
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
    # 「ファイルが無い」は正常。**旧レイアウトの吸収ではない。**
    axes_path = paths.data_path(run_dir, AXES_NAME)
    if axes_path.exists():
        layout.load_axes_file(axes_path)

    data_dir = paths.data_dir(run_dir)
    wiring = load_connectivity(data_dir)
    if layout.total_neurons != wiring.shape[0]:
        # config から復元した layout と記録された結合の大きさが食い違う。この先どの図も
        # 「ずれた行列」を黙って描くので、読んだ時点で落とす。
        raise ValueError(
            f"layout の total_neurons={layout.total_neurons} が "
            f"connectivity の {wiring.shape[0]} と一致しません: {run_dir}"
        )
    series = Series(run_dir, config, layout, wiring)
    series._windows = _discover_windows(series, data_dir)
    if require_windows and not series.windows:
        raise FileNotFoundError(f"{record_filename(SPIKES, 0)} 形式の記録がありません: {data_dir}")
    return series


def _discover_windows(series: Series, data_dir: Path) -> tuple[Window, ...]:
    """スパイク記録を並べ、各時刻の相方 (重み・トレース) の在処を決める。

    ファイル名は**並べ替えと人間の目印**にしか使わない。時刻は npz の中の原点から作る。
    """
    windows = [_make_window(series, data_dir, item.path, read_record_start_ms(item.path))
               for item in discover_records(data_dir, SPIKES)]
    return tuple(sorted(windows, key=lambda window: window.hour))


def _make_window(series: Series, data_dir: Path, spikes_path: Path,
                 start_ms: float) -> Window:
    hour = start_ms / MS_PER_HOUR
    weights_path = data_dir / record_filename(WEIGHTS, hour)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"{spikes_path.name} に対応する {weights_path.name} がありません。"
        )
    trace_path = data_dir / record_filename(TRACE, hour)
    return Window(series, hour=hour, record_start_ms=start_ms,
                  spikes_path=spikes_path, weights_path=weights_path,
                  trace_path=trace_path if trace_path.exists() else None)
