"""develop の run を読む。`Window` (記録窓 1 つ) と `Series` (run 全体)。

`scripts/tools/runview.py` の契約を、この実験のファイル名規約 (`records.py`) の上に
実装したもの。**ここは読み出しだけ。加工はしない。**

以前はここに `weight_trajectories()` / `firing_rate_series()` / `weight_deltas()` という
run 全体の計算が同居していた。「何を測るか」で変わるものなので、それぞれ**唯一の
利用者である図**へ移した (`figures/fig2c.py` / `figures/fig2d.py` /
`figures/weight_matrix.py`)。読み手に加工を混ぜると、図のファイルを見ても何を計算して
いるのか分からなくなる。

**時刻の正は npz が持つ。** ファイル名の `{hour:g}` は有効数字 6 桁なので、record_hours が
非整数だと往復で元に戻らない。`Window.hour` は npz の `record_start_ms` から作る。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config_manager import ConfigManager
from src.core.layout import NetworkLayout
from src.core.output_manager import AXES_NAME, CONFIG_NAME

from scripts.tools import runview
from scripts.tools.runview import Coo, MissingData, Spikes, Trace, Wiring

from scripts.develop.store import paths
from scripts.develop.store.records import (METRICS_NAME, MS_PER_HOUR, SPIKES, TRACE, WEIGHTS,
                                           discover_records, load_connectivity, load_spikes,
                                           load_trace, load_weight_values,
                                           read_record_start_ms, record_filename)


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

    def weights(self) -> np.ndarray:
        """重みの値ベクトル。**wiring と本数が合うことをここで確かめる。**

        合わないまま下流へ流すと、ブロック分けが 1 本ずつずれた図と指標になる
        (どこも例外を出さない)。
        """
        values = load_weight_values(self.weights_path)
        wiring = self.wiring()
        if values.size != wiring.row.size:
            raise ValueError(
                f"{self.weights_path} の重み {values.size} 本が "
                f"connectivity の {wiring.row.size} 本と一致しません。"
            )
        return values

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

    @property
    def windows(self) -> tuple[Window, ...]:
        return self._windows

    def wiring(self) -> Wiring:
        return self._wiring

    def metrics(self) -> pd.DataFrame:
        """`metrics.csv` を読む。

        **メソッドであってフィールドではない。** 再解析は「開く → 指標を書き直す →
        図を描く」の順に進むので、開いた時点で読んでしまうと図が**前回の値**を描く
        (しかも例外は出ない)。呼ばれた時点で読めば構造的に起こらない。
        """
        path = paths.data_path(self.run_dir, METRICS_NAME)
        if not path.exists():
            raise MissingData(METRICS_NAME, f"{path} がありません")
        return pd.read_csv(path)

    # --- 走りながら使う口 ---------------------------------------------------------

    def window(self, hour: float) -> Window:
        """**いま書いたばかりの**記録を指す `Window` を作る。

        `run_one.py` が使う。本番も再解析も「一度ファイルに書いてから読み直す」ことで
        描画経路を 1 本に保つための入口で、これが無いと本番だけ in-memory の値を
        手で組み立てて渡すことになる (以前はそうなっていた)。
        """
        data_dir = paths.data_dir(self.run_dir)
        spikes_path = data_dir / record_filename(SPIKES, hour)
        start_ms = read_record_start_ms(spikes_path)
        return _make_window(self, data_dir, spikes_path, start_ms)


def open_run(run_dir: str | Path, *, require_windows: bool = True) -> Series:
    """run ディレクトリを開く。config を読み、layout を復元し、記録を時刻順に並べる。

    `connectivity.npz` は run を通して不変なのでここで 1 回だけ読む。

    Args:
        require_windows: 記録が 1 つも無いときに落とすか。`run_one.py` は**これから
            記録を書く**ところで開くので `False` を渡す。

    **layout が復元できなければ落とす。** 無いと E/I 列が作れず、本番より列の少ない
    `metrics.csv` で上書きしてしまう。「並べ替えだけ諦めて続行」は、失われるのが図では
    なく記録なので割に合わない。
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
