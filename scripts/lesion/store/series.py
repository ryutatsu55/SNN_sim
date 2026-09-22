"""lesion の run を読む。`Window` (probe 1 つ) と `Series` (run 全体)。

`src/utils/runview.py` の契約を、この実験のファイル名規約 (`records.py`) の上に
実装したもの。**ここは読み出しだけ。加工はしない。** 「run 全体を通した計算」
(重み軌跡・発火レートの時系列) は、それを使う図のファイルにある。

## `hour` の基準は**切断時刻**

契約の `Window.hour` は「基準時刻からの経過時間 [h]」で、基準を何に置くかは実験が決める。
develop は run の開始、**lesion は切断の瞬間**。したがって:

- baseline (Phase 1, 切断前) の窓は **負の hour** を持つ
- post (Phase 2) の窓は 0 以上

0.0 を baseline に使わないのは、post の最初の probe と重なって「切断の瞬間」が図でも
CSV でも判別できなくなるため。baseline は窓の幅ぶん手前 (`-baseline_window_ms`) に置く。

## シナプス本数が run の途中で変わる

損傷実験は**同じ run の中で結合が変わる**唯一の実験。`Window.wiring()` はその窓の
phase に応じて Phase 1 / Phase 2 の結合を返す。`Series.wiring()` が返すのは
**切断後 (生き残り)** —— run 全体を代表するのはそちらなので。

2 つをまたいで重みを比べる図は `analysis/restore.align_subset_to_coo()` で引き当てる。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.core.config_manager import CONFIG_NAME, ConfigManager
from src.core.layout import NetworkLayout
from src.core.layout import AXES_NAME

from src.utils import runview
from src.utils.runview import Coo, MissingData, Spikes, Wiring

from scripts.lesion.store import paths
from scripts.lesion.store.records import (METRICS_NAME, MS_PER_HOUR, PHASE_BASELINE,
                                          SPIKES, WEIGHTS, discover_probes, load_cut,
                                          load_connectivity, load_manifest, load_spikes,
                                          load_weight_values, post_connectivity_path,
                                          pre_connectivity_path, probe_filename,
                                          read_window_meta)


class Window(runview.Window):
    """probe 1 つぶんの記録。

    `hour` は**切断からの経過時間**。baseline は負。`phase` を持つのはこの実験だけで、
    契約には無い —— 図のタイトルと「どちらの結合を見るか」に使う。
    """

    def __init__(self, series: "Series", *, index: int, hour: float,
                 record_start_ms: float, record_window_ms: float, phase: str,
                 spikes_path: Path, weights_path: Path):
        super().__init__(series.run_dir, series.config, series.layout,
                         hour=hour, record_start_ms=record_start_ms)
        self._series = series
        self.index = index
        self.phase = phase
        self._record_window_ms = record_window_ms
        self.spikes_path = spikes_path
        self.weights_path = weights_path

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

    def weights(self) -> np.ndarray:
        """重みの値ベクトル。**wiring と本数が合うことをここで確かめる。**

        合わないまま下流へ流すと、ブロック分けが 1 本ずつずれた図と指標になる
        (どこも例外を出さない)。
        """
        values = load_weight_values(self.weights_path)
        wiring = self.wiring()
        if values.size != wiring.row.size:
            raise ValueError(
                f"{self.weights_path} の重み {values.size} 本が phase={self.phase} の"
                f" 結合 {wiring.row.size} 本と一致しません。"
            )
        return values

    def coo(self) -> Coo:
        wiring = self.wiring()
        return Coo(row=wiring.row, col=wiring.col, weights=self.weights(),
                   delays=None, shape=wiring.shape)

    def spikes(self) -> Spikes:
        return load_spikes(self.spikes_path)


class Series(runview.Series):
    """完走した run 1 つ。`open_run()` が作る。"""

    def __init__(self, run_dir: Path, config, layout, wiring: Wiring, wiring_pre: Wiring,
                 windows: tuple[Window, ...] = ()):
        super().__init__(run_dir, config, layout)
        self._wiring = wiring
        self._wiring_pre = wiring_pre
        self._windows = windows

    @property
    def windows(self) -> tuple[Window, ...]:
        return self._windows

    def wiring(self) -> Wiring:
        """**切断後**の結合。run 全体を代表するのはこちら。"""
        return self._wiring

    def wiring_pre(self) -> Wiring:
        """切断**前** (Phase 1) の結合。baseline の probe だけがこれを使う。"""
        return self._wiring_pre

    def metrics(self) -> pd.DataFrame:
        """`metrics.csv` を読む。**呼ばれた時点で読む** (開いた時点ではない)。

        再解析は「開く → 指標を書き直す → 図を描く」の順に進むので、先に読むと図が
        前回の値を描く。
        """
        path = paths.data_path(self.run_dir, METRICS_NAME)
        if not path.exists():
            raise MissingData(METRICS_NAME, f"{path} がありません")
        return pd.read_csv(path)

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
        raise FileNotFoundError(f"{probe_filename(SPIKES, 0)} 形式の記録がありません: {data_dir}")
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
    return Window(series, index=index, hour=start_ms / MS_PER_HOUR,
                  record_start_ms=start_ms, record_window_ms=window_ms, phase=phase,
                  spikes_path=spikes_path, weights_path=weights_path)
