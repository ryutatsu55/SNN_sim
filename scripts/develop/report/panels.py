"""**記録窓ごと**に出すもの。

本番の run も再解析もここを通る。**記録窓ごとの出力を足すときに開くのはここ 1 か所**
—— 以前は `run_one` と `replot` が別々に描画を呼んでいたので、片方に足し忘れると
「本番には出るが再解析では出ない図」ができた。

`metrics` を受け取るのはこの段階だけ。`metrics.csv` は記録窓をまたいで 1 行ずつ
追記するので窓で閉じない (72 時間の run が 50 時間で落ちても、そこまでの指標が残る)。
隠すべき事情ではなくこの段階の性質そのものなので、引数に出しておく。
"""
from __future__ import annotations

from scripts.develop.report import guard
from scripts.develop.store import paths
from scripts.develop.analysis.metrics import build_row
from scripts.develop.figures.avalanche import avalanche_distribution
from scripts.develop.figures.raster import raster
from scripts.develop.figures.trace import neuron_trace


def emit(window, *, metrics) -> None:
    """記録時刻 1 点ぶんの出力。

    Args:
        window: `store/series.py` の `Window`。**直前に書いた npz を読み直したもの。**
        metrics: `records.MetricsWriter`。この窓の行を 1 行追記する。
    """
    run_dir, hour = window.run_dir, window.hour

    # --- 表 ---
    metrics.append(build_row(window))

    # --- 図 ---
    guard("raster", raster, window,
          paths.fig_path(run_dir, paths.RASTER, f"raster_{hour:g}h.png"))
    guard("avalanche plot", avalanche_distribution, window,
          paths.fig_path(run_dir, paths.AVALANCHE, f"avalanche_{hour:g}h.png"))
    # トレースを採っていない run では window.trace() が MissingData を投げる。
    guard("neuron trace", neuron_trace, window,
          paths.fig_path(run_dir, paths.TRACE, f"neuron_trace_{hour:g}h.png"))
