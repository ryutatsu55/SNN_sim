"""**記録窓ごと**に出すもの。

本番の run も再解析もここを通る。**記録窓ごとの出力を足すときに開くのはここ 1 か所。**

`metrics` を受け取るのはこの段階だけ。`metrics.csv` は記録窓をまたいで 1 行ずつ
追記するので窓で閉じない —— 途中で落ちた run もそこまでの指標が残る。
"""
from __future__ import annotations

from scripts.develop.report import guard
from scripts.develop.store import paths
from scripts.develop.analysis.metrics import build_row
from scripts.develop.figures.avalanche import avalanche_distribution
from scripts.develop.figures.raster import raster
from scripts.develop.figures.trace import neuron_trace


# **この段階で出るものの一覧。** 足すならここへ 1 行足す。
#
# `(ログ上の名前, 描く関数, 図の種類, ファイル名)`。ファイル名の `{tag}` には
# その窓を一意に指す短い文字列が入る。**種類が図ごとに違う**ので、structure /
# overview の表と違って置き場所も表が持つ。
FIGURES = (
    ("raster", raster, paths.RASTER, "raster_{tag}.png"),
    ("avalanche plot", avalanche_distribution, paths.AVALANCHE, "avalanche_{tag}.png"),
    # トレースを採っていない run では window.trace() が MissingData を投げるので、
    # この行があっても図は出ない (登録簿は「出そうとするもの」の一覧)。
    ("neuron trace", neuron_trace, paths.TRACE, "neuron_trace_{tag}.png"),
)


def emit(window, *, metrics) -> None:
    """記録時刻 1 点ぶんの出力。

    Args:
        window: `store/series.py` の `Window`。**直前に書いた npz を読み直したもの。**
        metrics: `records.MetricsWriter`。この窓の行を 1 行追記する。
    """
    # --- 表 ---
    # **`FIGURES` に入れていない。** 図と違って `(view, out_path)` ではなく、
    # 窓をまたいで開きっぱなしのライターへ 1 行足す操作だから。
    metrics.append(build_row(window))

    # --- 図 ---
    # 窓を指すのは記録時刻。`f"{hour:g}"` は有効数字 6 桁なので**目印にしか使わない**
    # (時刻の正は npz の record_start_ms が持つ)。
    tag = f"{window.hour:g}h"
    for label, draw, kind, template in FIGURES:
        guard(label, draw, window,
              paths.fig_path(window.run_dir, kind, template.format(tag=tag)))
