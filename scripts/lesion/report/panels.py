"""**probe ごと**に出すもの。

本番の run も再解析もここを通る。**probe ごとの出力を足すときに開くのはここ 1 か所。**

`metrics` を受け取るのはこの段階だけ。`metrics.csv` は probe をまたいで 1 行ずつ
追記するので窓で閉じない —— 途中で落ちた run もそこまでの指標が残る。
"""
from __future__ import annotations

from scripts.lesion.report import guard
from scripts.lesion.store import paths
from scripts.lesion.analysis.metrics import build_row
from scripts.lesion.figures.avalanche import avalanche_distribution
from scripts.lesion.figures.raster import raster


# **この段階で出るものの一覧。** 足すならここへ 1 行足す。
#
# `(ログ上の名前, 描く関数, 図の種類, ファイル名)`。ファイル名の `{tag}` には
# その窓を一意に指す短い文字列が入る。**種類が図ごとに違う**ので、structure /
# overview の表と違って置き場所も表が持つ。
FIGURES = (
    ("raster", raster, paths.RASTER, "raster_{tag}.png"),
    ("avalanche plot", avalanche_distribution, paths.AVALANCHE, "avalanche_{tag}.png"),
)


def emit(window, *, metrics) -> None:
    """probe 1 点ぶんの出力。

    Args:
        window: `store/series.py` の `Window`。**直前に書いた npz を読み直したもの。**
        metrics: `records.MetricsWriter`。このprobeの行を 1 行追記する。
    """
    # --- 表 ---
    # **`FIGURES` に入れていない。** 図と違って `(view, out_path)` ではなく、
    # 窓をまたいで開きっぱなしのライターへ 1 行足す操作だから。
    metrics.append(build_row(window))

    # --- 図 ---
    # 窓を指すのは probe の番号と phase。**時刻はファイル名に埋めない** ——
    # 切断前の probe は負の時刻を持ち、間隔が細かいと float の往復で戻らないため。
    tag = f"p{window.index:03d}_{window.phase}"
    for label, draw, kind, template in FIGURES:
        guard(label, draw, window,
              paths.fig_path(window.run_dir, kind, template.format(tag=tag)))
