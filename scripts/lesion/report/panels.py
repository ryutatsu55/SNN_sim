"""**probe ごと**に出すもの。

本番の run も再解析もここを通る。**probe ごとの出力を足すときに開くのはここ 1 か所。**

`metrics` を受け取るのはこの段階だけ。`metrics.csv` は probe をまたいで 1 行ずつ
追記するので窓で閉じない —— 途中で落ちた run もそこまでの指標が残る。
"""
from __future__ import annotations

from scripts.lesion.report import guard
from scripts.lesion.store import paths
from scripts.lesion.store.formats import probe_token
from scripts.lesion.analysis.metrics import build_row
from scripts.lesion.figures.avalanche import avalanche_distribution
from scripts.lesion.figures.correlation import correlation_matrix
from scripts.lesion.figures.correlation_network import correlation_network
from scripts.lesion.figures.isi import isi_distribution
from scripts.lesion.figures.network import weight_network
from scripts.lesion.figures.raster import raster
from scripts.lesion.figures.synapse_hist import weight_distribution
from scripts.lesion.figures.trace import neuron_trace
from scripts.lesion.figures.weight_matrix import weight_matrix


# **この段階で出るものの一覧。** 足すならここへ 1 行足す。
#
# `(ログ上の名前, 描く関数, 図の種類, ファイル名)`。ファイル名の `{tag}` には
# その窓を一意に指す短い文字列が入る。**種類が図ごとに違う**ので、structure /
# overview の表と違って置き場所も表が持つ。
FIGURES = (
    ("raster", raster, paths.RASTER, "raster_{tag}.png"),
    ("avalanche plot", avalanche_distribution, paths.AVALANCHE, "avalanche_{tag}.png"),
    ("weight matrix", weight_matrix, paths.WEIGHT, "weight_matrix_{tag}.png"),
    # 機能的結合 (Watanabe et al. 2026 Fig. 2b)。重み行列と同じ並びで出るので、
    # 「繋がっている」と「実際に同期している」を 2 枚並べて読める。
    ("correlation matrix", correlation_matrix, paths.CORRELATION,
     "correlation_{tag}.png"),
    # 同じ相関を空間に置いたもの (論文 Fig. 3c 上段)。座標を持たない run (`no_space`)
    # では window.coords() が MissingData を投げるので図は出ない。
    ("correlation network", correlation_network, paths.CORRELATION_NETWORK,
     "correlation_network_{tag}.png"),
    # スパイク間隔の分布。平均発火率だけでは見えない「たまに高頻度で撃つ」が読める。
    ("ISI distribution", isi_distribution, paths.ISI, "isi_{tag}.png"),
    ("weight distribution", weight_distribution, paths.WEIGHT_DIST,
     "weight_distribution_{tag}.png"),
    # 座標を持たない run (`no_space`) では window.coords() が MissingData を投げるので、
    # この行があっても図は出ない (登録簿は「出そうとするもの」の一覧)。
    ("weight network", weight_network, paths.WEIGHT_NETWORK, "weight_network_{tag}.png"),
    # 膜電位を採っていない probe では window.voltage() が MissingData を投げるので、
    # この行があっても図は出ない (登録簿は「出そうとするもの」の一覧)。
    ("neuron trace", neuron_trace, paths.TRACE, "neuron_trace_{tag}.png"),
)


def emit(window, *, metrics) -> None:
    """probe 1 点ぶんの出力。

    Args:
        window: `store/views.py` の `Window`。**直前に書いた npz を読み直したもの。**
        metrics: `formats.MetricsWriter`。このprobeの行を 1 行追記する。
    """
    # --- 表 ---
    # **`FIGURES` に入れていない。** 図と違って `(view, out_path)` ではなく、
    # 窓をまたいで開きっぱなしのライターへ 1 行足す操作だから。
    metrics.append(build_row(window))

    # --- 図 ---
    # 窓を指すのは probe の token (`p003` / `baseline`)。**npz と同じもの**を使うので、
    # 図から元の記録を引くのに読み替えが要らない。**時刻はファイル名に埋めない** ——
    # 切断前の probe は負の時刻を持ち、間隔が細かいと float の往復で戻らないため。
    tag = probe_token(window.index)
    for label, draw, kind, template in FIGURES:
        guard(label, draw, window,
              paths.fig_path(window.run_dir, kind, template.format(tag=tag)))
