"""**run をまたいで**出すもの。損傷 run の束 (`Collection`) を比べる図と表。

run 1 本の 3 段階 (structure / panels / overview) の外側にある 4 つ目の段階。形は
overview と同じで、表を書いてから、その表を読む図を描く。出力先は
`<条件>/lesion_compare/` (`paths.compare_dir()`)。
"""
from __future__ import annotations

from src.utils.runview import MissingData

from scripts.lesion.report import guard
from scripts.lesion.store import paths
from scripts.lesion.analysis.detours import (measure_collection, write_last_hop_summary,
                                             write_path_summary, write_reverse_summary,
                                             write_turnover_summary)
from scripts.lesion.analysis.summary import write_summary
from scripts.lesion.figures.dose_response import (activity_drop, cut_weight,
                                                  recovery_vs_corr_drop, recovery_vs_cut,
                                                  recovery_vs_drop)


# **この段階で出る図の一覧。** 足すならここへ 1 行足す。
# `(ログ上の名前, 描く関数, ファイル名)`。渡す view は束全体の `Collection`。
FIGURES = (
    ("cut weight", cut_weight, "cut_weight.png"),
    ("activity drop", activity_drop, "activity_drop.png"),
    ("recovery vs drop", recovery_vs_drop, "recovery_vs_drop.png"),
    ("recovery vs corr drop", recovery_vs_corr_drop, "recovery_vs_corr_drop.png"),
    ("recovery vs cut", recovery_vs_cut, "recovery_vs_cut.png"),
)

# **この段階で出る表の一覧。** 足すならここへ 1 行足す。
# `(ログ上の名前, 書く関数, ファイル名)`。渡す view は束全体の迂回経路の測定結果
# (`analysis/detours.PooledDetours`)。表の列の定義は `analysis/detours.py` の docstring。
TABLES = (
    ("detour paths", write_path_summary, "detour_paths.csv"),
    ("reverse regrowth", write_reverse_summary, "detour_reverse.csv"),
    ("last hop order", write_last_hop_summary, "detour_last_hop.csv"),
    ("E-E turnover", write_turnover_summary, "detour_turnover.csv"),
)


def emit(collection) -> None:
    """記録の表を `lesion_compare/` へ書き、図と報告の表を `lesion_compare/report/` へ。"""
    # --- 記録 (図より先) ---
    # **guard で包まない。** 図は全部この表を読むので、書けなかったときに続けると
    # 前回の表で描いてしまう。
    write_summary(collection, collection.out_dir)

    for label, draw, name in FIGURES:
        guard(label, draw, collection, paths.compare_report_path(collection.cond_dir, name))

    # 迂回経路の測定は全 run のスパイク列を読むので、**ここで 1 回だけ**測って 4 つの表に渡す
    # (表ごとに `Collection` から測り直さない)。E→E を切った run が無い束は表ごと skip。
    try:
        detours = measure_collection(collection)
    except MissingData as missing:
        print(f"  skip detours: {missing}")
        return
    for label, make, name in TABLES:
        guard(label, make, detours, paths.compare_report_path(collection.cond_dir, name))
