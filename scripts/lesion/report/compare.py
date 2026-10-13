"""**run をまたいで**出すもの。損傷 run の束 (`Collection`) を比べる図。

run 1 本の 3 段階 (structure / panels / overview) の外側にある 4 つ目の段階。形は
overview と同じで、表を書いてから、その表を読む図を描く。出力先は
`<条件>/lesion_compare/` (`paths.compare_dir()`)。
"""
from __future__ import annotations

from scripts.lesion.report import guard
from scripts.lesion.store import paths
from scripts.lesion.analysis.summary import write_summary
from scripts.lesion.figures.dose_response import (activity_drop, cut_weight,
                                                  recovery_vs_cut, recovery_vs_drop)


# **この段階で出る図の一覧。** 足すならここへ 1 行足す。
# `(ログ上の名前, 描く関数, ファイル名)`。渡す view は束全体の `Collection`。
FIGURES = (
    ("cut weight", cut_weight, "cut_weight.png"),
    ("activity drop", activity_drop, "activity_drop.png"),
    ("recovery vs drop", recovery_vs_drop, "recovery_vs_drop.png"),
    ("recovery vs cut", recovery_vs_cut, "recovery_vs_cut.png"),
)


def emit(collection) -> None:
    """記録の表を `lesion_compare/` へ書き、図を `lesion_compare/report/` へ。"""
    # --- 記録 (図より先) ---
    # **guard で包まない。** 図は全部この表を読むので、書けなかったときに続けると
    # 前回の表で描いてしまう。
    write_summary(collection, collection.out_dir)

    for label, draw, name in FIGURES:
        guard(label, draw, collection, paths.compare_report_path(collection.cond_dir, name))
