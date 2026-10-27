"""**run をまたいで**出すもの。損傷 run の束 (`Collection`) を比べる図と表。

run 1 本の 3 段階 (structure / panels / overview) の外側にある 4 つ目の段階。形は
overview と同じで、表を書いてから、その表を読む図を描く。出力先は
`<条件>/lesion_compare/` (`paths.compare_dir()`)。
"""
from __future__ import annotations

from src.utils.runview import MissingData

from scripts.lesion.report import guard
from scripts.lesion.store import paths
from scripts.lesion.analysis import detours, flip_rule, freeze_split
from scripts.lesion.analysis.detours import (write_last_hop_summary, write_path_summary,
                                             write_reverse_summary, write_turnover_summary)
from scripts.lesion.analysis.summary import write_summary
from scripts.lesion.figures.dose_response import (activity_drop, cut_weight,
                                                  recovery_vs_corr_drop, recovery_vs_cut,
                                                  recovery_vs_drop)
from scripts.lesion.figures.freeze_split import freeze_rate_track, freeze_recovery_split


# **この段階で出る図の一覧。** 足すならここへ 1 行足す。
# `(ログ上の名前, 描く関数, ファイル名)`。渡す view は束全体の `Collection`。
FIGURES = (
    ("cut weight", cut_weight, "cut_weight.png"),
    ("activity drop", activity_drop, "activity_drop.png"),
    ("recovery vs drop", recovery_vs_drop, "recovery_vs_drop.png"),
    ("recovery vs corr drop", recovery_vs_corr_drop, "recovery_vs_corr_drop.png"),
    ("recovery vs cut", recovery_vs_cut, "recovery_vs_cut.png"),
    # 切断後に片方の STDP を止めた run と止めない run の比較。止めた run が無い束では skip。
    ("freeze rate track", freeze_rate_track, "freeze_rate_track.png"),
    ("freeze recovery split", freeze_recovery_split, "freeze_recovery_split.png"),
)

# **この段階で出る表の一覧。** 足すならここへ 1 行足す。
# `(ログ上の名前, 書く関数, 測る関数, ファイル名)`。表はどれも全 run のスパイク列を読む重い測定の
# 結果から作るので、書く関数には `Collection` ではなく**測る関数の戻り値**を渡す。`emit()` は
# 測る関数ごとに 1 回だけ測り、同じ測定を使う表で使い回す。列の定義は測る関数のモジュールの docstring。
TABLES = (
    ("detour paths", write_path_summary, detours.measure_collection, "detour_paths.csv"),
    ("reverse regrowth", write_reverse_summary, detours.measure_collection, "detour_reverse.csv"),
    ("last hop order", write_last_hop_summary, detours.measure_collection, "detour_last_hop.csv"),
    ("E-E turnover", write_turnover_summary, detours.measure_collection, "detour_turnover.csv"),
    ("flip rule", flip_rule.write_rule_summary, flip_rule.measure_collection, "flip_rule.csv"),
    ("flip rule by class", flip_rule.write_class_summary, flip_rule.measure_collection,
     "flip_rule_classes.csv"),
    ("freeze split", freeze_split.write_freeze_split, freeze_split.measure_collection,
     "freeze_split.csv"),
)


def emit(collection) -> None:
    """記録の表を `lesion_compare/` へ書き、図と報告の表を `lesion_compare/report/` へ。"""
    # --- 記録 (図より先) ---
    # **guard で包まない。** 図は全部この表を読むので、書けなかったときに続けると
    # 前回の表で描いてしまう。
    write_summary(collection, collection.out_dir)

    for label, draw, name in FIGURES:
        guard(label, draw, collection, paths.compare_report_path(collection.cond_dir, name))

    # 測定は表ごとではなく、測る関数ごとに 1 回だけ (`TABLES` の説明)。測れない束
    # (例: E→E を切った run が無い) では、その測定を使う表をまとめて skip する。
    measured = {}
    for label, make, measure, name in TABLES:
        if measure not in measured:
            try:
                measured[measure] = measure(collection)
            except MissingData as missing:
                print(f"  skip {label}: {missing}")
                measured[measure] = None
        if measured[measure] is not None:
            guard(label, make, measured[measure],
                  paths.compare_report_path(collection.cond_dir, name))
