"""**いつ何を出すか**の登録簿。

出力のタイミングは 3 つしかない。それぞれに 1 ファイル。

    structure.py   build 直後   —— ネットワークの形 (まだ回していない)
    panels.py      記録窓ごと   —— その窓のスパイク・重み・トレース
    overview.py    run 終了後   —— 全記録窓

**「何がいつ出るか」はこの 3 ファイルを見れば尽きる。** 各ファイルは先頭に `FIGURES`
という表を持ち、`emit()` は**その表を上から順に回すだけ**。図を足すのは表に 1 行足す
ことで、`emit()` は触らない。

表に入らない出力もある —— 置き場所が `figures/` ではないもの (CSV) と、`(view, out_path)`
の形をしていないもの (`metrics.csv` への 1 行追記)。それらは `emit()` に明示行として
残し、**なぜ表に入らないか**をその場に書いてある。表が「出力の全部」ではなく「図の全部」
であることは、読む側が知っている必要がある。

不変条件: **このパッケージは matplotlib も numpy も import しない。** 橋渡ししか
しないので、計算も描画も入り込めない。破れていたら、その処理は `analysis/` か
`figures/` へ行くべきもの。
"""
from __future__ import annotations

# `guard()` の実体は契約と対になるので `src/utils/runview.py` にある
# (`MissingData` を投げる側と受ける側を離さない)。ここは登録簿からの呼び名を揃える
# ための再エクスポート —— `report.guard(...)` と書けることに意味がある。
from src.utils.runview import guard  # noqa: F401

__all__ = ["guard"]
