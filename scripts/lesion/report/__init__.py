"""**いつ何を出すか**の登録簿。

出力のタイミングは 3 つしかない。それぞれに 1 ファイル。

    structure.py   build 直後   —— ネットワークの形 (まだ回していない)
    panels.py      記録窓ごと   —— その窓のスパイク・重み・トレース
    overview.py    run 終了後   —— 全記録窓

**「何がいつ出るか」はこの 3 ファイルを見れば尽きる。** 出力は 2 種類ある。

- **報告** (人が読むもの) —— 図 (`FIGURES`) と表 (`TABLES`)。各ファイルの先頭にある
  この 2 つの表に並び、`emit()` は**その表を上から順に回すだけ**。置き場所は
  `<run>/report/<段階>/` で、消しても再解析 (`replot.py`) で作り直せる。報告を足すのは
  表に 1 行足すことで、`emit()` は触らない。
- **記録** (測った値と、プログラムが使う計算結果) —— `data/` に置く CSV。`metrics.csv` への
  1 行追記のように `(view, out_path)` の形をしていないので、`emit()` に明示行として残し、
  **なぜ表に入らないか**をその場に書いてある。

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
