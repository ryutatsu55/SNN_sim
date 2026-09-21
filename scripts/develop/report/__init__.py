"""**いつ何を出すか**の登録簿。

出力のタイミングは 3 つしかない。それぞれに 1 ファイル。

    structure.py   build 直後   —— ネットワークの形 (まだ回していない)
    panels.py      記録窓ごと   —— その窓のスパイク・重み・トレース
    overview.py    run 終了後   —— 全記録窓

**「何がいつ出るか」はこの 3 ファイルを見れば尽きる。** 図も表も同じ扱いで、
1 出力 = 1 行。

不変条件: **このパッケージは matplotlib も numpy も import しない。** 橋渡ししか
しないので、計算も描画も入り込めない。破れていたら、その処理は `analysis/` か
`figures/` へ行くべきもの。
"""
from __future__ import annotations

from scripts.tools.runview import MissingData


def guard(label: str, fn, *args) -> None:
    """出力 1 つ。**失敗しても残りを止めない。**

    ここへ来る時点で npz と metrics の行は確定しているので、**数時間回した結果を
    描画の都合で失わない**ようにする。

    `MissingData` とそれ以外を分けるのが要点。「この run はそのデータを持たない」
    (`no_space` に座標が無い、トレースを採っていない) と「バグで落ちた」は、
    ログ上で区別できなければ意味がない。
    """
    try:
        fn(*args)
    except MissingData as missing:
        print(f"  skip {label}: {missing}")
    except Exception as error:
        print(f"  Warning: {label} generation failed: {error}")
