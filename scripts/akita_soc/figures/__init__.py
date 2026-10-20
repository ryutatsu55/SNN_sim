"""図を描く層。**1 ファイル = 1 種類の図。**

各関数は `(view, out_path)` を受ける。`view` は `src/utils/runview.py` の契約
(`Built` / `Window` / `Series`) で、必要なデータはそこから自分で取る。

    def raster(window, out_path) -> None:
        spikes = window.spikes()
        ...

**「いつ描くか」「どこに置くか」は知らない。** それを決めるのは `report/`。
無いデータを要求したときは `MissingData` をそのまま上へ投げる —— 図の側に
「持っていなければスキップ」の分岐を書かない (受けるのは `report.guard()`)。

このパッケージが持つのは `save()` だけ。見た目の値のうち**2 枚以上が一致すべきもの**は
`style.py`、1 枚しか使わないものは各図のファイル先頭。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save(fig, out_path, *, tight_layout: bool = True, **savefig_kwargs) -> None:
    """図を書き出す。**png を作る唯一の出口。**

    親ディレクトリを作り、保存し、figure を閉じる。dpi や bbox_inches は
    `savefig_kwargs` としてそのまま渡す。**既定値は持たない** —— 解像度は各図が決める。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        fig.tight_layout()
    fig.savefig(out_path, **savefig_kwargs)
    plt.close(fig)
