"""描画モジュール間で共有する最小限のプリミティブ。

`plotting/` の各モジュールは「1 種類の図」を担当する対等な兄弟。共有物をここへ集める
ことで、どの図も他の図の内部を借りずに済む。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# 送信種別 × 受信種別の描画色 (ブロック名の正準順は analysis.weights.BLOCK_ORDER)。
# E/I ブロック別の図はすべてこの色を使う。どの図でも EE が同じ色であることが要点。
BLOCK_COLORS = {
    "EE": "tab:red",
    "EI": "tab:orange",
    "IE": "tab:blue",
    "II": "tab:purple",
}


def save_figure(fig, out_path: Path, *, dpi: int | None = 200,
                bbox_inches: str | None = None, tight_layout: bool = True) -> None:
    """図を保存して閉じる。親ディレクトリは必要なら作る。

    **`plotting/` が図をファイルにする唯一の出口。**

    `dpi` / `bbox_inches` / `tight_layout` は図の種類ごとに必要な値が違うので引数。
    `dpi=None` は matplotlib の既定 (rcParams) に任せるという意味。

    **既定値も、呼び出し側が渡している値も変えないこと。** 変えると既存の figure が
    すべて差し替わる。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        fig.tight_layout()
    fig.savefig(out_path, bbox_inches=bbox_inches,
                **({} if dpi is None else {"dpi": dpi}))
    plt.close(fig)
