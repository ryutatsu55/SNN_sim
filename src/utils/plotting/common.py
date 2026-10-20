"""描画モジュール間で共有する最小限のプリミティブ。

`plotting/` の各モジュールは「1 種類の図」を担当する対等な兄弟であって、どれかが
他のユーティリティ置き場を兼ねてはいけない。共有物をここへ集めることで、
`area.py` が `network.py` の内部関数を借りる (= 図の種類の間に上下関係ができ、
循環 import を関数内 import で回避する羽目になる) 構図を無くしている。
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


def save_figure(fig, out_path: Path) -> None:
    """図を保存して閉じる。保存先の親ディレクトリは必要なら作る。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
