"""pytest 実行時の共通セットアップ。

ここが引き受けるのは**環境の準備だけ**で、フィクスチャは置かない。
個々のテストが自分で書いていたボイラープレートのうち、どのファイルでも同じになる
3 つをここへ集約する:

1. プロジェクトルートを `sys.path` に通す (`import src.*` を成立させる)
2. `MPLCONFIGDIR` をテンポラリへ逃がす (matplotlib のキャッシュ書き込み警告を消す)
3. matplotlib のバックエンドを Agg に固定する (表示の無い環境・並列実行で落ちないように)

2 と 3 は matplotlib を import する前に済ませる必要があるので、conftest でやる意味がある
(conftest はテストモジュールの import より前に読み込まれる)。

`test/models/` と `test/archive/` のスクリプトは pytest 経由では走らないので、
conftest の恩恵を受けない。あちらは今まで通り各ファイルが自分で sys.path を通す。
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
