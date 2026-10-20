"""実験出力ディレクトリ (run ディレクトリ) の規約と、その作成・整理・探索。

**run ディレクトリの構造とファイル名を知っているのはこのモジュールだけ**、という状態を
保つこと。解析・可視化側 (`src/utils/`) はここの定数と `locate()` を import して使い、
`"data"` や `"config.yaml"` といったリテラルを自前で持たない。

run ディレクトリの中身:

    outputs/<name>/<timestamp>/
    ├── config.yaml          … 解決後 config (seed/backend/assignment は実値)。再実行用の記録
    ├── source_config.yaml   … resolve() に渡した入力 YAML の逐語コピー
    ├── layout_axes.npz      … 外部軸 (layer / module …)。config からは再導出できない
    ├── connectivity.npz     … 疎 (COO) 経路での row/col/shape。run につき 1 回
    ├── axon_geometry.npz    … 軸索の折れ線と接触点。connection: axon_growth 系のときだけ
    └── data/                … organize_output() 後は上記と npz/csv がここへ移る
"""
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# run ディレクトリの規約 (ファイル名・サブディレクトリ名の唯一の定義)
# --------------------------------------------------------------------------- #
DATA_SUBDIR = "data"
CONFIG_NAME = "config.yaml"
SOURCE_CONFIG_NAME = "source_config.yaml"
AXES_NAME = "layout_axes.npz"
CONNECTIVITY_NAME = "connectivity.npz"
# 軸索の折れ線 (AxonGeometry)。connection が axon_growth 系のときだけ書かれる。
# 「どの軸索がどのブリッジを通ったか」= 損傷実験で必要になる記録。
AXONS_NAME = "axon_geometry.npz"

# locate() が探すサブディレクトリ。organize_output() が data/ へ移動するため、run ルートを
# 渡された場合と data/ を直接渡された場合の両方を受け付ける。
_SEARCH_SUBDIRS = ("", DATA_SUBDIR)


def locate(run_dir: Path | str, filename: str) -> Optional[Path]:
    """run_dir 直下、無ければ data/ 配下から filename を探す。見つからなければ None。

    `organize_output()` がデータファイルを data/ へ移すため、同じ run ディレクトリでも
    整理前後でファイルの位置が変わる。その差を吸収する唯一の入口。
    """
    run_dir = Path(run_dir)
    for subdir in _SEARCH_SUBDIRS:
        candidate = (run_dir / subdir / filename) if subdir else (run_dir / filename)
        if candidate.exists():
            return candidate
    return None


def data_dir(run_dir: Path | str) -> Path:
    """run ディレクトリの中で、データファイルが実際に置かれているディレクトリを返す。

    `organize_output()` 前なら run ルート、後なら `<run_dir>/data`。判定は config.yaml の
    所在で行う (npz と一緒に移動するため)。`weights_*h.npz` のように glob で列挙する
    処理は、run ルートを直接 glob せずここを経由すること。
    """
    config_path = locate(run_dir, CONFIG_NAME)
    return config_path.parent if config_path is not None else Path(run_dir)


def require(run_dir: Path | str, filename: str) -> Path:
    """`locate()` と同じだが、見つからなければ例外を投げる。"""
    path = locate(run_dir, filename)
    if path is None:
        raise FileNotFoundError(
            f"{filename} が見つかりません: {run_dir} "
            f"(探索先: {run_dir}, {Path(run_dir) / DATA_SUBDIR})"
        )
    return path


def _sanitize_dir_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    sanitized = sanitized.strip("._-")
    return sanitized or "simulation"


def create_run_output_dir(
    simulation_name: str,
    base_dir: str | Path = "outputs",
    timestamp: str | None = None,
) -> Path:
    """シミュレーション1回分の出力ディレクトリを作成する。"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    output_dir = Path(base_dir) / _sanitize_dir_name(simulation_name) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_timestamped_output_dir(
    base_dir: str | Path,
    timestamp: str | None = None,
    suffix: str | None = None,
) -> Path:
    """指定されたベースディレクトリ直下に日時ディレクトリを作成する。

    suffix を渡すと ``<timestamp>_<suffix>`` というディレクトリ名になり、
    複数プロセスを同一秒に並列起動してもディレクトリが衝突しない。
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    name = f"{timestamp}_{_sanitize_dir_name(suffix)}" if suffix else timestamp
    output_dir = Path(base_dir) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def organize_output(output_dir: Path, patterns: list[str] | None = None, dry_run: bool = False) -> None:
    """
    output_dir 内のデータファイルを data/ フォルダにまとめる。

    Args:
        output_dir: 処理対象ディレクトリ
        patterns: 移動するファイルパターン（glob形式）。デフォルト: ['*.npz', '*.csv', 'config.yaml']
        dry_run: True の場合、実際の移動は行わずに予定をリスト表示
    """
    if patterns is None:
        patterns = ['*.npz', '*.csv', CONFIG_NAME]

    data_dir = output_dir / DATA_SUBDIR

    # 移動予定のファイルを収集
    files_to_move = []
    for pattern in patterns:
        for file_path in output_dir.glob(pattern):
            if file_path.is_file():
                files_to_move.append(file_path)

    if not files_to_move:
        return

    # data/ ディレクトリが既に存在し、同名ファイルがあるかチェック
    if data_dir.exists():
        conflicts = [f for f in files_to_move if (data_dir / f.name).exists()]
        if conflicts:
            print(f"警告: data/ フォルダに同名ファイルが既に存在します:")
            for f in conflicts:
                print(f"  - {f.name}")
            return

    if dry_run:
        print("=== Dry run: 以下のファイルが移動される予定です ===")
        for file_path in files_to_move:
            print(f"  {file_path.name} → data/")
        print(f"\n実際に実行するには --dry-run フラグを削除してください。")
    else:
        data_dir.mkdir(exist_ok=True)
        for file_path in files_to_move:
            dest_path = data_dir / file_path.name
            file_path.rename(dest_path)
        print(f"Organized output: {data_dir}")


def restore_output(output_dir: Path, dry_run: bool = False) -> None:
    """
    data/ フォルダのファイルを親ディレクトリに戻す。

    Args:
        output_dir: 処理対象ディレクトリ
        dry_run: True の場合、実際の移動は行わずに予定をリスト表示
    """
    data_dir = output_dir / DATA_SUBDIR

    if not data_dir.exists():
        print(f"警告: {data_dir} が見つかりません。")
        return

    files_to_restore = list(data_dir.glob('*'))
    if not files_to_restore:
        print(f"警告: {data_dir} は空です。")
        return

    # 上位ディレクトリに同名ファイルがあるかチェック
    conflicts = [f for f in files_to_restore if (output_dir / f.name).exists()]
    if conflicts:
        print(f"警告: 上位ディレクトリに同名ファイルが既に存在します:")
        for f in conflicts:
            print(f"  - {f.name}")
        return

    if dry_run:
        print("=== Dry run: 以下のファイルが移動される予定です ===")
        for file_path in files_to_restore:
            print(f"  data/{file_path.name} → {file_path.name}")
        print(f"\n実際に実行するには --dry-run フラグを削除してください。")
    else:
        for file_path in files_to_restore:
            dest_path = output_dir / file_path.name
            file_path.rename(dest_path)
        print(f"Restored output.")

