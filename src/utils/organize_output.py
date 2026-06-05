"""
実験出力ディレクトリのデータファイルを整理するスクリプト。
npz, csv などの生データを data/ フォルダにまとめる。
"""
import argparse
import shutil
from pathlib import Path


def organize_output(output_dir: Path, patterns: list[str] | None = None, dry_run: bool = False) -> None:
    """
    output_dir 内のデータファイルを data/ フォルダにまとめる。

    Args:
        output_dir: 処理対象ディレクトリ
        patterns: 移動するファイルパターン（glob形式）。デフォルト: ['*.npz', '*.csv', 'config.yaml']
        dry_run: True の場合、実際の移動は行わずに予定をリスト表示
    """
    if patterns is None:
        patterns = ['*.npz', '*.csv', 'config.yaml']

    data_dir = output_dir / 'data'

    # 移動予定のファイルを収集
    files_to_move = []
    for pattern in patterns:
        for file_path in output_dir.glob(pattern):
            if file_path.is_file():
                files_to_move.append(file_path)

    if not files_to_move:
        print("移動対象ファイルが見つかりません。")
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
            print(f"Moved: {file_path.name} → data/")
        print(f"\n✓ データを整理しました: {data_dir}")


def restore_output(output_dir: Path, dry_run: bool = False) -> None:
    """
    data/ フォルダのファイルを親ディレクトリに戻す。

    Args:
        output_dir: 処理対象ディレクトリ
        dry_run: True の場合、実際の移動は行わずに予定をリスト表示
    """
    data_dir = output_dir / 'data'

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
            print(f"Restored: data/{file_path.name} → {file_path.name}")
        print(f"\n✓ データを復元しました。")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="実験出力ディレクトリを整理し、データファイルを data/ フォルダにまとめます。"
    )
    parser.add_argument("output_dir", type=str, help="実験出力ディレクトリ")
    parser.add_argument(
        "--restore",
        action="store_true",
        help="data/ フォルダのファイルを上位ディレクトリに戻す",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実際の移動は行わず、予定をリスト表示",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        print(f"エラー: {output_dir} が見つかりません。")
        return 1

    if args.restore:
        restore_output(output_dir, dry_run=args.dry_run)
    else:
        organize_output(output_dir, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
