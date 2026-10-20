"""1 つの親条件の損傷 run を束ねて比べる (用量反応の表と図、迂回経路の表)。

    python -m scripts.lesion.compare <条件ディレクトリ> --control <対照の条件ディレクトリ>

条件ディレクトリは親 run (develop) を並べたもの。その下の損傷 run を全部拾う。
対照は「切らずに走り続けた」develop 形式の run を並べたもの (切断時刻 + 20 h 以上の記録が要る)。
出力は `<条件ディレクトリ>/lesion_compare/`。GeNN は使わない。
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/snn_sim_matplotlib")

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.lesion.report import compare
from scripts.lesion.store.collection import open_collection


def parse_args():
    parser = argparse.ArgumentParser(description="損傷 run の束を比べる表と図を作る。")
    parser.add_argument("cond_dir", help="親 run を並べた条件ディレクトリ")
    parser.add_argument("--control", required=True,
                        help="対照 (切らずに走り続けた run) を並べた条件ディレクトリ")
    return parser.parse_args()


def main():
    args = parse_args()
    collection = open_collection(args.cond_dir, args.control)
    compare.emit(collection)
    print(f"出力: {collection.out_dir}")


if __name__ == "__main__":
    main()
