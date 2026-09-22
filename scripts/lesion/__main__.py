"""lesion 実験のランチャ (親プロセス)。

    python -m scripts.lesion --parent <親 run または条件ディレクトリ> --condition <条件名>

**引数は「どの run を引き継ぐか」「どの損傷プロトコルか」「どの条件として記録するか」の
3 つだけ。** それ以外 (切断 spec・タイムライン・解析オプション) はすべて `task.yaml` 側に
ある。

## なぜ develop と違って `--config` を取らないのか

lesion の**ネットワーク設定は親 run から来る**。同じ seed で同じネットワークを再ビルド
しないと重みを復元できないので、config を別に指定する余地が無い。代わりに親を指す。

ランチャがやること:

1. `--parent` の下から run ディレクトリを探す (条件ディレクトリなら `seedNN/` を列挙)
2. 各親の `config.yaml` を読む = その run のネットワーク設定
3. `task` をこのディレクトリの `task.yaml` の損傷プロトコルへ**差し替える**
4. `task.parent_run` に親のパスを焼き込む
5. `outputs/lesion/<条件>/` を作り、seed ごとの run ディレクトリへ引き継ぎ config を置く
6. `run_one` を子プロセスとして並列に起動し、各々の出力を `<run>/run.log` へ流す

**3 と 4 が要点。** おかげで run ディレクトリの `config.yaml` だけで「どのネットワークを、
どこで切って、どう観察したか」が全部読める。ディレクトリ名からは何も読み取らない。

出力の形:

    outputs/lesion/<条件>/            親が 1 本  -> ここが run ディレクトリそのもの
    outputs/lesion/<条件>/seed01/     親が複数   -> ここが run ディレクトリ
                        /seed02/
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from src.core.config_manager import ConfigManager, load_yaml
from src.core.output_manager import CONFIG_NAME

from scripts.lesion.store.paths import PENDING_CONFIG_NAME

_HERE = Path(__file__).resolve().parent

# この実験の名前。出力の第 1 階層になる。
EXPERIMENT = "lesion"
OUTPUT_ROOT = Path("outputs")
# 損傷プロトコル。**記録プロトコルと損傷条件の両方**がここに入る。
TASK_PATH = _HERE / "task.yaml"
DEFAULT_TASK = "lesion"
RUN_LOG_NAME = "run.log"
# 並列度。親 run の config には lesion 用の値が無いので、ここだけ引数で受ける
# (**結果に影響しない**ので引数にしてよい。CLAUDE.md の線引き)。
DEFAULT_PARALLEL = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="育った run の結合を切って回復を追う (親 run ごとに子プロセスを並列起動)。")
    parser.add_argument(
        "--parent", required=True,
        help="引き継ぐ親 run。run ディレクトリそのものでも、seedNN/ を並べた条件ディレクトリでもよい")
    parser.add_argument(
        "--condition", required=True,
        help="条件名。outputs/lesion/<条件>/ になる。**条件の記述ではなくラベル** "
             "(何で走ったかは run ディレクトリの config.yaml が持つ)")
    parser.add_argument(
        "--task", default=DEFAULT_TASK,
        help=f"task.yaml の損傷プロトコル名 (既定: {DEFAULT_TASK})")
    parser.add_argument(
        "--parallel", type=int, default=DEFAULT_PARALLEL,
        help="同時に走らせる run の数。GeNN のコンパイルが run につき 2 回走るので控えめに")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="GeNN を触らず、切断対象を数えて終了 (子へそのまま渡す)")
    return parser.parse_args()


def discover_parents(parent: Path) -> list[Path]:
    """親 run を列挙する。run ディレクトリそのものでも条件ディレクトリでもよい。

    判定は `config.yaml` の有無だけ。**ディレクトリ名は見ない** (`seedNN` という名前に
    意味を持たせると、名前と中身がずれたときに黙って間違う)。
    """
    parent = Path(parent)
    if (parent / CONFIG_NAME).exists():
        return [parent]
    found = sorted(child for child in parent.iterdir()
                   if child.is_dir() and (child / CONFIG_NAME).exists())
    if not found:
        raise SystemExit(
            f"{parent} の下に config.yaml を持つ run が見つかりません。"
            " 親 run ディレクトリか、それを並べた条件ディレクトリを指してください。")
    return found


def load_protocol(name: str) -> dict:
    """`task.yaml` から損傷プロトコルを読む。"""
    profiles = load_yaml(TASK_PATH)
    if name not in profiles:
        raise SystemExit(
            f"損傷プロトコル {name!r} が {TASK_PATH} にありません "
            f"(あるもの: {sorted(profiles)})")
    protocol = dict(profiles[name])
    protocol["profile_name"] = name
    return protocol


def prepare_runs(manager: ConfigManager, parents: list[Path], protocol: dict,
                 condition: str) -> list[Path]:
    """`<条件>/` と各 run ディレクトリを作り、引き継ぎ config を置く。

    置くのは `config.yaml` ではなく `pending_config.yaml`。**ここで書けるのは記録ではなく
    引き継ぎだから** —— `network.sparse` の実値は build 時にしか決まらない。
    `config.yaml` を書くのは build を通した run 本体の仕事 (`run_one.py`)。
    """
    condition_dir = OUTPUT_ROOT / EXPERIMENT / condition
    if condition_dir.exists():
        raise SystemExit(
            f"条件 {condition!r} は既にあります: {condition_dir}\n"
            "  上書きすると前の結果と混ざるので止めました。"
            " 別の条件名を付けるか、不要なら手で消してください。")
    condition_dir.mkdir(parents=True)

    run_dirs = []
    for parent in parents:
        config = manager.load_resolved(parent / CONFIG_NAME)
        seed = int(config.simulation.seed)
        run_dir = condition_dir if len(parents) == 1 else condition_dir / f"seed{seed:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # **task を丸ごと差し替える。** 親の記録プロトコル (develop の record_hours など)
        # はこの run では使わないので残さない。親のパスをここで焼き込むことで、
        # run ディレクトリの config.yaml だけで何が起きたかが読めるようになる。
        merged = dict(protocol)
        merged["parent_run"] = str(parent.resolve())
        config.task = type(config.task).model_validate(merged)
        manager.dump_config(config, run_dir / PENDING_CONFIG_NAME)
        run_dirs.append(run_dir)
    return run_dirs


def launch(run_dir: Path, dry_run: bool) -> int:
    """run_one を子プロセスとして起動し、出力を <run>/run.log へ流す。

    親がログの行き先を決めるので、シェル側のリダイレクトも EXIT トラップも要らない。
    """
    # `-u` で無バッファにする。子の stdout はファイルなので、既定ではブロック
    # バッファされて**走り終えるまで run.log が空のまま**になる。長い run の進捗を
    # 追えないうえ、途中で落ちたときに直前の出力ごと失われる。
    command = [sys.executable, "-u", "-m", "scripts.lesion.run_one", str(run_dir)]
    if dry_run:
        command.append("--dry-run")
    log_path = run_dir / RUN_LOG_NAME
    with open(log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=str(project_root), stdout=log,
                                   stderr=subprocess.STDOUT)
    return completed.returncode


def main():
    args = parse_args()
    os.chdir(project_root)

    parents = discover_parents(Path(args.parent))
    protocol = load_protocol(args.task)
    manager = ConfigManager()

    parallel = max(1, min(args.parallel, len(parents)))
    print(f"parent    : {args.parent}  ({len(parents)} run)")
    print(f"task      : {args.task}")
    print(f"condition : {args.condition}")
    print(f"cut       : {protocol.get('cut')}")
    print(f"parallel  : {parallel}")

    run_dirs = prepare_runs(manager, parents, protocol, args.condition)
    for parent, run_dir in zip(parents, run_dirs):
        print(f"  {parent} -> {run_dir}")

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        codes = list(pool.map(lambda d: launch(d, args.dry_run), run_dirs))

    failed = [d for d, code in zip(run_dirs, codes) if code != 0]
    print()
    if failed:
        print(f"失敗した run が {len(failed)} / {len(run_dirs)} 本あります:", file=sys.stderr)
        for run_dir in failed:
            print(f"  {run_dir}  (ログ: {run_dir / RUN_LOG_NAME})", file=sys.stderr)
        print("  再実行: python -m scripts.lesion.run_one <run ディレクトリ>", file=sys.stderr)
        raise SystemExit(1)
    print(f"完了しました: {OUTPUT_ROOT / EXPERIMENT / args.condition}")


if __name__ == "__main__":
    main()
