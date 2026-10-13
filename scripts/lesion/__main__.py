"""lesion 実験のランチャ (親プロセス)。

    python -m scripts.lesion --parent <親 run または条件ディレクトリ>

**結果を左右する引数は「どの run を引き継ぐか」(`--parent`) と「どの損傷プロトコルか」
(`--task`) の 2 つだけ。** それ以外 (切断 spec・タイムライン・解析オプション) はすべて
`task.yaml` 側にある。残りの `--parallel` (同時に走らせる数) と `--dry-run` (下見だけ) は
走らせ方を変えるだけで、run の結果は変えない。

## なぜ develop と違って `--config` を取らないのか

lesion の**ネットワーク設定は親 run から来る**。同じ seed で同じネットワークを再ビルド
しないと重みを復元できないので、config を別に指定する余地が無い。代わりに親を指す。

ランチャがやること:

1. `--parent` の下から run ディレクトリを探す (条件ディレクトリなら `seedNN/` を列挙)
2. 各親の `config.yaml` を読む = その run のネットワーク設定 (並列度もここから取る)
3. `task` をこのディレクトリの `task.yaml` の損傷プロトコルへ**差し替える**
4. `task.parent_run` に親のパスを焼き込む
5. **親 run の下に** `<切断条件>/` を作り、`config.yaml` を置く
6. `run_one` を子プロセスとして並列に起動し、各々の出力を `<run>/run.log` へ流す

**3 と 4 が要点。** おかげで run ディレクトリの `config.yaml` だけで「どのネットワークを、
どこで切って、どう観察したか」が全部読める。ディレクトリ名からは何も読み取らない。

## 出力は親 run の下

    <親 run>/bridge-inter_cluster/          親が 1 本
    <親条件>/seed01/bridge-inter_cluster/   親が sweep なら seed ごとに 1 つずつ
    <親条件>/seed02/bridge-inter_cluster/

**`outputs/lesion/` は使わない。** 損傷 run は親 run に完全に従属していて、親を消したら
再現も再解析もできないので、置き場所も親に従属させる。ディレクトリ名 `<切断条件>` は
切断 spec から機械的に決まる (`selectors.cut_label()`) ので、**名前を引数で受けない** ——
名前と中身がずれる余地を作らないため。同じ親を同じ spec で 2 度切ることはできない
(既にあれば止まる)。

`--dry-run` の下見は `<切断条件>.dry-run/` に作る。本番と同じ名前にすると、下見の後に
本番が「既にあります」で止まる。下見のディレクトリには `config.yaml` と `run.log` だけが
残り、切断本数などの結果は `run.log` に出る。要らなくなったら手で消す。
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


from src.core.config_manager import (CONFIG_NAME, ConfigManager, expand_hours_spec,
                                     load_yaml)

from scripts.lesion.analysis.selectors import cut_label

_HERE = Path(__file__).resolve().parent

# 損傷プロトコル。**記録プロトコルと損傷条件の両方**がここに入る。
TASK_PATH = _HERE / "task.yaml"
DEFAULT_TASK = "lesion"
RUN_LOG_NAME = "run.log"
# `--dry-run` の下見を置くディレクトリ名の印。人が見分けるためだけのもので、読み取らない。
DRY_RUN_SUFFIX = ".dry-run"


def parse_args():
    parser = argparse.ArgumentParser(
        description="育った run の結合を切って回復を追う (親 run ごとに子プロセスを並列起動)。")
    parser.add_argument(
        "--parent", required=True,
        help="引き継ぐ親 run。run ディレクトリそのものでも、seedNN/ を並べた条件ディレクトリでもよい。"
             "**損傷 run はこの下に <切断条件>/ として作られる**")
    parser.add_argument(
        "--task", default=DEFAULT_TASK,
        help=f"task.yaml の損傷プロトコル名 (既定: {DEFAULT_TASK})")
    parser.add_argument(
        "--parallel", type=int, default=None,
        help="同時に走らせる run の数。**既定は親 run の simulation.parallel**。"
             "GeNN のコード生成とコンパイルが run ごとに走るので控えめに")
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
    """`task.yaml` から損傷プロトコルを読み、`record_hours` を展開する。

    展開をここでやるのは、lesion の task が `ConfigManager.resolve()` を通らないため
    (通る develop / akita_soc は resolve() の中で展開される)。**run の config.yaml には
    展開後の時刻が並ぶ**、という約束は 3 実験とも同じ。
    """
    profiles = load_yaml(TASK_PATH)
    if name not in profiles:
        raise SystemExit(
            f"損傷プロトコル {name!r} が {TASK_PATH} にありません "
            f"(あるもの: {sorted(profiles)})")
    protocol = dict(profiles[name])
    protocol["profile_name"] = name
    if "record_hours" not in protocol:
        raise SystemExit(f"損傷プロトコル {name!r} に record_hours がありません: {TASK_PATH}")
    protocol["record_hours"] = expand_hours_spec(protocol["record_hours"])
    return protocol


def resolve_parallel(configs: list, requested: int | None) -> int:
    """同時に走らせる run の数。**既定は親 run の `simulation.parallel`。**

    lesion は自分のネットワーク設定を持たないので、並列度も親から引き継ぐ。
    `--parallel` で上書きした値も `prepare_runs()` が run の `config.yaml` に書き込むので、
    記録と実行が食い違わない。

    親が複数あって値が割れているときは**選ばずに止める**。多数決も最大値も、
    どちらを採ったか記録に残らないため。
    """
    if requested is not None:
        return max(1, int(requested))
    values = {int(getattr(config.simulation, "parallel", 1) or 1) for config in configs}
    if len(values) > 1:
        raise SystemExit(
            f"親 run の simulation.parallel が揃っていません ({sorted(values)})。"
            " --parallel で明示してください。")
    return max(1, values.pop())


def prepare_runs(manager: ConfigManager, parents: list[Path], configs: list, protocol: dict,
                 run_name: str, parallel: int) -> list[Path]:
    """**親 run の下に** `<run_name>/` を作り、`config.yaml` を置く。

    `run_name` は切断条件 (`cut_label()`)。`--dry-run` ならそれに `DRY_RUN_SUFFIX` を付けたもの。

    `config.yaml` の `simulation.parallel` は、実際に使う並列度 `parallel` で上書きする。
    """
    run_dirs = []
    for parent, config in zip(parents, configs):
        run_dir = parent / run_name
        if run_dir.exists():
            raise SystemExit(
                f"{run_dir} は既にあります。\n"
                "  上書きすると前の結果と混ざるので止めました。"
                " 同じ親を同じ切断条件で切り直すなら、先に手で消してください。")
        run_dir.mkdir(parents=True)

        # **task を丸ごと差し替える。** 親の記録プロトコル (develop の record_hours など)
        # はこの run では使わないので残さない。親のパスをここで焼き込むことで、
        # run ディレクトリの config.yaml だけで何が起きたかが読めるようになる。
        merged = dict(protocol)
        merged["parent_run"] = str(parent.resolve())
        config.task = type(config.task).model_validate(merged)
        config.simulation.parallel = int(parallel)
        manager.save_config(config, save_dir=run_dir)
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
    # **ネットワークを組む前に決まる。** 切断 spec だけから決まるので、ランチャが
    # run ディレクトリ名に使える (`run_one` 側で付く label と同じ値)。
    label = cut_label(protocol.get("cut") or [], str(protocol.get("cut_combine", "or")))
    run_name = label + DRY_RUN_SUFFIX if args.dry_run else label
    manager = ConfigManager()
    # **親の config がこの run のネットワーク設定。** 並列度もここから取るので、
    # run ディレクトリを作る前に全部読んでおく。
    configs = [manager.load_resolved(parent / CONFIG_NAME) for parent in parents]

    # 親の本数を超えて並べても意味がないので頭打ちにする。
    parallel = max(1, min(resolve_parallel(configs, args.parallel), len(parents)))
    backend = configs[0].simulation.backend
    if backend == "cuda" and parallel > 1:
        print(
            f"Warning: backend=cuda で parallel={parallel} です。1 枚の GPU に複数プロセスを"
            " 載せてもドライバが時分割するだけで速くなりません (小さい N では起動レイテンシ"
            " 律速なのでむしろ遅くなります)。--parallel 1 にしてください。",
            file=sys.stderr,
        )

    print(f"parent    : {args.parent}  ({len(parents)} run)")
    print(f"task      : {args.task}")
    print(f"cut       : {protocol.get('cut')}  -> {label}")
    print(f"parallel  : {parallel}  "
          f"({'--parallel' if args.parallel is not None else '親 run の simulation.parallel'}"
          f", backend={backend})")

    run_dirs = prepare_runs(manager, parents, configs, protocol, run_name, parallel)
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
    print(f"完了しました: {len(run_dirs)} run  (それぞれ <親 run>/{run_name}/)")
    if args.dry_run:
        print(f"  下見の結果は各 <親 run>/{run_name}/{RUN_LOG_NAME} にあります")


if __name__ == "__main__":
    main()
