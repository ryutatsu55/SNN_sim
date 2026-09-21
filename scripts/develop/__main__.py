"""develop 実験のランチャ (親プロセス)。

    python -m scripts.develop --config <config ファイル名> --condition <条件名>

**引数はこの 2 つだけ。「どの config か」と「どの条件として記録するか」しか受けない。**
それ以外 (seed・並列度・記録条件・トレース) はすべて config 側にある —— 引数で渡せる
ものがあると、run ディレクトリの `config.yaml` を見ても何が起きたか分からなくなる。

やること:

1. config を解決する (task プロファイルはこのディレクトリの `task.yaml` から)
2. `simulation.seed` の指定を展開する (`5` → 1 本 / `[1, 10]` → 10 本)
3. `outputs/develop/<条件>/` を作る。**既にあれば拒否して止まる**
4. seed ごとの run ディレクトリに、seed を実値にした `config.yaml` を置く
5. `run_one` を子プロセスとして並列に起動し、各々の出力を `<run>/run.log` へ流す

出力の形:

    outputs/develop/<条件>/            seed が 1 つ  -> ここが run ディレクトリそのもの
    outputs/develop/<条件>/seed01/     seed が複数   -> ここが run ディレクトリ
                        /seed02/

ディレクトリ名からは何も読み取らない。seed も条件も `config.yaml` の中にある。
"""
import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigManager, expand_seed_spec
from src.core.output_manager import SOURCE_CONFIG_NAME

from scripts.develop.store.paths import PENDING_CONFIG_NAME

_HERE = Path(__file__).resolve().parent

# この実験の名前。出力の第 1 階層になる。**config の名前ではない**
# (条件を変えるたびに config と outputs が両方増えるのを止めるため、ここは固定)。
EXPERIMENT = "develop"
OUTPUT_ROOT = Path("outputs")
# メイン config もこの実験の持ち物なので、このディレクトリに置く
# (1 実験 = 1 ディレクトリ)。`configs/` に残っているものはパスで明示すれば使える。
CONFIG_ROOT = _HERE
# 記録プロトコル。**どのプロファイルを使うかはメイン config の `task:` が決める**
# (記録条件は結果を変えるので、選択も config 側にある)。
TASK_PATH = _HERE / "task.yaml"
RUN_LOG_NAME = "run.log"


def parse_args():
    parser = argparse.ArgumentParser(
        description="develop 実験を走らせる (seed ごとに子プロセスを並列起動)。")
    parser.add_argument(
        "--config", required=True,
        help="config ファイル名。名前だけなら scripts/develop/ の中を見る "
             "(例: axon_growth_grid)。パスを含めれば configs/ のものも指定できる")
    parser.add_argument(
        "--condition", required=True,
        help="条件名。outputs/develop/<条件>/ になる。**条件の記述ではなくラベル** "
             "(何で走ったかは run ディレクトリの config.yaml が持つ)")
    return parser.parse_args()


def resolve_config_path(name: str) -> Path:
    """`axon_growth_grid` / `axon_growth_grid.yaml` / `configs/akita_soc.yaml` を受ける。

    名前だけを渡した場合は **`scripts/develop/` の中**を見る。パスを含む形で渡せば
    そのまま使うので、`configs/` に置いたままの config も指定できる。
    """
    candidate = Path(name)
    if candidate.suffix != ".yaml":
        candidate = candidate.with_suffix(".yaml")
    if candidate.parent == Path("."):
        candidate = CONFIG_ROOT / candidate
    if candidate.resolve() == TASK_PATH:
        raise SystemExit(
            f"{TASK_PATH.name} は記録プロトコルであってメイン config ではありません。"
            " 使うプロファイル名はメイン config の `task:` に書いてください。"
        )
    if not candidate.exists():
        raise SystemExit(f"config が見つかりません: {candidate}")
    return candidate


def seed_dir_name(seed: int) -> str:
    """`seed01`。ゼロ埋め 2 桁にするのは辞書順と数値順を一致させるため。

    (旧 run の `seed1` … `seed11` は `seed1, seed10, seed11, seed2` と並んでいた)
    """
    return f"seed{seed:02d}"


def prepare_runs(manager: ConfigManager, config, config_path: Path, condition: str,
                 seeds: list[int]) -> list[Path]:
    """`<条件>/` と各 run ディレクトリを作り、seed を実値にした引き継ぎ config を置く。

    置くのは `config.yaml` ではなく `pending_config.yaml`。**ここで書けるのは記録ではなく
    引き継ぎだから** —— `network.sparse` の実値は build 時にしか決まらないので、この時点の
    config を `config.yaml` の名前で置くと「run の記録」の不変条件を破る。
    `config.yaml` を書くのは build を通した run 本体の仕事 (`run_one.py`)。
    """
    condition_dir = OUTPUT_ROOT / EXPERIMENT / condition
    if condition_dir.exists():
        raise SystemExit(
            f"条件 {condition!r} は既にあります: {condition_dir}\n"
            "  上書きすると前の結果と混ざるので止めました。"
            " 別の条件名を付けるか、不要なら手で消してください。"
        )
    condition_dir.mkdir(parents=True)

    run_dirs = []
    for seed in seeds:
        run_dir = condition_dir if len(seeds) == 1 else condition_dir / seed_dir_name(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        # **run ごとの config.yaml には必ずスカラーの seed を書く。** 範囲指定のまま
        # 残すと「この run の seed はどれか」が決まらなくなる (Hard Rule 7)。
        # 範囲そのものは source_config.yaml (入力の逐語コピー) に残る。
        config.simulation.seed = int(seed)
        manager.dump_config(config, run_dir / PENDING_CONFIG_NAME)
        # 入力 YAML の逐語コピー。seed の範囲指定とコメントが残るのはこちら。
        shutil.copy2(config_path, run_dir / SOURCE_CONFIG_NAME)
        run_dirs.append(run_dir)
    return run_dirs


def launch(run_dir: Path) -> int:
    """run_one を子プロセスとして起動し、出力を <run>/run.log へ流す。

    親がログの行き先を決めるので、シェル側のリダイレクトも EXIT トラップも要らない
    (run ディレクトリ名が起動前に決まっているからできること)。
    """
    command = [sys.executable, "-m", "scripts.develop.run_one", str(run_dir)]
    log_path = run_dir / RUN_LOG_NAME
    with open(log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=str(project_root), stdout=log,
                                   stderr=subprocess.STDOUT)
    return completed.returncode


def main():
    args = parse_args()
    os.chdir(project_root)

    config_path = resolve_config_path(args.config)
    manager = ConfigManager()
    # task プロファイル名は config の `task:` が持つ (ここでは指定しない)。
    config = manager.resolve(str(config_path), task_path=TASK_PATH)

    seeds = expand_seed_spec(config.simulation.seed)
    # seed の本数を超えて並べても意味がないので頭打ちにする。
    parallel = max(1, min(config.simulation.parallel, len(seeds)))

    if config.simulation.backend == "cuda" and parallel > 1:
        print(
            f"Warning: backend=cuda で parallel={parallel} です。1 枚の GPU に複数プロセスを"
            " 載せてもドライバが時分割するだけで速くなりません (小さい N では起動レイテンシ"
            " 律速なのでむしろ遅くなります)。backend: cpu にするか parallel: 1 にしてください。",
            file=sys.stderr,
        )

    print(f"config    : {config_path}")
    print(f"task      : {config.task.profile_name}")
    print(f"condition : {args.condition}")
    print(f"seeds     : {seeds}")
    print(f"parallel  : {parallel}  (backend={config.simulation.backend})")

    run_dirs = prepare_runs(manager, config, config_path, args.condition, seeds)
    for run_dir in run_dirs:
        print(f"  -> {run_dir}")

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        codes = list(pool.map(launch, run_dirs))

    failed = [d for d, code in zip(run_dirs, codes) if code != 0]
    print()
    if failed:
        print(f"失敗した run が {len(failed)} / {len(run_dirs)} 本あります:", file=sys.stderr)
        for run_dir in failed:
            print(f"  {run_dir}  (ログ: {run_dir / RUN_LOG_NAME})", file=sys.stderr)
        print("  再実行: python -m scripts.develop.run_one <run ディレクトリ>", file=sys.stderr)
        raise SystemExit(1)
    print(f"完了しました: {OUTPUT_ROOT / EXPERIMENT / args.condition}")


if __name__ == "__main__":
    main()
