"""run ディレクトリの内部構造。

**`data/` と `report/` というリテラルを知っているのはこのモジュールだけ。** 書く側も
読む側も必ずここを通るので、片方だけ場所がずれることがない。

    <run>/
    ├── config.yaml          … run の**記録** (親 run・切断 spec・タイムラインが全部入る)。ランチャが置く
    ├── run.log              … ランチャが子プロセスの出力を流し込む先
    ├── data/                … **記録** (npz と csv と lesion.json。座標は coords.npz)
    └── report/              … **報告** (人が読むもの: 図と表)。消しても replot で作り直せる
        ├── structure/       … 切断**後**のネットワークの形と、群間結合確率・切断したものの素性の表
        ├── panels/          … **probe ごと**の図。probe の数だけ増えるものはここへ
        │   ├── raster/
        │   ├── avalanche/
        │   ├── weight/      … その probe の重み行列
        │   ├── weight_distribution/ … 重み分布
        │   ├── weight_network/ … 重みで描いたネットワーク図
        │   ├── correlation/ … ニューロン間相関行列
        │   ├── correlation_network/ … 相関の強いペアだけを空間に描いた図
        │   ├── ISI/         … スパイク間隔の分布と CV
        │   └── trace/       … task.trace_neuron を指定した run のみ
        └── overview/        … 重み軌跡 / 重み分布の変化 / 発火レート散布 / 平均相関の推移 /
                               群ごとの同期 (χ・参加率) / RTE

`scripts/develop/store/paths.py` と同じ形だが、**別のファイルとして持つ**。報告の種類が
実験ごとに変わりうるので、共有すると片方の都合でもう片方が動く (いまは同じ一覧)。

`source_config.yaml` は置かない。lesion の入力は「親 run + 損傷プロトコル」であって
1 つの YAML ではないので、逐語コピーする対象が無い。代わりに **`config.yaml` に全部
焼き込む** —— 親 run のパスも切断 spec も probe のタイムラインも `task.*` に入る。
"""
from __future__ import annotations

from pathlib import Path

DATA_SUBDIR = "data"
REPORT_SUBDIR = "report"

# 報告の種類 = `report/` から見た相対パス。ここに無い種類を report_path() に渡すと弾かれる
# (タイポで図や表が迷子になるのを防ぐ)。
#
# **probe ごとに出るものは `panels/` の下にまとめる。** run に 1 枚しか出ないもの
# (structure / overview) と同じ階層に並べると、probe の数だけ中身が増えるディレクトリに
# 埋もれて「run 全体の図」がどれか読めなくなる。
STRUCTURE = "structure"
PANELS = "panels"
RASTER = f"{PANELS}/raster"
AVALANCHE = f"{PANELS}/avalanche"
WEIGHT = f"{PANELS}/weight"
WEIGHT_DIST = f"{PANELS}/weight_distribution"
WEIGHT_NETWORK = f"{PANELS}/weight_network"
CORRELATION = f"{PANELS}/correlation"   # ニューロン間相関行列
CORRELATION_NETWORK = f"{PANELS}/correlation_network"   # 相関の強いペアの空間図
ISI = f"{PANELS}/ISI"                    # スパイク間隔の分布
CORRELATION_WEIGHT = f"{PANELS}/correlation_weight"   # 結合ごとの相関と重みの散布図
TRACE = f"{PANELS}/trace"
OVERVIEW = "overview"
REPORT_KINDS = (STRUCTURE, RASTER, AVALANCHE, WEIGHT, WEIGHT_DIST, WEIGHT_NETWORK,
                CORRELATION, CORRELATION_NETWORK, ISI, CORRELATION_WEIGHT, TRACE, OVERVIEW)

# run を走らせる前に必ず作るディレクトリ。trace は task.trace_neuron を指定したときだけ作る
# (空ディレクトリが残ると「採取したのに空だった」のか「採取していない」のか読めなくなる)。
DEFAULT_REPORT_KINDS = (STRUCTURE, RASTER, AVALANCHE, WEIGHT, WEIGHT_DIST, WEIGHT_NETWORK,
                        CORRELATION, CORRELATION_NETWORK, ISI, CORRELATION_WEIGHT, OVERVIEW)


def data_dir(run_dir: str | Path) -> Path:
    """npz / csv の置き場所。**書く側も読む側もここを通る。**"""
    return Path(run_dir) / DATA_SUBDIR


def data_path(run_dir: str | Path, name: str) -> Path:
    return data_dir(run_dir) / name


def report_dir(run_dir: str | Path, kind: str) -> Path:
    if kind not in REPORT_KINDS:
        raise ValueError(f"未知の報告の種類: {kind!r} (使えるもの: {REPORT_KINDS})")
    return Path(run_dir) / REPORT_SUBDIR / kind


def report_path(run_dir: str | Path, kind: str, name: str) -> Path:
    return report_dir(run_dir, kind) / name


def prepare(run_dir: str | Path, kinds: tuple[str, ...] = DEFAULT_REPORT_KINDS) -> Path:
    """run ディレクトリ内のサブディレクトリを作る。run を走らせる前に 1 回呼ぶ。"""
    run_dir = Path(run_dir)
    data_dir(run_dir).mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        report_dir(run_dir, kind).mkdir(parents=True, exist_ok=True)
    return run_dir


# ------------------------------------------------------------------------------------
# run の束 (`compare`) の出力。**run ディレクトリの中ではなく、親 run を並べた条件
# ディレクトリの直下**に置く (束ねる単位 = 同じ親条件)。
#
#     <条件>/lesion_compare/
#     ├── runs.csv        損傷 run 1 本 = 1 行
#     ├── control.csv     対照 run 1 本 = 1 行
#     └── report/              … **報告** (人が読むもの: 図と表)。消しても replot で作り直せる
# ------------------------------------------------------------------------------------
COMPARE_SUBDIR = "lesion_compare"


def compare_dir(cond_dir: str | Path) -> Path:
    """run の束の表の置き場所。**書く側も読む側もここを通る。**"""
    return Path(cond_dir) / COMPARE_SUBDIR


def compare_report_path(cond_dir: str | Path, name: str) -> Path:
    return compare_dir(cond_dir) / REPORT_SUBDIR / name
