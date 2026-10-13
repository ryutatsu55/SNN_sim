"""run ディレクトリの内部構造。

**`data/` と `report/` というリテラルを知っているのはこのモジュールだけ。** 書く側も
読む側も必ずここを通るので、片方だけ場所がずれることがない。

    <run>/
    ├── config.yaml          … run の**記録** (seed は実値のスカラー)。ランチャが置く
    ├── source_config.yaml   … 入力 YAML の逐語コピー (seed の範囲指定はここに残る)
    ├── run.log              … ランチャが子プロセスの出力を流し込む先
    ├── data/                … **記録** (npz と csv)。**最初からここに書く**
    │                          (座標は coords.npz。`no_space` の run には無い)
    └── report/              … **報告** (人が読むもの: 図と表)。消しても replot で作り直せる
        ├── structure/       … area / connection_mask / network_sample / 各種分布 /
        │                      群間結合確率の表
        ├── panels/          … **記録窓ごと**の図。窓の数だけ増えるものはここへ
        │   ├── raster/
        │   ├── avalanche/
        │   ├── weight/      … 重み行列
        │   ├── weight_distribution/ … 重み分布
        │   ├── weight_network/ … 重みで描いたネットワーク図
        │   ├── correlation/ … ニューロン間相関行列
        │   ├── correlation_network/ … 相関の強いペアだけを空間に描いた図
        │   ├── ISI/         … スパイク間隔の分布と CV
        │   └── trace/       … task.trace_neuron を指定した run のみ
        └── overview/        … figure2c / figure2d / 平均相関の推移

**最初から正しい場所へ書く。** 走り終えてから `data/` へ移すような後処理は無いので、
完走したか否かで run の形が変わらない。

読む側も同じ `data_dir()` を通る。**旧レイアウトの吸収は持たない** —— 記録窓の原点を
持たない古い run はどのみち再解析できない。読みたくなったら再実行すること。
"""
from __future__ import annotations

from pathlib import Path

DATA_SUBDIR = "data"
REPORT_SUBDIR = "report"

# 報告の種類 = `report/` から見た相対パス。ここに無い種類を report_path() に渡すと弾かれる
# (タイポで図や表が迷子になるのを防ぐ)。
#
# **記録窓ごとに出るものは `panels/` の下にまとめる。** run に 1 枚しか出ないもの
# (structure / overview) と同じ階層に並べると、窓の数だけ中身が増えるディレクトリに
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
TRACE = f"{PANELS}/trace"
OVERVIEW = "overview"
REPORT_KINDS = (STRUCTURE, RASTER, AVALANCHE, WEIGHT, WEIGHT_DIST, WEIGHT_NETWORK,
                CORRELATION, CORRELATION_NETWORK, ISI, TRACE, OVERVIEW)

# run を走らせる前に必ず作るディレクトリ。trace は task.trace_neuron を指定したときだけ作る
# (空ディレクトリが残ると「採取したのに空だった」のか「採取していない」のか読めなくなる)。
DEFAULT_REPORT_KINDS = (STRUCTURE, RASTER, AVALANCHE, WEIGHT, WEIGHT_DIST, WEIGHT_NETWORK,
                        CORRELATION, CORRELATION_NETWORK, ISI, OVERVIEW)


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
