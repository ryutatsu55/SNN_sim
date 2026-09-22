"""run ディレクトリの内部構造。

**`data/` と `figures/` というリテラルを知っているのはこのモジュールだけ。** 書く側も
読む側も必ずここを通るので、片方だけ場所がずれることがない。

    <run>/
    ├── pending_config.yaml  … ランチャ → run 本体の**引き継ぎ**。まだ記録ではない
    ├── config.yaml          … run の**記録** (seed は実値のスカラー、sparse も実値)
    ├── source_config.yaml   … 入力 YAML の逐語コピー (seed の範囲指定はここに残る)
    ├── run.log              … ランチャが子プロセスの出力を流し込む先
    ├── data/                … npz と csv。**最初からここに書く**
    └── figures/
        ├── structure/       … area / connection_mask / network_sample / 各種分布
        ├── raster/
        ├── avalanche/
        ├── trace/           … task.trace_neuron を指定した run のみ
        └── overview/        … figure2c / figure2d / weight_track

**最初から正しい場所へ書く。** `src/core/output_manager.py` の `organize_output()`
(走り終えてから `data/` へ移す) は使わないので、完走したか否かで run の形が変わらない。

読む側も同じ `data_dir()` を通る。**旧レイアウトの吸収は持たない** —— 記録窓の原点を
持たない古い run はどのみち再解析できない。読みたくなったら再実行すること。
"""
from __future__ import annotations

from pathlib import Path

DATA_SUBDIR = "data"
FIGURES_SUBDIR = "figures"

# ランチャが run 本体へ config を引き継ぐためのファイル。
#
# **`config.yaml` と分けてあるのは、役割が違うから。** `config.yaml` は「この run はこうして
# 走った」という記録で、`network.sparse` が実値 ("on"/"off") である必要がある。しかし実値を
# 焼き込むのは NetworkBuilder の生成時なので、ランチャが起動前に書けるのは "auto" のまま。
# それを `config.yaml` の名前で置くと、記録の不変条件が崩れる (`save_config()` はまさに
# それを警告する)。
#
# なので引き継ぎは別名で置き、run 本体が build() の後で `config.yaml` を書いてから
# これを消す。おかげで **`config.yaml` は「build を通った run の記録」以外にはならない**。
# build の前に落ちた run には config.yaml が無い、というのも正しい状態で、
# 「途中で死んだ」ことがディレクトリを見れば分かる。
PENDING_CONFIG_NAME = "pending_config.yaml"

# 図の種類。ここに無い種類を fig_path() に渡すと弾かれる (タイポで図が迷子になるのを防ぐ)。
STRUCTURE = "structure"
RASTER = "raster"
AVALANCHE = "avalanche"
TRACE = "trace"
OVERVIEW = "overview"
FIG_KINDS = (STRUCTURE, RASTER, AVALANCHE, TRACE, OVERVIEW)

# run を走らせる前に必ず作るディレクトリ。trace は task.trace_neuron を指定したときだけ作る
# (空ディレクトリが残ると「採取したのに空だった」のか「採取していない」のか読めなくなる)。
DEFAULT_FIG_KINDS = (STRUCTURE, RASTER, AVALANCHE, OVERVIEW)


def data_dir(run_dir: str | Path) -> Path:
    """npz / csv の置き場所。**書く側も読む側もここを通る。**"""
    return Path(run_dir) / DATA_SUBDIR


def data_path(run_dir: str | Path, name: str) -> Path:
    return data_dir(run_dir) / name


def fig_dir(run_dir: str | Path, kind: str) -> Path:
    if kind not in FIG_KINDS:
        raise ValueError(f"未知の図の種類: {kind!r} (使えるもの: {FIG_KINDS})")
    return Path(run_dir) / FIGURES_SUBDIR / kind


def fig_path(run_dir: str | Path, kind: str, name: str) -> Path:
    return fig_dir(run_dir, kind) / name


def prepare(run_dir: str | Path, fig_kinds: tuple[str, ...] = DEFAULT_FIG_KINDS) -> Path:
    """run ディレクトリ内のサブディレクトリを作る。run を走らせる前に 1 回呼ぶ。"""
    run_dir = Path(run_dir)
    data_dir(run_dir).mkdir(parents=True, exist_ok=True)
    for kind in fig_kinds:
        fig_dir(run_dir, kind).mkdir(parents=True, exist_ok=True)
    return run_dir
