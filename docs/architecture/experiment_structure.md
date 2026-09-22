# 実験ディレクトリの構成（設計）

目的は 1 つ。

> **「どのタイミングで、どの関数が呼ばれ、何を出すのか」と「その関数はどこにあるか」が、
> 探さなくても分かる状態にする。**

この形は `scripts/develop/` で確立し、**`scripts/akita_soc/` と `scripts/lesion/` にも
そのまま適用されている** (2026-09-22)。以下は develop を実例に説明するが、§1〜§4 の
決め事・層の責務・不変条件・プロトコルは**3 実験すべてに効く**。実験ごとに違うのは
`store/` の実装（記録ファイル名の規約）と `figures/` の顔ぶれだけで、§2 のツリーと
§5 の出力一覧は develop の具体例として読むこと。

前提となる読み出し契約は `docs/architecture/runview_contract.md`。
run の回し方と出力の規約は `docs/develop_refactor_plan.md`。
なぜこの形に統一したかは `docs/scripts_unification_plan.md`。

---

## 1. 核になる 2 つの決め事

### ① 時間軸で 3 つに切る

出力のタイミングは 3 つしかない。

| いつ | 何が分かっている | view |
|---|---|---|
| build 直後 | ネットワークの形（まだ回していない） | `Built` |
| 記録窓ごと | その窓のスパイク・重み・トレース | `Window` |
| run 終了後 | 全記録窓 | `Series` |

この 3 つに対応する**登録簿**が `report/` に 1 つずつある。
**「何がいつ出るか」はこの 3 ファイルを見れば尽きる。**

### ② 本番も再解析も、必ず一度ファイルに書いてから読み直す

```python
# run_one.py（記録時刻ごと）           # replot.py
save_spikes(...); save_weights(...)
window = series.window(hour)           for window in series.windows:
report.panels.emit(window)                 report.panels.emit(window)
```

**同じ 1 行になる。** `Trace` の手組みも、`total_neurons` / `layout` / `order_axes` /
`smax` の受け渡しも消える。コストは直前に書いた npz を読み直す数 ms だけ。

---

## 2. ディレクトリ

```
scripts/develop/
├── __main__.py  run_one.py  replot.py      いつ report を呼ぶか。それだけ
├── task.yaml    axon_growth_hierarchy.yaml
│
├── report/          「いつ何を出すか」の登録簿
│   ├── __init__.py     guard() の再エクスポートだけ (実体は src/utils/runview.py)
│   ├── structure.py    build 直後
│   ├── panels.py       記録窓ごと
│   └── overview.py     run 終了後
│
├── store/           読み書きだけ。加工しない
│   ├── paths.py        run 内部のディレクトリ規約
│   ├── records.py      1 ファイルの読み書き（名前 ↔ 中身）
│   ├── series.py       Window / Series
│   └── (Built は契約の側 = src/utils/runview.py の BuiltNetwork)
│
├── analysis/        表を作る
│   ├── metrics.py
│   └── connectivity.py
│
└── figures/         図を描く
    ├── __init__.py     save() だけ
    ├── style.py        色 + 並び順（2 枚以上が一致すべきもの）
    ├── area.py  connection_mask.py  network.py  synapse_hist.py
    ├── raster.py  avalanche.py  trace.py
    ├── weight_matrix.py
    └── fig2c.py  fig2d.py
```

**1 ファイル = 1 種類の図。** 「1 枚」ではなく「1 種類」で、同じ描き方から 2〜3 枚出る
ものは 1 ファイルにまとまる (`network.py` の直線版と軸索版、`synapse_hist.py` の
遅延/距離/重み、`weight_matrix.py` の 1 枚版・時系列パネル・差分パネル)。
**片方だけ直すと壊れるものを同じファイルに置く**、というのが基準。

---

## 3. 各層が知ること・知らないこと

| | 知る | **知らない** |
|---|---|---|
| `run_one` / `replot` | GeNN の回し方。`report` の 3 つを呼ぶ順番 | どんな図が出るか。どんな指標があるか |
| `report/*` | この段階で何を出すか。出力先の種類 | 描き方。測り方。ファイル名の規約 |
| `store/*` | ファイル名。npz のキー。ディレクトリ | 何のために読まれるか。matplotlib |
| `analysis/*` | 何を測るか。列名 | どう描かれるか。どこに置かれるか |
| `figures/*` | 見た目のすべて | いつ呼ばれるか。どこに置かれるか |

### 不変条件（import 文を見れば違反が分かる）

以下の `store/` `analysis/` `figures/` `report/` は、**すべてその実験のもの**
(`scripts/<実験>/`) を指す。共有層 (`src/utils/analysis/` = 数式、
`src/utils/runview.py` = 契約) はどの層から import してもよい。

1. `store/` は `scripts/<実験>/analysis/` も図も import しない
2. `analysis/` は matplotlib を import しない
3. **`report/` は matplotlib も numpy も import しない** —— 橋渡ししかしないので、
   計算も描画も入り込めない。各ファイルは先頭の `FIGURES` という表を上から回すだけで、
   表に入らない出力 (CSV・`metrics.csv` への追記) だけが `emit()` に明示行として残る。
   3 実験ぶんまとめて `test/experiments/test_report_registries.py` が固定している
4. **`figures/` は `scripts/<実験>/analysis/` を import しない。** 逆 (`analysis/` が
   `figures/style.py` の `available_order_axes` を借りる) は許す —— 群分けの軸は
   絵と数値が一致していなければ意味が無く、選び方を 2 つ持つ方が危ない。
   **向きを 1 つに保つ**のが要点で、両方向に辺があると 3 件目の置き場所が決まらなくなる。
   `smax` は config の `simulation.N` を各自が読めば済むので、共有しない

   **`src/utils/analysis/` はこの禁止の対象外。** 図が `powerlaw.fit_distribution_curves`
   や `weights.excitatory_flags` を直接呼ぶのは正しい (実際どの実験の `figures/` もそうして
   いる)。禁じているのは**その実験の `analysis/` が組み立てた表**に図が依存することで、
   共有された数式に依存することではない。前者は「同じ実験の中で列名と絵が絡む」ので
   向きが要るが、後者は**分岐したらバグになる数式**なので誰が呼んでもよい。

---

## 4. プロトコル

**リーダー 1 つ + 出力先 1 つ。** それだけ。

```python
figures/network.py        def network(built,  out_path) -> None
figures/raster.py         def raster(window, out_path) -> None
figures/fig2c.py          def fig2c(series, out_path) -> None
analysis/metrics.py       def build_row(window) -> dict          # 表なので dict を返す
analysis/connectivity.py  def write_report(built, out_dir) -> None
```

`total_neurons` / `order_axes` / `smax` / `wmax` は引数から消える。
すべて `view.config` と `view.layout` からその図・その指標が自分で導く。
`order_axes` は `style.available_order_axes(view.layout)` を図が自分で呼ぶ
（スタイルは引数にしない、という既存の決定のまま）。

### 登録簿の形

```python
# report/panels.py
"""記録窓ごとに出すもの。"""
from scripts.develop.report import guard
from scripts.develop.store import paths
from scripts.develop.analysis.metrics import build_row
from scripts.develop.figures.raster import raster
from scripts.develop.figures.avalanche import avalanche_distribution
from scripts.develop.figures.trace import neuron_trace


def emit(window, *, metrics) -> None:
    run, hour = window.run_dir, window.hour
    metrics.append(build_row(window))
    guard("raster", raster, window,
          paths.fig_path(run, paths.RASTER, f"raster_{hour:g}h.png"))
    guard("avalanche", avalanche_distribution, window,
          paths.fig_path(run, paths.AVALANCHE, f"avalanche_{hour:g}h.png"))
    guard("neuron trace", neuron_trace, window,
          paths.fig_path(run, paths.TRACE, f"neuron_trace_{hour:g}h.png"))
```

`metrics` を受け取るのは `panels` だけ。**`metrics.csv` は記録窓をまたいで 1 行ずつ
追記する**ので窓で閉じない（Ctrl-C しても走った分が残る、という性質を保つ）。
これはこの段階の性質そのものなので隠さない。

トレースを採っていない run では `window.trace()` が `MissingData` を投げ、
`guard()` が `skip neuron trace: trace がありません` と出して次へ進む。
**図の側に分岐は書かない。**

---

## 5. 出力の一覧

| 段階 | 図 | 表 |
|---|---|---|
| `structure` | `area` `connection_mask_coarse` `network_sample` `axon_network` `connection_probability` `delay_distribution` `distance_distribution` `weight_distribution` | `connection_probability.csv` `bridge_hops.csv` |
| `panels` | `raster_{h}h` `avalanche_{h}h` `neuron_trace_{h}h` | `metrics.csv` の 1 行 |
| `overview` | `figure2c` `figure2d` `weight_matrix_{h}h` `weight_matrix_panel` `weight_delta_panel` | — |

---

## 6. replot は再ビルドする

構造図は `Built` が要るので、これまで再解析できなかった。**再ビルドして描く。**

調べた条件:

- **構造生成の乱数源は 1 つだけ。** `NetworkBuilder.__init__` の
  `self.rng = np.random.RandomState(config.simulation.seed)`。
  backend が効くのは GeNN のデバイス RNG（= スパイク列）で、**構造には影響しない**
- **再ビルドは安い。** コード生成とコンパイルは `sim.setup()` の `model.build()` 側。
  `builder.build()` は GeNN モデルオブジェクトを組み立てるだけ
- したがって **cuda で走らせた run でも replot は cpu バックエンドで再ビルドしてよい**（CUDA 不要）

### 自己検証

```python
built = Built.from_builder(builder, run_dir)
built.verify_against(series.wiring())     # 一致しなければ落とす
```

**「再ビルドが元の run と同じネットワークか」が replot のたびに検証される。**
再現性を仮定するのではなく毎回確かめる。

比べるのは**結合の集合**。並び順には依存しない形にしてある (この関数が答えるのは
「同じネットワークか」であって「同じ並びか」ではないので)。

実装中、この照合が一度誤検知した。当時は `builder.global_coo()` が行優先、
`data/connectivity.npz` が GeNN の格納順 (集団ごとのブロック連結) で、**同じネットワークに
本数の等しい 2 通りの並びが存在していた**。位置で対応づけると黙って別のシナプスに値が
乗る状態だったので、`simulator.synapse_connectivity_coo()` の側で並びを揃えた
(`docs/technical/compat_breaks.md`)。いまは COO の並びはプロジェクト全体で 1 つ。

旧い run の npz は古い並びのままなので、`store/records.py::load_connectivity()` が
読んだ時点で行優先かを検査して弾く。

コスト: 大きい run（N=40000 級）では `build()` のシナプス登録に時間とメモリがかかる。
先回りせず、実装後に実測して必要なら考える。

---

## 7. 検証

いままでと同じ。**PNG のバイト一致**と `metrics.csv` のバイト一致。

- 基準は `test/experiments/develop/reference_figures.md5`（21 枚）。手順は同ディレクトリの
  README.md。`scripts/akita_soc/`（15 枚）と `scripts/lesion/` にも同じ基準がある
- 重みパネル 2 枚は `run_dir.name` をタイトルに埋めるので、run を `_ref` に改名して replot し、
  全 21 枚で比較する
- **今回は replot が構造図も描くので、`run_one` の構造図と `replot` の構造図が
  バイト一致することが追加の検査になる**（= 再ビルドが同じネットワークを作った証拠）
- 新規 CSV は `run.log` の整形表と突き合わせる
- `pytest test/ -q`（criticality の 5 件は既知の未修正分）

---

## 8. 新しい実験を足すとき

1. `store/` を書く（契約の実装。ファイル名の規約はその実験のもの）
2. 出力を 3 つのタイミングに割り振って `report/` に登録する
3. 図を `(view, out_path)` で書く（共有層から複製してよい）
4. `run_one` を「書いてから読み直す」形にする
5. **その実験にも図の md5 基準を作る**（`test/experiments/<実験>/reference_figures.md5`）。
   develop はこれで全工程を検証できた。基準が無いと「変わっていないこと」を言えない
6. テストは `test/experiments/<実験>/`。ネットワーク設定は
   `test/experiments/no_space_100.yaml` を使い、**実験の config を借りない**
   （借りると実験の条件を変えるたびに別の実験のテストが落ちる）

### 実験か、道具か

**両方**満たすときだけこの 5 層を敷く:

1. 記録窓が複数ある (時刻ごとの図と run 全体の図が両方出る)
2. 本番実行と再解析の 2 経路がある

満たさないもの (`c_elegans` / `spike_animation` / `visualize_network_structure` /
`pipeline_check`) は `scripts/tools/` の 1 ファイル。記録窓も再解析も無いので、層を
敷いても埋まらない。

---

## 9. 未決

1. **`report/` という名前。** 「いつ何を出すか」の登録簿という中身に対して、
   もっと良い名前があるかもしれない（`stages/` / `emit/` / 直下に 3 ファイル平置き）
2. **akita_soc と develop の図をどこまで揃えるか。** いまは複製で、片方だけ直したときに
   気づく仕組みが `reference_figures.md5` しかない
   （`docs/scripts_unification_plan.md`）
