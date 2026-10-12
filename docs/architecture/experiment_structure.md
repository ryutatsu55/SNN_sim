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
run の回し方と出力の規約は各実験の README (`scripts/develop/README.md` ほか)。

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

run を**またいで**比べる段階を持つ実験もある (いまは lesion の `report/compare.py` だけ)。
3 つの外側にある 4 つ目で、view は run の束 (`Collection`)。形は run 終了後と同じで、
`analysis/` が束から表を書き、図はその表を読む。束の表の置き場所は run の中ではなく
束ねた単位のディレクトリ (lesion なら親の条件ディレクトリ)。共有の契約
(`src/utils/runview.py`) にはまだ無い —— run の束を使う実験が 2 つ目に出たときに上げる。
要約を各 run の終わりに書く (run ごとの `summary.json`) 形は採らなかった。定義を変えるたびに
全 run の replot が要るうえ、既存の run には要約が無い。束ねる側で `metrics.csv` と
`lesion.json` から計算すれば 1 コマンドで済み、定義も 1 か所に収まる。

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
│   ├── formats.py      1 ファイルの読み書き（名前 ↔ 中身）
│   ├── built.py        build 直後に何を書くかの一覧 (write_built)
│   └── views.py        Built / Window / Series (formats と契約をつなぐ)。open_run()
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
遅延/距離/重み)。
**片方だけ直すと壊れるものを同じファイルに置く**、というのが基準。

### 入口が 3 つある理由

`__main__.py` (親) は seed を展開して run ディレクトリを作り、`run_one.py` (子) を
**`subprocess` で**並列起動する。スレッドや `multiprocessing` ではなく別プロセスなのは
3 つとも同じ理由:

1. **プロセス完全分離。** GeNN / CUDA は fork されたプロセスで初期化すると壊れる
2. **1 seed の異常終了が他へ波及しない**
3. **部分再実行がそのまま手に入る。** 子を単独で叩けば失敗した run だけやり直せる
   (`python -m scripts.develop.run_one <run ディレクトリ>`)

親が run ディレクトリ名を起動前に決められるので、各子のログを `<run>/run.log` へ
置ける (ログの置き場所問題が根ごと消える)。

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

膜電位を採っていない run では `window.voltage()` が `MissingData` を投げ、
`guard()` が `skip neuron trace: voltage がありません` と出して次へ進む。
**図の側に分岐は書かない。**

---

## 5. 出力の一覧

| 段階 | 図 | 表 |
|---|---|---|
| `structure` | `area` `connection_mask_coarse` `network_sample` `axon_network` `connection_probability` `delay_distribution` `distance_distribution` `weight_distribution` | `connection_probability.csv` `bridge_hops.csv` |
| `panels` | `raster_{h}h` `avalanche_{h}h` `weight_matrix_{h}h` `neuron_trace_{h}h` | `metrics.csv` の 1 行 |
| `overview` | `figure2c` `figure2d` | — |

---

## 6. 構造図も記録から描く (replot は再ビルドしない)

**build の直後に記録を書き、それを読み直して構造図を描く。** 窓ごとの図 (`panels`) と
同じ「書いてから読み直す」形で、本番と再解析が同じ経路を通る:

```python
write_built(run_dir, builder)                         # store/built.py
series = open_run(run_dir, require_windows=False)
structure.emit(series.built())                        # run_one も replot も同じ 1 行
```

書くもの (`store/built.py` の一覧): 外部軸・座標・軸索幾何・結合・遅延・build 直後の
重み。自前のスキーマを持つオブジェクト (`NetworkLayout` / `AxonGeometry`) は自分で直列化し、
ただの配列は `formats.py` が直列化する。`area` は記録せず config から組み直す (`BaseArea` は
乱数を使わないので config の純関数。中身は SDF の合成 = コードなので npz に落とす意味が無い)。

### 以前は replot のたびに再ビルドしていた

構造図が「構築されたネットワークそのもの」を見る図で、記録が結合マスクしか持っていなかった
ため。そのついでに `BuiltNetwork.verify_against()` で「再ビルドが記録と同じ結合を作ったか」を
毎回確かめていた。やめた理由は 2 つ:

- **構造図だけが「書いてから読み直す」の例外だった。** 本番はメモリ上の builder から、
  再解析は再ビルドした builder から描くので、描画経路が 2 本あった
- **その検査は run ではなくコードの性質を見ている。** 「seed から同じネットワークが出るか」が
  崩れるのは、リファクタで乱数の消費順が変わったときだけ。それを run を replot したときに
  初めて知るのでは遅い。**golden テスト (`test/test_network_golden.py`) がコミット前に捕まえる**。
  意図した変更で落ちたら期待値を更新し、`docs/technical/compat_breaks.md` に追記する

ついでに速くなった: develop の `axon_growth_fine` (N=256, 記録窓 13) で replot 1 本が
約 5 分 → 図を描く時間だけになった (その 5 分はほぼ全部が軸索伸長の作り直しだった)。

再ビルドは `replot --verify` として残してある (古い run がいまのコードでも同じネットワークを
作るか確かめたいとき用。cuda の run も cpu で再ビルドしてよい —— 構造生成の乱数源は
`NetworkBuilder.rng` の 1 本だけで、backend が効くのは GeNN のデバイス RNG = スパイク列)。
比べるのは**結合の集合**で、並び順には依存しない。

**seed からの再現に本番が依存している場所は lesion だけ** (親 run を組み直して重みを戻す)。
そこには `verify_axon_geometry()` という、結合集合の一致より強い実行時の検査がある。

### 並びが 2 通りあった件

再ビルド照合は一度誤検知した。当時は `builder.global_coo()` が行優先、
`data/connectivity.npz` が GeNN の格納順 (集団ごとのブロック連結) で、**同じネットワークに
本数の等しい 2 通りの並びが存在していた**。位置で対応づけると黙って別のシナプスに値が
乗る状態だったので並びを揃えた (`docs/technical/compat_breaks.md`)。いまは COO の並びは
プロジェクト全体で 1 つで、`connectivity.npz` は builder の `global_coo()` から直接書く。

旧い run の npz は古い並びのままなので、`store/formats.py::load_connectivity()` が
読んだ時点で行優先かを検査して弾く。

---

## 7. 検証

いままでと同じ。**PNG のバイト一致**と `metrics.csv` のバイト一致。

- 手順は `test/experiments/develop/README.md`（本番と `replot` の自己一致チェック。**図の md5 基準は廃止した** —— 図の定義を直すたびに基準も作り直すことになり、「意図して変えた」と「壊れた」を区別できなかったため）。続きは同ディレクトリの
  README.md。`scripts/akita_soc/`（15 枚）と `scripts/lesion/` にも同じ基準がある
- 重みパネル 2 枚は `run_dir.name` をタイトルに埋めるので、run を `_ref` に改名して replot し、
  全 21 枚で比較する
- `run_one` の構造図と `replot` の構造図もバイト一致する（どちらも build の記録から描く）。
  `replot --verify` を付ければ、再ビルドが同じネットワークを作ることも併せて確かめられる
- 新規 CSV は `run.log` の整形表と突き合わせる
- `pytest test/ -q`（criticality の 5 件は既知の未修正分）

---

## 8. 新しい実験を足すとき

1. `store/` を書く（契約の実装。ファイル名の規約はその実験のもの）。build 直後の記録は
   `store/built.py`、読み口は `store/views.py` の `Built` / `Window` / `Series`
2. 出力を 3 つのタイミングに割り振って `report/` に登録する
3. 図を `(view, out_path)` で書く（共有層から複製してよい）
4. `run_one` を「書いてから読み直す」形にする
5. **その実験にも「本番と再解析が一致する」手順を書く**（`test/experiments/<実験>/README.md`）。
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
   気づく仕組みが `test/experiments/<実験>/README.md` の手動チェックしかない
3. **seed 横断の `summary/` に何を置くか。** 時刻ごとの mean±SD だけでは足りない ——
   **遷移時刻が seed でばらつく量は、時刻ごとに平均すると遷移がなまって消える。**
   seed ごとのスカラー要約 (遷移時刻など) の分布も要る
