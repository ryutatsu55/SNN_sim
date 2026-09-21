# develop 実験

Akita SoC Fig.2 相当の発達実験。ネットワークを長時間走らせ、指定した時刻に記録窓を取って
スパイク・重み・指標・図を残す。

---

## 走らせる

```bash
python -m scripts.develop --config axon_growth_grid --condition gmax25_baseline
```

**引数はこの 2 つだけ。**

| 引数 | 意味 |
|---|---|
| `--config` | config ファイル名。**名前だけなら `scripts/develop/` の中**を見る。パスを含めれば `configs/` のものも指定できる |
| `--condition` | 条件名。`outputs/develop/<条件名>/` になる |

受けるのは「**どの config か**」と「**条件名**」だけ。seed も並列度も
記録条件もトレースも、すべて config 側にある。

**条件名は「条件の記述」ではなくラベル。** 何で走ったかは run の `config.yaml` が持つので、
名前にすべての条件を詰め込む必要はない。人が後で見分けられればよい。

**同名の条件が既にあると拒否して止まる。** 前の結果と混ざらないようにするため。やり直す
なら別名を付けるか、古い方を手で消す。

---

## 条件の変え方

**config ファイルは複製しない。** `scripts/develop/<名前>.yaml` を直接書き換えて走らせる。
過去にやった実験の条件は run ディレクトリの `config.yaml` に残るので、条件ごとのファイルを
溜める必要がない。

メイン config もこの実験の持ち物なので `scripts/develop/` に置く (1 実験 = 1 ディレクトリ)。
`configs/` に残っているものを使いたいときは `--config configs/akita_soc.yaml` のように
パスで渡す。

> **注意: 複数の run を時間差で起動する最中に config を書き換えないこと。**
> config が読まれるのは起動時の 1 回だけなので、走行中の run は後の編集に影響されない。
> 危険なのは起動の途中で書き換えたときで、後から起動した run だけ条件が変わる。

### seed

`simulation.seed` に**スカラーでも範囲でも**書ける。

```yaml
simulation:
  seed: 5              # seed 5 の run を 1 本
  seed: [1, 10]        # seed 1〜10 の run を 10 本 (両端を含む)
  seed: [1, 10, 2]     # STEP 付き = 1, 3, 5, 7, 9
  parallel: 8          # 同時に走らせる run の数
```

**`[1, 10]` は「1 と 10 の 2 本」ではなく「1 から 10 まで」。**

seed が複数のときは seed ごとに子プロセスを立てて並列に走らせる。各 run の `config.yaml`
には展開後の**スカラーの seed** が入り、範囲指定は `source_config.yaml` (入力の逐語コピー)
に残る。

#### 並列度 (`simulation.parallel`) の決め方

**backend で決まる。**

| backend | N | 推奨 `parallel` | 理由 |
|---|---|---|---|
| `cpu` | 小 (100〜256) | 4〜8 | GeNN の CPU バックエンドは 1 プロセス 1 コア。上限を決めるのはコア数ではなく、seed ごとに走る GeNN のコンパイルが食うメモリ |
| `cuda` | 大 | 1 | 1 run で GPU を埋めるので、並列にする意味がない |
| `cuda` | 小 | 1 | 1 枚の GPU に複数プロセスを載せてもドライバが時分割するだけ。小さい N ではカーネル起動レイテンシ律速なので、むしろ遅くなる |

`backend: cuda` かつ `parallel > 1` のときは警告が出る。seed の本数を超える値を書いても
本数で頭打ちになる。

### 記録条件

記録時刻・窓幅・バッファは `scripts/develop/task.yaml` を書き換える。
**どのプロファイルで走るかはメイン config の `task:` が決める。**

```yaml
# scripts/develop/axon_growth_grid.yaml
task: develop          # task.yaml のプロファイル名
```

基本は `develop` の 1 つを手で書き換えて使う。記録プロトコルとして本質的に別物のとき
(発達の追跡ではなく短時間のスモークなど) だけプロファイルを足し、config 側で選ぶ。

### 膜電位トレース

1 ニューロンの V と Isyn_rec を記録窓の先頭で採るかどうかも `task.yaml` で決める。

```yaml
trace_neuron: null     # null なら採らない (既定)。採るならニューロン ID
trace_window_s: 10.0   # 記録窓の先頭から何秒ぶん
```

採ると記録窓の先頭を 1 ステップずつ進めて毎ステップ `pull()` するので遅くなる。長い run
では効く。そのぶん、同じ窓のラスターと時間軸が揃った V / I が残る。

**引数にしていないのは、採ったかどうかがその run の記録の一部だから。** 引数にすると、
完走した run に `trace_*.npz` が無いときに「採らない設定だった」のか「採ろうとして失敗
した」のかが区別できなくなる。範囲外のニューロン ID は起動直後に弾く (長い run を回し
切ってから `IndexError` で落ちないように)。

---

## 出力

```
outputs/develop/<条件>/            seed が 1 つ  -> ここが run ディレクトリそのもの
outputs/develop/<条件>/seed01/     seed が複数   -> ここが run ディレクトリ
                     /seed02/
```

run ディレクトリの中身:

```
<run>/
├── config.yaml           **build を通った run の記録**。seed も sparse も実値
├── source_config.yaml    入力 YAML の逐語コピー (seed の範囲指定はここに残る)
├── run.log               この run の標準出力・標準エラー
├── data/
│   ├── connectivity.npz      結合構造 (row/col/shape)。run に 1 回だけ
│   ├── layout_axes.npz       外部軸 (layer / module …)
│   ├── axon_geometry.npz     軸索の折れ線 (axon_growth 系の config のみ)
│   ├── metrics.csv           記録時刻ごとの指標。**1 行ずつ追記される**
│   ├── connection_probability.csv  群間の結合確率 (build 直後に 1 回)
│   ├── bridge_hops.csv       ブリッジのホップ数別の結合確率 (複合エリアの run のみ)
│   ├── weights_{h}h.npz      重みの値ベクトル (connectivity と index 整合)
│   ├── spikes_{h}h.npz       スパイク (絶対時刻 ms)
│   └── trace_{h}h.npz        膜電位トレース (task.trace_neuron を指定した run のみ)
└── figures/
    ├── structure/        area / connection_mask / network_sample / 各種分布
    ├── raster/
    ├── avalanche/
    ├── trace/            task.trace_neuron を指定した run のみ
    └── overview/         figure2c / figure2d / weight_matrix_*
```

**ディレクトリ名からは何も読み取らないこと。** seed も条件も `config.yaml` の中にある。

`metrics.csv` は記録時刻ごとに 1 行ずつ書かれるので、途中で落ちた run もそこまでの指標が
読める。

### 記録ファイルの中身

| ファイル | 鍵 |
|---|---|
| `connectivity.npz` | `row` `col` `shape` |
| `weights_{h}h.npz` | `data` (値ベクトル。`connectivity` の row/col と index 整合) |
| `spikes_{h}h.npz` | `times` (絶対時刻 ms) `ids` **`record_start_ms`** |
| `trace_{h}h.npz` | `V` `I` `dt` `neuron_id` `window_s` `spike_times` `spike_ids` |

**ファイル名の時刻は目印であって、データではない。** `{h}` は `f"{hour:g}"` = 有効数字
6 桁なので、`record_hours` が非整数だと往復で元の値に戻らない (1/3 h なら約 1.2 ms ずれ、
`burstiness_index` のビン割りが変わる)。記録窓の原点は `spikes_{h}h.npz` の
`record_start_ms` が持ち、**再解析はそちらを読む**。おかげで本番と再解析の `metrics.csv` は
バイト単位で一致する。

鍵の名前を知っているのは `records.py` だけで、書く側と読む側が対 (`save_spikes` /
`load_spikes`) で持つ。

### `config.yaml` と `pending_config.yaml`

走っている最中の run には、もう 1 つ `pending_config.yaml` がある。**ランチャから run 本体
への引き継ぎ**で、run 本体が build を通して `config.yaml` を書いた時点で消える。

分けてあるのは役割が違うから。`network.sparse` の実値 (`on` / `off`) を決めるのは
NetworkBuilder の生成時なので、ランチャが起動前に書ける config は `auto` のまま —— それを
`config.yaml` の名前で置くと「この run はこうして走った」という記録の意味が崩れる。

おかげで **`config.yaml` があれば build を通った run**、という読み方ができる。

| run ディレクトリの状態 | 意味 |
|---|---|
| `pending_config.yaml` だけ | まだ走っていない / build の前に落ちた |
| `config.yaml` だけ | build を通った |

---

## 失敗した run だけやり直す

run ディレクトリを指して子プロセスを直接叩く。config を書き換える必要はない。

```bash
python -m scripts.develop.run_one outputs/develop/gmax25_baseline/seed03
```

ランチャは失敗した run を一覧で出すので、そのパスをそのまま渡せばよい。
`pending_config.yaml` が残っていればそれを、完走した run をもう一度走らせるなら
`config.yaml` を読む。

---

## 図と指標を作り直す

```bash
python -m scripts.develop.replot <run ディレクトリ> [--no-structure]
```

`data/` の npz から指標を計算し直し、図を描き直します。

**作り直すのは全部。** 指標も、構造図も、窓ごとの図 (ラスター / アバランチ / トレース) も、
run 全体の図 (fig2c / fig2d / 重み行列) も、例外なく上書きします。一部だけ新しい状態を
作らないためです。

本番と**同じ関数**を通る (`report.structure` / `report.panels` / `report.overview`) ので、
列も値も図もビット単位で一致します。書き出し先も本番と同じ `metrics.csv` で、**上書き
します** —— 同じ関数が同じ列を作るので別名に逃がす理由がありません (逃がしておく方が
むしろ危険で、図がどちらを読むかを明示しないと例外を出さずに古い値が描かれます)。

**構造図のためにネットワークを再ビルドします。** 構造図は「構築されたネットワークその
もの」を見る図なので npz からは作れません。GeNN のコード生成とコンパイルは `setup()` の
側にあり `replot` は通らないので、再ビルド自体は安いです。ネットワークの生成は
`config.simulation.seed` だけに依存する (backend は GeNN のデバイス RNG = スパイク列に
しか効かない) ので、cuda で走らせた run でも cpu で再ビルドできます。

そして**再ビルドが元の run と同じ結合を作ったことを毎回確かめます。** `connectivity.npz`
の row/col と突き合わせ、食い違えば止まります。再現性を仮定するのではなく、replot の
たびに検証している形です。大きい run で再ビルドの時間を惜しむときは `--no-structure`。

**旧レイアウトの run は受け付けません。** `data/` に npz が無い run、`connectivity.npz` を
持たない run (密形式)、記録窓の原点を持たない run はエラーで止まります。列や図を減らして
続行はしません —— 欠けた `metrics.csv` で上書きすると、あとで図と突き合わせたときに
原因が追えなくなります。読みたければ再実行してください。

---

## ファイル構成

```
scripts/develop/
├── __main__.py     入口: ランチャ (親)。seed を展開し run ディレクトリを作って子を並列起動
├── run_one.py      入口: 1 run を走らせる (子)。引数は run ディレクトリ 1 つ
├── replot.py       入口: 再解析。指標と図を全部作り直す
├── task.yaml       記録プロトコル
├── <名前>.yaml     メイン config
│
├── report/         いつ何を出すか  ← ここを見れば出力が一覧できる
├── store/          どこに何があり、どう読み書きするか
├── analysis/       何を測るか (表)
└── figures/        どう描くか (図)
```

**編集する場所は 1 問で決まります。**

> いつ出すかの話か / 測り方の話か / 見た目の話か / 場所の話か
> → `report/` `analysis/` `figures/` `store/`

| したいこと | 開くファイル |
|---|---|
| 新しい指標を `metrics.csv` に足す | `analysis/metrics.py` |
| 指標の計算方法を変える | 同上。数式そのものは `src/utils/analysis/` |
| 図の見た目を細かく変える | `figures/<その図>.py`。2 枚以上が一致すべき値なら `figures/style.py` |
| 新しい図を足す | `figures/` に 1 ファイル + `report/<いつ出すか>.py` に 1 行 |
| 図を出すタイミングを変える | `report/` の 3 ファイル間で 1 行移す |
| 出力先のディレクトリを変える | `store/paths.py` |

### report/ — いつ何を出すか

出力のタイミングは 3 つしかありません。**何がいつ出るかはこの 3 ファイルで尽きます。**

| ファイル | いつ | 受け取るもの |
|---|---|---|
| `structure.py` | build 直後、`setup()` の**前** | `Built` |
| `panels.py` | 記録窓ごと | `Window` |
| `overview.py` | run 終了後 | `Series` |

1 出力 = 1 行で、図も表も同じ扱いです。`__init__.py` の `guard()` が 1 つずつ包み、
**「この run はそのデータを持たない」(`MissingData`) と「バグで落ちた」を分けて**報告します。

**このパッケージは matplotlib も numpy も import しません。** 橋渡ししかしないので、
計算も描画も入り込めません。破れていたら、その処理は `analysis/` か `figures/` のものです。

### store/ — run ディレクトリとのやりとり

| ファイル | 役割 |
|---|---|
| `paths.py` | run の内部構造。`data/` と `figures/` と引き継ぎファイル名を知るのはここだけ |
| `records.py` | 記録ファイル **1 つ**の読み書き。ファイル名の規約と **npz の中の鍵の名前**を、書く側と読む側が対で持つ。CSV は**書き出しだけ** (`MetricsWriter` / `write_table`) —— `metrics.csv` を読むのは `series.py` の `pd.read_csv` 1 か所、結合確率の 2 本は人間向けで誰も読み返さない |
| `series.py` | `Window` (記録窓 1 つ) と `Series` (run 全体)。`open_run()` が入口 |
| `built.py` | `Built` (build 直後のネットワーク)。構造図が `NetworkBuilder` を知らずに済むようにする |

**読み出しだけで、加工はしません。** 「run 全体を通した計算」(重み軌跡・発火レートの
時系列) は、それを使う図のファイルにあります。

### analysis/ — 何を測るか (表)

| ファイル | 役割 |
|---|---|
| `metrics.py` | **1 記録窓**の指標を 1 行の dict にする → `metrics.csv` |
| `connectivity.py` | 構築されたネットワークの群間結合確率 → `connection_probability.csv` / `bridge_hops.csv` + run.log |

「どう測るか」の数式は `src/utils/analysis/` にある共有層に任せます。**ΔCr やべき乗フィット
は実験ごとに違ってはいけない**ので、ここだけは複製しません (図は分岐が目的、数式は分岐が
バグ)。

### figures/ — どう描くか (図)

**1 ファイル = 1 種類の図。** `raster` / `avalanche` / `trace` / `weight_matrix` /
`synapse_hist` / `connection_mask` / `area` / `network` / `fig2c` / `fig2d`。

「1 種類」であって「1 枚」ではありません。同じ描き方から 2〜3 枚出るものは 1 ファイルに
まとまります —— `network.py` は直線版と軸索版 (2 枚は**同じ乱数を同じ順に消費**するので
同じニューロンが映る)、`synapse_hist.py` は遅延/距離/重みの 3 枚 (骨格が 1 つ)、
`weight_matrix.py` は 1 枚版・時系列パネル・差分パネル。**片方だけ直すと壊れるものを
同じファイルに置く**、というのが分け方の基準です。

| ファイル | 役割 |
|---|---|
| `style.py` | **2 枚以上の図が一致すべきものだけ** —— E/I の色と、並べ替えの軸・順序・境界線 |
| `__init__.py` | `save()`。png への書き出し (親ディレクトリを作り、保存し、figure を閉じる) |

**「いつ描くか」「どこに置くか」は知りません。** それを決めるのは `report/` です。
無いデータを要求したときは `MissingData` をそのまま上へ投げます —— 図の側に「持って
いなければスキップ」の分岐は書きません。

**スタイルは引数にしません。** 引数にすると、その関数が用意した枠の中でしか見た目を
変えられないうえ、変更のたびに呼び出し側も直すことになります。関数が受け取るのは
**リーダー 1 つと出力先 1 つ**だけで、見た目 (軸範囲・色・点の大きさ・dpi・figsize) は
モジュール先頭の名前付き定数です。2 通りの見た目で使う図は、引数で切り替えず**関数を
2 つに割ります** (`weight_panel` と `weight_delta_panel`)。

---

## 解析・描画関数に渡すもの

全部この形です。**リーダー 1 つ + 出力先 1 つ。**

```python
figures/network.py        def network(built,  out_path) -> None
figures/raster.py         def raster(window, out_path) -> None
figures/fig2c.py          def fig2c(series, out_path) -> None
analysis/metrics.py       def build_row(window) -> dict          # 表なので dict を返す
analysis/connectivity.py  def write_report(built, out_dir) -> None
```

`total_neurons` も `order_axes` も `smax` も引数にありません。全部リーダーの
`config` と `layout` から、その図・その指標が自分で導きます。

リーダーの形は全実験共通の契約で、`scripts/tools/runview.py` にあります
(`docs/runview_contract.md`)。この実験の `store/` はその実装です。

---

## 本番と再解析が同じ経路を通る仕組み

`run_one.py` は記録を書いたあと、**書いたものを読み直してから**図を描きます。

```python
save_spikes(...); save_weight_values(...); save_trace(...)
panels.emit(series.window(hour), metrics=metrics_csv)   # ← replot.py と同一の行
```

in-memory の値をそのまま渡すと本番と再解析で描画経路が 2 本になり、片方だけ直したときに
静かにずれます (以前は `Trace` を「再解析が npz から組み立てるのと同じ形」に手で組み直して
いました)。直前に書いた npz を読み直すコストは数 ms で、ついでに**書いた npz が読めること**
をその場で確かめたことになります。

---

## 依存の向き

**`scripts/develop/` が `src/` を見るのは 2 つだけです。**

| 見るもの | 何のため |
|---|---|
| `src/core/` | ConfigManager / NetworkBuilder / GeNNSimulator / NetworkLayout。シミュレータ本体 |
| `src/utils/analysis/` | 数式。matplotlib を import しない層 |

このほかに `scripts/tools/runview.py` (読み出し契約) を見ます。実験に依存しない道具なので、
**一方向に** import してよい層です。

`src/utils/plotting/` は**使いません**。相当するものは `figures/` に複製してあります。
共有せず複製してあるのは、どんな図をどんな形式で出すかが実験ごとに違ってくるからで、
1 つの関数を全実験で共有すると、違いを吸収するための引数が際限なく増えるためです。
**この実験の出力はこの実験が全部持つ** —— 線引きに迷う余地を無くすのが目的です。

他の実験 (lesion など) も `scripts/` 直下のスクリプトも import しません。

テストは `test/experiments/develop/test_develop.py`。
