# akita_soc 実験

Ikeda-Akita-Takahashi (2023, APL Bioeng.) Fig.2 の再現。領域も空間も持たない確率結合
ネットワーク (N=100, E/I = 80/20) を 72 時間走らせ、E-STDP と I-STDP による臨界への
自己組織化を追う。

**`scripts/develop/` と同じ構造だが、別の実験として個別に管理する。** develop は同じ
プロトコルを空間構造 (軸索伸長 + モジュール領域) の上で走らせるもので、条件も図も別々に
育てていく。共有するのは数式 (`src/utils/analysis/`) と読み出し契約
(`src/utils/runview.py`) だけ。

---

## 走らせる

```bash
python -m scripts.akita_soc --config akita_soc --condition <条件名>
```

| 引数 | 意味 |
|---|---|
| `--config` | config ファイル名。**名前だけなら `scripts/akita_soc/` の中**を見る |
| `--condition` | 条件名。`outputs/akita_soc/<条件名>/` になる |

引数はこの 2 つだけ。seed も並列度も記録条件もトレースもすべて config 側にある。
同名の条件が既にあると拒否して止まる。

失敗した run だけやり直す:

```bash
python -m scripts.akita_soc.run_one outputs/akita_soc/<条件>/seed03
```

図と指標を作り直す (本番と同じ関数を通り、`metrics.csv` を上書きする):

```bash
python -m scripts.akita_soc.replot <run ディレクトリ> [--no-structure]
```

**引数・出力ディレクトリ・seed の書き方・並列度・`pending_config.yaml` の意味は
`scripts/develop/README.md` と同じ。** そちらに詳しく書いてある。

---

## この実験に固有のこと

### config と記録プロトコル

| ファイル | 中身 |
|---|---|
| `akita_soc.yaml` | メイン config。`area: no_space` / `connection: constant_prob_full` / `weight: constant_zero` |
| `task.yaml` | `akita_soc` (0 / 6 / 72 h の 3 点) と `akita_soc_stable_probe` (8 点)。記録時刻は数値なら 1 点、`0..12` のように書けば両端を含む範囲 |

初期重みが全 0 (`weight: constant_zero`) なのがこの実験の要点で、結合は最初から全部
張られていて重みだけが 0 から育つ。「繋がっていない」と「重み 0」が別物であることを
COO が保っているので、統計に 0 が混ざらない。

### 構造図が develop より少ない

`area: no_space` なので、空間を前提にした図 —— エリア・ネットワーク配置・軸索・距離依存の
結合確率 —— を**持たない**。`report/structure.py` が出すのは初期重み行列と重み・遅延の
分布、それに群間結合確率 (E/I 軸) の表だけ。

持っていない図を並べて毎回 skip させるより出す側に置かない方が「何が出るか」が読める、
という判断。空間構造を見たいときは `scripts/develop/` を走らせる。

### 並べ替え軸は E/I だけ

モジュール構造を持たないので `figures/style.py` の `ORDER_AXES` は `("polarity",)`。
develop は `("module", "polarity")`。

---

## ファイル構成

```
scripts/akita_soc/
├── __main__.py     入口: ランチャ (親)。seed を展開し run ディレクトリを作って子を並列起動
├── run_one.py      入口: 1 run を走らせる (子)。引数は run ディレクトリ 1 つ
├── replot.py       入口: 再解析。指標と図を全部作り直す
├── task.yaml       記録プロトコル
├── akita_soc.yaml  メイン config
│
├── report/         いつ何を出すか  ← ここを見れば出力が一覧できる
│   ├── structure.py   build 直後: 初期重み行列 / 重み・遅延の分布 / 結合確率の表
│   ├── panels.py      記録窓ごと: ラスター / アバランチ / トレース / metrics 1 行
│   └── overview.py    run 終了後: fig2c / fig2d / 重み行列 (1 枚・パネル・差分)
├── store/          どこに何があり、どう読み書きするか
├── analysis/       何を測るか (表)
└── figures/        どう描くか (図)
```

層の役割・編集する場所の決め方・「本番と再解析が同じ経路を通る仕組み」は
`scripts/develop/README.md` と同じ。

---

## 旧 `scripts/akita_soc_fig2.py` からの変更

| 旧 | 新 |
|---|---|
| CLI 引数 11 個 (`--record-hours` / `--seed` / `--out-dir` …) | `--config` と `--condition` の 2 つ。残りは config へ |
| 出力が `outputs/<日時>_seedN/` 直下に平置き | `outputs/akita_soc/<条件>/` の `data/` と `figures/<種類>/` |
| 走り終えてから `organize_output()` で移す | 最初から `data/` へ書く |
| 再解析が `metrics_replot.csv` に逃げる (列が本番より少ない) | 本番と同じ関数・同じ列なので `metrics.csv` を上書き |
| `configs/components/tasks.yaml` の `akita_soc_fig2*` | `scripts/akita_soc/task.yaml` |
| `src/utils/experiments/akita_soc/` (runio / fig2c / fig2d / weight_track) | `store/records.py` と `figures/` |

**旧レイアウトの run は読めない。** `data/` に npz が無い run、密形式で重みを保存した run、
記録窓の原点 (`record_start_ms`) を持たない run はエラーで止まる。読みたければ再実行する。

テストは `test/experiments/akita_soc/`。
