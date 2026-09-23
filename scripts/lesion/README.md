# lesion 実験

`scripts/develop/` が育てたネットワークを引き継ぎ、**特定の結合を構造的に除去**して、
切断直前から回復までを等間隔で記録する。

**`scripts/develop/` と同じ 5 層構造** (`report/` `store/` `analysis/` `figures/`)。
違うのは「何を記録し、何を測り、どう描くか」だけで、層の役割も解析・描画関数の引数も同じ。

---

## 走らせる

```bash
python -m scripts.lesion --parent outputs/develop/<条件> --condition <条件名>
```

| 引数 | 意味 |
|---|---|
| `--parent` | 引き継ぐ親 run。**run ディレクトリそのものでも、`seedNN/` を並べた条件ディレクトリでもよい** |
| `--condition` | 条件名。`outputs/lesion/<条件名>/` になる |
| `--task` | `task.yaml` の損傷プロトコル名 (既定 `lesion`) |
| `--parallel` | 同時に走らせる run の数 (既定 1)。**結果に影響しない**ので引数でよい |
| `--dry-run` | GeNN を触らず、切断対象を数えて終了 |

### なぜ develop と違って `--config` を取らないのか

**ネットワーク設定が親 run から来る**ため。同じ seed で同じネットワークを再ビルドしないと
重みを復元できないので、config を別に指定する余地が無い。ランチャは親の `config.yaml` を
読み、`task` を損傷プロトコルへ差し替え、`task.parent_run` に親のパスを焼き込んでから
run ディレクトリへ置く。

**おかげで run の `config.yaml` だけで「どのネットワークを、どこで切って、どう観察したか」が
全部読める。** ディレクトリ名からは何も読み取らない。

失敗した run だけやり直す / 図と指標を作り直す:

```bash
python -m scripts.lesion.run_one outputs/lesion/<条件>/seed03
python -m scripts.lesion.replot  outputs/lesion/<条件>/seed03 [--no-structure]
```

### 条件ディレクトリを丸ごと作り直す

`replot.py` が取るのは**run ディレクトリ 1 つ** (`config.yaml` と `data/` を持つ階層) です。
sweep の条件ディレクトリはその下に `seed01`…`seedNN` を束ねたものなので、シェルの
ループで回します。**暫定の手順で、専用の入口はまだありません。**

```bash
for d in outputs/lesion/<条件>/seed*; do
    python -m scripts.lesion.replot "$d"
done
```

1 run あたりの時間は再ビルドが支配します。lesion は親と同じネットワークを組んでから
切り直すので、親の `connection` プロファイルの重さがそのまま効きます。

---

## 損傷プロトコル (`task.yaml`)

記録プロトコルと損傷条件の**両方**がここに入る。develop の `task.yaml` が「いつ記録するか」
だけなのに対し、損傷実験は「どこを切るか」「いつ切るか」も結果を決めるため。

```yaml
lesion:
  from_hour: null              # 親のどの記録時刻の重みを引き継ぐか。null なら最後
  cut:                         # 切断 spec。複数書ける
    - "bridge:kind=inter_cluster"
  cut_combine: or              # "or" | "and"
  settle_ms: 60000.0           # 復元直後・切断直後に落ち着かせる時間
  baseline_window_ms: null     # null なら record_window_ms と同じ
  recovery_hours: 12.0         # 切断後に観察する総時間
  probe_interval_hours: 1.0    # probe の間隔 (等間隔)
  record_window_ms: 600000.0   # 1 probe の記録窓
  record_buffer_ms: 10000.0
  hub_z: 2.5                   # ハブ判定の within-module z 閾値
  include_betweenness: true    # 重い指標。大きな N では切る
  include_clustering: true
```

### 切断 spec

| spec | 意味 |
|---|---|
| `bridge:kind=inter_cluster` | ブリッジを跨いだ結合 (**幾何的**な定義) |
| `between:axis=module` | 群をまたぐ結合 (**位相的**な定義) |
| `hub:metric=participation,top=3,direction=out` | ハブの出力 |
| `synapses:pairs=3-7+12-40` | 名指し |

**`bridge` と `between` は一致しない。** 「ブリッジを横切った軸索が作った結合」と
「module ラベルが違う結合」は別の集合。`lesion.json` の `cross_check` に両方の本数が
入るので、解釈するときに取り違えないこと。

---

## 2 フェーズ / 2 モデル構成

構造的除去を選んだ時点で、切断は再ビルドを意味する (GeNN のシナプス集団は
`set_sparse_connections()` で確定し、実行中に行を削れない)。したがって 1 プロセスで
**2 回 build する**のが唯一の素直な形になる。

```
Phase 1 (intact)                      Phase 2 (lesioned)
─────────────────                     ──────────────────
親 config を同一 seed で再ビルド
  → 保存済み重みを join で復元
  → settle → baseline probe を 1 点
  → w_pre を pull して保存  ──────────→ 同じ config で再ビルド
                                        → w_pre を復元し、切断マスクで行を削除
                                        → 等間隔 probe で回復を追う
```

こうする理由は 2 つ。**ベースラインが「切断直前のまさにその重み」になる** (親 run の
記録時刻の値を借りるのではなく、同じプロセスで測った値になる) こと。そして Phase 2 の
復元重みが Phase 1 の dump と残存シナプス上で一致することを assert できるので、
**復元経路そのものが自己検証になる**こと。代償は GeNN のコンパイルが 2 回走ること。

### 重みの引き当ては位置ではなく (pre, post)

親 run の重み、Phase 1 の dump、Phase 2 の COO —— どれも本数が違いうるので、
`analysis/restore.py` が **(pre, post) の組で join** する。位置で対応づけると、1 本でも
ずれたときに黙って別のシナプスへ値が乗る。引き当てに失敗したら止まる。

---

## 出力

```
outputs/lesion/<条件>/            親が 1 本  -> ここが run ディレクトリそのもの
outputs/lesion/<条件>/seed01/     親が複数   -> ここが run ディレクトリ
```

```
<run>/
├── config.yaml           **run の記録**。親 run・切断 spec・タイムラインが全部入る
├── run.log
├── data/
│   ├── connectivity_pre.npz  Phase 1 (切断前) の結合
│   ├── connectivity.npz      Phase 2 (切断後) の結合
│   ├── layout_axes.npz
│   ├── coords.npz            soma の座標 (`no_space` の run には無い)。切断で動かない
│   ├── lesion.json           **切断で何が起きたかの記録**
│   ├── lesion_cut.npz        切断されたシナプス 1 本ごとの素性
│   ├── cut_profile.csv       属性ごとの「切断群 vs 残存群」
│   ├── metrics.csv           probe ごとの指標 (**絶対値のみ、1 行ずつ追記**)
│   ├── metrics_delta.csv     ベースラインとの差 (run 終了後に一括)
│   ├── connection_probability.csv / bridge_hops.csv   切断**後**の群間結合確率
│   ├── weights_p{NNN}.npz    その probe の重み
│   └── spikes_p{NNN}.npz     スパイク + 窓の原点 / 窓幅 / phase
└── figures/
    ├── structure/        切断**後**のネットワークの形
    ├── panels/           **probe ごと**の図。probe の数だけ増えるものはここへ
    │   ├── raster/
    │   ├── avalanche/
    │   └── weight/       その probe の重み行列
    └── overview/         weight_trajectories / weight_distribution_shift / firing_rate_scatter
```

### ファイル名が develop と違う

`{kind}_p{index:03d}.npz`。理由は 3 つ:

1. **時刻をファイル名に埋めると精度を失う。** probe 間隔が細かいと `1.66667e-05h.npz` になる
2. **切断前の probe は負の時刻を持つ。** 記録軸は「切断からの経過」であって絶対時刻ではない
3. **develop の解析 CLI に誤読されない。** あちらの glob は `spikes_*h.npz`

**時刻の正は npz が持つ** (`record_start_ms` / `record_window_ms` / `phase`)。index は
並べ替えと人間の目印だけに使う。これは develop と同じ約束で、違うのは名前の付け方だけ。

### `hour` の基準は切断時刻

契約の `Window.hour` は「基準時刻からの経過 [h]」で、基準は実験が決める。develop は
run の開始、**lesion は切断の瞬間**。baseline は窓の幅ぶん手前 (負) に置く —— 0.0 にすると
post の最初の probe と重なり、切断の瞬間が図でも CSV でも判別できなくなる。

### 窓幅は probe ごとに読む

baseline と post で窓幅を変えられるので、**全部を同じ幅で割ってはいけない**。窓の違いが
そのまま発火率の段差として現れる。窓幅は各 probe の npz が持つ。

---

## 結果を読むときの注意

**sham (切らない対照) を作っていない。** ベースラインは切断直前の 1 点だけなので、
「回復」と「損傷が無くても進んだ発達の続き」は原理的に分離できない。`lesion.json` の
`parent_drift` に親 run の終盤の動きが入っているので、**回復幅がこのドリフト幅と同オーダー
なら結論を出さないこと。**

---

## ファイル構成

```
scripts/lesion/
├── __main__.py     入口: ランチャ (親)。親 run を探して run ディレクトリを作り子を並列起動
├── run_one.py      入口: 1 run (Phase 1 → 切断 → Phase 2)
├── replot.py       入口: 再解析。再ビルドして切り直し、指標と図を全部作り直す
├── task.yaml       損傷プロトコル
│
├── report/         いつ何を出すか
│   ├── structure.py   切断直後: 切断後のネットワークの形
│   ├── panels.py      probe ごと: ラスター / アバランチ / metrics 1 行
│   └── overview.py    run 終了後: 重み軌跡 / 重み分布の変化 / 発火レート / 差の表
├── store/          どこに何があり、どう読み書きするか
│   ├── paths.py       run 内部のディレクトリ規約
│   ├── records.py     記録ファイル 1 つの読み書き
│   ├── series.py      Window (probe 1 つ) / Series (run 全体)
│   └── parent.py      **親 run (develop 形式) を読む唯一の場所**
├── analysis/       何を測るか (表)
│   ├── metrics.py     probe 1 つの指標 + 切断したものの素性
│   ├── selectors.py   どこを切るか (spec のパースと選択)
│   ├── restore.py     重みの引き当て ((pre, post) の join)
│   └── connectivity.py 群間結合確率
└── figures/        どう描くか (図)
```

### `store/parent.py` —— 実験どうしの横 import を作らないための場所

lesion は develop の run を読むが、**`scripts/develop/` を import しない**。代わりに
「develop 形式の記録を読む実装」をこちらが持つ。読むのは 6 つだけ
(`config.yaml` / `connectivity.npz` / `weights_{h}h.npz` / `layout_axes.npz` /
`axon_geometry.npz` / `metrics.csv`) で、これは develop の内部事情ではなく
**develop と lesion の契約**。ファイル名も `store/parent.py` がローカルに宣言する。

規約が変わったら直す場所は 1 つで、そのことがモジュールの docstring に書いてある。
**この一覧が、develop 側で記録の規約を変える人にとっての索引になる。**

---

## 旧 `scripts/lesion.py` からの変更

| 旧 | 新 |
|---|---|
| CLI 引数 19 個 | `--parent` / `--condition` / `--task` / `--parallel` / `--dry-run` |
| 実験条件が `run_lesion.sh` の設定ブロックにあり `config.yaml` に残らない | 全部 `config.yaml` の `task.*` に焼き込む |
| 出力先が `outputs/lesion/<日時>_seedN_<label>/` = **ディレクトリ名から条件を読む形** | `outputs/lesion/<条件>/`。名前からは何も読まない |
| 走り終えてから `organize_output()` で移す | 最初から `data/` へ書く |
| 再解析の入口が無い | `replot.py` (本番と同じ関数を通り、図も `metrics.csv` もバイト一致) |
| 時刻は `probes.csv` を index で引く | 窓の npz が原点・窓幅・phase を持つ |
| 図が `src/utils/experiments/lesion/figures.py` | `figures/` (1 ファイル 1 種類) |

テストは `test/experiments/lesion/`。
