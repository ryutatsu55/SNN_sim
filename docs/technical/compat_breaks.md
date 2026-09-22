# 非互換変更ログ

**過去の run と結果が比較できなくなる変更**をここに記録する。同じシード・同じバックエンドでも
実現値が変わる種類の変更が対象。**旧形式を読むための分岐や後方互換フラグは一切書かない**
（旧 run は「実行時に生成された図と CSV が記録」と割り切る）方針なので、ここが唯一の
追跡手段になる。

> **ルール:** 既存 run と結果が変わる変更をしたら、必ずこのファイルに追記すること。
> 書式は「何が変わったか / 影響の大きさ（実測） / 古い run をどうするか」。

---

## `axon_growth`: 境界処理と結合抽選の変更

`axon_growth` の実現値は以下の変更をまたいで**一切比較できない**。互換フラグは無い。

- 壁での advance-and-slide 方式への変更（旧方式は disk で長さの 28% を切り落としていた）
- `_CONTAINS_TOL = 1e-9` の導入
- 結合抽選が「セグメントごと」から「invasion ごと」へ

**凸領域も例外ではない** — 境界ハンドラと抽選の両方が変わったため。

実測の桁感: `axon_growth_modular` の同一 config が、抽選を per-invasion に移した時点で
**34,319 → 14,243 シナプス**。

詳細 → `docs/technical/axon_growth_confinement.md`

---

## 解析層が COO 専用化 → 重み統計が変わった

不在の結合がもう 0 を寄与しない。`block_values()` は `(weights, row, col, layout)` を取り、
実在するシナプスだけを見る。したがって対角ブロック（EE, II）— 以前は空の自己結合セルを
含んでいた — の平均が上がる。

実測: akita_soc のテスト (当時 `test/test_akita_soc.py`、現 `test/experiments/akita_soc/test_akita_soc.py`) の4ニューロンのフィクスチャで **ee 0.3 → 0.6**、**ii 0.4 → 0.8**。

**全結合でない config では、古い run の指標は新しい run と比較できない。**

---

## Akita run ディレクトリが COO 専用に。旧密形式のリーダは無い

各 run は `connectivity.npz`（row/col/shape）を1度書き、加えて `data` だけを持つ
`weights_{h}h.npz` を書く。

`runio.load_weight_values()` は legacy の `weights` キーを持つ npz に対して、どの 0 がシナプス
だったかを推測せずに **raise する**。

**旧密形式の run は、解析するには再実行するしかない。**

---

## NetworkLayout 導入: ID 割り当てが変わった

`group_info`（dict）が `NetworkLayout` に置き換わった際、ID 割り当てがランダム散布から
config 順の連番に変わった（その後 `layout.assignment` として明示的な選択肢になった）。
シード → ネットワーク実現が非互換。Akita 実験に影響。

`sequential` ↔ `random` の切り替え一般については `docs/technical/reproducibility.md` §5。

---

## 領域(area)の幾何を変えると重み・遅延もずれる

汎用 `Area.sample()` は棄却サンプリングなので draw 回数が形状に依存し、`NetworkBuilder.rng` の
下流（重み・遅延）が全部ずれる。`allow_soma: false` / `soma_in_bridge: false` も同様。

→ `docs/technical/reproducibility.md` §4

---

## develop: スパイク npz が記録窓の原点を持つようになった

`scripts/develop/` の `spikes_{h}h.npz` に `record_start_ms` を書くようにした
(`records.save_spikes()` / `load_spikes()`)。

それまで再解析はファイル名から記録時刻を復元していたが、ファイル名は `f"{hour:g}h"` =
**有効数字 6 桁**なので、`record_hours` が非整数だと往復で元に戻らない。1/3 h なら
記録窓の原点が約 1.2 ms ずれ、`burstiness_index` の 1000 ms ビン割りが変わって
**本番と再解析で値が食い違う**。`record_hours` が全部整数の run では差は出ない。

**このキーを持たない run は `scripts/develop/replot.py` が読めない** (`load_spikes` が
KeyError)。フォールバックは置かない —— 置いても、ずれた値を静かに返すだけなので。
解析し直したければ再実行すること。

あわせて `paths.records_dir()` (新レイアウト / organize 済み / organize 前の 3 通りを
吸収していた) を廃止し、npz の置き場所は `data/` だけになった。

---

## develop: `weight_block_metrics.csv` を出さなくなった

同じ重みブロック統計を `metrics.csv` の `weight_*` 列 (`weight_block_metrics()`) と
`weight_block_metrics.csv` (`compute_block_metrics()`) が別形式で持っていたので、後者を
廃止した。書いていたのは `weight_track.py` = **図のモジュール**で、図の生成に失敗すると
データまで失われる構造だった。

列の内容は `metrics.csv` 側が引き継いでいる (平均・飽和率)。`weight_block_metrics.csv`
だけにあった `max` / `ge_0p5_fraction` / `ge_0p9_fraction` / `mean_delta_from_previous` は
**無くなった**。必要になったら `metrics.py` に列として足すこと。

旧 `scripts/akita_soc_fig2.py` 経路は従来どおり出力する (`src/utils/experiments/akita_soc/`)。

---

## COO の並びを 1 つに統一した (`connectivity.npz` / `weights_*h.npz` の行順)

`simulator.synapse_connectivity_coo()` / `pull_synapse_coo()` が、GeNN の格納順ではなく
**行優先ソート順**で返すようになった。`NetworkBuilder.global_coo()` と同じ並びになる。

### なぜ 2 系統あったか

GeNN はシナプスを SynapseGroup (送信 population × 受信 population) ごとに持つ。各集団の
中だけが集団ローカルの行優先なので、値を取り出すと必ず「集団ごとの塊」で出てくる。
**1 つのニューロンから出るシナプスは複数の集団に分かれる**ため、GeNN の格納順が
グローバル行優先になることは原理的に無い。

その結果、同じネットワークに対して**本数が同じでどちらも正しい 2 通りの並び**が存在し、
位置で対応づけたコードが黙って別のシナプスに値を乗せる状態だった。
`src/utils/experiments/lesion/restore.py` の (pre, post) join はこれを避けるためのもので、
`scripts/develop/` の再ビルド照合も一度これで誤検知した。

### 何が変わるか

- **新しい run の `connectivity.npz` と `weights_*h.npz` は行の順序が変わる。**
  どのシナプスにどの重みが乗るかの対応は保たれる。
- **図は変わらない。** develop の 21 枚で md5 一致を確認済み (統計も `densify()` も
  集合演算なので順序に依らない)。重み軌跡をアルファ合成で重ね描きする fig2c も一致した。
- **`metrics.csv` は最終桁が動きうる。** 実測では 1 フィールド (`weight_mean`) が
  1 ULP だけ違った (`0.00017078643355964877` → `0.0001707864335596488`)。配列の並びが
  変われば `np.mean` の加算順が変わるため。**科学的な結論に影響する差ではないが、
  バイト一致での比較はできない。** 旧 run と新 run の CSV を突き合わせるときは
  許容誤差を置くこと。
- `synapse_connectivity_coo()` / `pull_synapse_coo()` の戻り値から `pair_names` と
  `pair_offsets` を削除した。グローバルソート後は意味を持たず、`simulator.py` の外に
  利用者もいなかった。

### 古い run

**古い run の npz はそのまま (GeNN の格納順)。** ファイル内では row/col と data の index が
揃っているので、順序に依存しない読み方 (統計・`densify`・(pre,post) join) をする限り
正しく読める。

ただし「記録は行優先」を仮定するコードを書くと静かに壊れるので、
`scripts/develop/store/records.py::load_connectivity()` が**読んだ時点で行優先かを検査し、
破れていたら拒否する**。develop の「旧レイアウトの run は受け付けない」方針と揃えてある。
古い run を develop で再解析したければ再実行すること。

`scripts/lesion.py` は `restore.align_saved_to_coo()` が (pre, post) で引き当てるので、
古い親 run でも正しく動く。

### 性能

置換は結合構造と同じく run を通して不変なので、`lexsort` は run につき 1 回だけ。記録
ごとに払うのは O(nnz) の gather。N=40000 / シナプス 1000 万本で、ソートが 1〜2 秒 (setup 時)、
gather が 20 ms 程度 (10 分の記録窓に対して無視できる)。

## 2026-09-22 — 実験スクリプトの構造統一 (`scripts/<実験>/` 4 層)

`scripts/akita_soc_fig2.py` と `scripts/lesion.py` を `scripts/develop/` と同じ形へ
作り直した。**旧レイアウトの run は読めない。**

| 旧 | 新 |
|---|---|
| `outputs/<日時>_seedN/` に平置き | `outputs/<実験>/<条件>/` の `data/` と `figures/<種類>/` |
| 走り終えてから `organize_output()` で移す | 最初から `data/` へ書く |
| 記録時刻をファイル名から復元 | 窓の原点は npz の `record_start_ms` が持つ |
| 密形式の重み (`weights` キーに (N,N)) | COO の値ベクトル (`data` キー) + `connectivity.npz` |
| lesion の時刻は `probes.csv` を index で引く | 窓の npz が原点・窓幅・phase を持つ |

**読めない run は黙って飛ばさずエラーで止まる。** 列や図を減らして続行すると、あとで
図と突き合わせたときに原因が追えなくなるため。読みたい場合は再実行すること。

### 結果そのものは変わらない

seed / backend / ネットワーク実現の決定経路には手を入れていない。develop の run で
図 21 枚のバイト一致を確認済み (`test/experiments/develop/reference_figures.md5`)。

### 併せて動いたもの

- `src/utils/experiments/` を削除。akita_soc → `scripts/akita_soc/`、
  lesion → `scripts/lesion/`、c_elegans → `scripts/tools/c_elegans.py`、
  beggs_plenz → `src/utils/analysis/{avalanche,criticality}.py`
- `scripts/tools/runview.py` → `src/utils/runview.py` (共有層の図が契約に対して
  書かれるようになったため)
- `src/utils/plotting/` の全関数が `(view, out_path)` を取る形になった。
  旧 API (`plot_raster(times, ids, out_path, title, ...)` など) は**残していない**
- `configs/akita_soc.yaml` → `scripts/akita_soc/akita_soc.yaml`
- `configs/components/tasks.yaml` の `akita_soc_fig2*` / `beggs_plenz_*` を削除
  (前者は `scripts/akita_soc/task.yaml` へ、後者は対応スクリプトが無いので廃止)。
  テスト用の短い記録プロトコルは `smoke` に改名

---

## `simulation.seed` の `[1, 10]` の意味が変わった

**入力 YAML の読み方だけの変更。** run の `config.yaml` に記録される seed は以前も今も
スカラーなので、**既に走り終えた run の解釈は何も変わらない。**

| | 旧 | 新 |
|---|---|---|
| `seed: [1, 10]` | seed 1〜10 の **10 本** | seed 1 と 10 の **2 本** |
| `seed: [1, 10, 2]` | 刻み付き = 1, 3, 5, 7, 9 | 1 と 10 と 2 の **3 本** |
| `seed: [1, 2, 3, 4]` | エラー (範囲として読めない) | 並べた **4 本** |
| `seed: [1..10]` | — | seed 1〜10 の 10 本 |
| `seed: [1..10:2]` | — | 1, 3, 5, 7, 9 |

範囲を別の綴りへ移し、リストは書いたとおりに読むようにした。`task.record_hours` と
同じ規則 (`config_manager.expand_seed_spec()` / `expand_hours_spec()`)。

**古い書き方は落ちずに別の本数で走る**ので、入力 config を見直すこと。リポジトリ内で
リスト指定を使っていたのは `scripts/develop/axon_growth_hierarchy.yaml` の 1 つだけで、
`[1..8]` へ変換済み。

### `task.record_hours` に範囲記法が入った

同時に `record_hours` でも `[0..12]` と書けるようにした。こちらは**リストの意味を
変えていない**ので既存のプロファイルはそのまま
(`[0, 6, 72]` は 3 点、`[0, 1, 3, 6, 12, 24, 48, 72]` は 8 点)。

### YAML の重複キーがエラーになった

`config_manager.load_yaml()` を通す読み出しは、同じマッピングに同じキーが 2 度出たら
止まる。PyYAML の既定は後勝ちで先に書いたほうを黙って捨てるため、プロファイルを
足したつもりが既存の定義を消していても**別のプロトコルで走った run** ができていた。
既存の YAML に重複は無かったので、動作が変わる config は無い。
