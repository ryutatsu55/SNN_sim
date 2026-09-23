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

---

## 2026-09-22 — ΔCr と LLR を supplementary の字面へ戻した

`metrics.csv` の **`delta_cr` と `llr` の両列が変わる**。スパイク npz は変わらないので、
`replot` すれば全部作り直せる。影響は develop / akita_soc / lesion の 3 実験すべて
(指標は `src/utils/analysis/` の共有層にある)。

### ΔCr: 差を取る空間を log から確率へ戻した

supplementary S21-S24 は `pemp(s) − pfit(s)` を**確率のまま**足す。log-log なのは
回帰を引く平面だけで、pfit はその直線を確率へ戻したもの。旧実装は差も log 空間で
取っていたため、**回帰と評価が同じ平面になり最小二乗の性質から Σ残差 = 0**、つまり
`Aupper = |Alower|` となって S24 の「絶対値の大きい方」が 1e-16 の丸めで決まっていた。

**回帰は観測個数で重み付ける (加重最小二乗)。** log10(pemp(s)) の分散は 1/count(s) に
比例するので、等重みだと数個しか観測の無い裾に直線が引きずられ、小サイズ側の切片が
跳ね上がる (0h で pfit(1)=2.9。論文 Fig S1 の回帰線は s=1 で ~0.8)。逆分散重み = count が
この分布に対する正しい回帰で、副作用として **値の尺度が標本数で動かなくなり、Tetzlaff の
±0.195 と同じ世界に収まる**。

| | 等重み | 重み付き |
|---|---|---|
| 0h (ポアソン) | −2.69 | **−0.46** (論文 −0.23) |
| 72h | −1.33 | **−0.053** (論文 +0.017) |
| pfit(1) の最大 | 2.9 (>1) | 0.66 |
| 指数分布 n=2000→50000 | −1.36 → −4.82 (膨張) | −0.50 → −0.45 (安定) |

**既知の限界**: 合成した「大サイズのこぶ」(超臨界) は等重みなら +0.52 だが、重み付きでは
−0.33 で**正にならない**。論文 Fig 2(c) の値も超臨界とされる 6h で −0.042 なので数値とは
整合するが、論文の文章 (positive = supercritical) とは整合しない。**時間発展で読むこと。**

実測 (同じ指数分布の標本、smin=1、等重み):

| データ | 旧 (log 平均) | 新 (確率の和) |
|---|---|---|
| 指数 λ=0.46, n=20000 | −0.190 (upper=+0.190471 / lower=−0.190471) | −3.52 |
| 指数 λ=0.30, n=20000 | **+0.181**(符号が逆) | −3.75 |
| べき乗 α=1.5, n=40000 | −0.026 | +0.018 |
| 大サイズのこぶ | −0.164 (符号が逆) | +0.52 |

旧実装は `smin` を 1 より大きく取ると縮退を逃れたが、その値 (3) は
**論文 Fig 2(c) の 0h に合うよう較正した定数**だった。較正をやめ、既定は
`smin=1`(下限を切らない) / `smax = N`。

**pfit は再正規化しない。** pemp と同じ台で和を 1 にすると `Σ(pemp − pfit) = 0` から
`Aupper = |Alower|` となり、S24 が常に Aupper (≥ 0) を返して劣臨界を報告できなくなる。
検証と 0h 較正点のデータは `docs/technical/akita_soc_reproduction_memo.md` §12。

**絶対値の読み方が変わる。** この定義の値は Tetzlaff の判定幅 ±0.195 とスケールが
揃わない (べき乗から外れた分布では回帰直線が小サイズ側で 1 を超え、和が数単位まで
振れる)。**符号と時間発展で読むこと。** 論文の絶対値 (0h ≈ −0.22) とも一致しない ——
論文の smin 規準「線形フィットの SSE 最小」を素直に実装すると点を減らすほど誤差が
減って走査範囲の端に張り付き、0h(結合ゼロ = 純ポアソン)で ΔCr ≈ 0 になってしまう。
どの読み方も論文の 0h を再現しないので、**較正して合わせるのをやめた**。

### LLR: 打ち切りをフィット範囲に戻した

supplementary II.B は「系が 100 ニューロンなので、サイズ 1〜100 のアバランシェを
fitting に使った」と書く。旧既定は `smax=None`(打ち切りなし) で、これは
[1,100] では論文の LLR に原理的に届かない (べき乗である限り 1 アバランシェあたり
0.48 が上限) ことへの対応だった。**論文準拠を優先して既定を `smax=100` に戻し、
実験側は系サイズ N を明示的に渡す。**

影響: 100 を超えるアバランシェが出る run で `llr` が下がる。`fit_exponent` /
`fit_distribution_curves` が返す `llr` も同じ範囲になったので、**アバランシェ分布図の
凡例と `metrics.csv` は引き続き一致する**(旧実装でも一致していたが、α は [1,100]、
LLR は全域という不揃いだった)。fig2c の LLR パネルに重なる天井線とも土俵が揃った。

**天井は 1 アバランシェあたりの量で、比べる相手は `天井 × アバランシェ数`。**
smax にも依存する (100 → 0.480, 256 → 0.756, 500 → 0.986)。LLR は和なので、
アバランシェ数の多い run は [1,N] 打ち切りでも数万を出す (例: N=256 の
`axon_growth_grid/b20_seg100_g005F_U8` は 194413 アバランシェで LLR=58970 ——
ただし 1 個あたり 0.303 = 天井の 40% にすぎない)。**見るべきは 1 個あたりの値。**

論文 72h (19862) は [1,100] の天井を超えて見えるが、**分母のアバランシェ数は論文に
書かれておらず、こちらの 72h (23839 個) からの推定**である点に注意。

---

## 2026-09-23 — 図の置き場所と、develop の構造図 3 枚

シミュレーションの実現値は変わらない。**変わるのは図だけ**なので、旧 run の `data/`
(npz / csv) はそのまま比較できる。図を揃えたければ `replot.py` を掛け直すこと。

### 置き場所: 記録窓ごとの図が `figures/panels/` の下へ (全実験)

`panels.py` が出す図の置き場所が 1 階層深くなった。

| 旧 | 新 |
|---|---|
| `figures/raster/` | `figures/panels/raster/` |
| `figures/avalanche/` | `figures/panels/avalanche/` |
| `figures/weight/` | `figures/panels/weight/` |
| `figures/trace/` | `figures/panels/trace/` |

`figures/structure/` と `figures/overview/` は動かない (run に 1 枚しか出ないため)。
**旧 run を replot すると古い場所のファイルが残る** —— 消すのは手作業。

### 追加: 記録窓ごとの重み行列 (lesion)

`figures/panels/weight/weight_matrix_{tag}.png`。develop / akita_soc と同じ図を lesion も
出すようになった (`scripts/lesion/figures/weight_matrix.py` を新設)。**baseline の probe は
切断前の結合、post は切断後の結合**を描く (`Window.wiring()` が phase で切り替えるため)。

### 追加: 記録窓ごとのネットワーク図 (develop)

`figures/panels/weight/weight_network_{tag}.png`。構造図の `network_sample` を
**その時刻の重み**で描いたもので、線の太さが `|w|`、重み 0 の結合は描かない。
構造図側が結合マスクだけを描くようになったぶん、重みの可視化をこちらが引き取る。

### 追加: `data/coords.npz` (全実験)

soma の座標を記録するようになった。座標は config だけからは復元できない (空間コンポーネントが
RNG を引く) ので、従来は**再ビルドしないと空間の図が描けなかった** —— 記録窓の view
(`Window`) は `coords()` に `MissingData` を返すしかなかった。

`save_coords()` / `load_coords()` / `Series.coords()` / `Window.coords()` を
**develop / akita_soc / lesion の 3 つとも**持つ (`weight_network` を出すのは develop
だけだが、契約に穴を開けないため機構は揃える)。lesion では切断の前後で 1 つ ——
切るのはシナプスであってニューロンの位置ではない。

**`no_space` の run では書かない。**「ファイルが無い = 空間を持たない run」を読む側の
判定にしている (akita_soc の論文再現 config は常にこちら)。

**旧 run にはこのファイルが無い。** replot しても座標を要する図だけは skip される
(`skip weight network: coords がありません`)。**replot は書き足さない** —— 再ビルドの
ついでに backfill することはできるが、記録を後から作る経路を持たない方針。
欲しければ run をやり直すこと。

### 追加: 記録窓ごとの重み分布 (develop / akita_soc)

`figures/panels/weight/weight_distribution_{tag}.png`。中身は構造図の
`delay_distribution` と同じ形 (左: 全体 / 右: E/I ブロック別)。これに伴い、構造図の
`weight_distribution.png` は **`weight_distribution_initial.png` へ改名**した
(build 直後 = 可塑性が 1 ステップも走る前、であることを名前に出すため)。

### develop の構造図 3 枚 (akita_soc / lesion は対象外)

- **`weight_matrix`** (lesion にも同じものを新設): 並べ替えが `("polarity",)` →
  **`("module", "polarity")`**。粗視化結合図と同じブロック配置・同じブロック名の目盛りに
  なったので、2 枚を位置で突き合わせられる。旧図と行・列の対応が付かない。
  **E/I の境界線は引かなくなった** —— 線は最外ブロック (module) だけ。粗視化結合図が
  元からそうで、2 枚が同じ格子に見えることが並べて読むための条件のため。
- **`connection_probability`**: E/I ブロック別の 4 本 + 理論曲線 → **実測 1 本だけ**。
  `p0*exp(-d^2/2σ^2)` は軸索伸長のコネクタが従う式ではないので、重ねる根拠が無い。
- **`network_sample`**: 重み (COO) ではなく**結合マスク**から描くようになった。線幅は
  一定。初期重みが全部 0 の config (`weight: constant_zero`) では、旧実装は
  `abs(w) != 0` で全エッジを落としていたので**線が 1 本も出ていなかった**。
  旧実装と同じ「重みで描く」絵は `weight_network` として panels に移した。
