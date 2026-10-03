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
(`skip weight network: coords がありません (この記録を足す前に走った run です ...)`)。
**replot は書き足さない** —— 記録を後から作る経路を持たない方針。欲しければ run を
やり直すこと。

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

---

## lesion: ベースラインの出どころ・タイムラインの書き方・出力先

損傷実験のプロトコルを 4 点変えた。**旧 run の `metrics.csv` と新 run のそれは、
ベースライン行 (`probe=0`) の意味が違うので並べて読めない。**

- **ベースラインを測らなくなった。** 旧: 切断前のネットワークをもう一度 build して
  settle してから 1 窓測る (GeNN のコンパイルが run につき 2 回)。
  新: 親 run の `spikes_{h}h.npz` / `weights_{h}h.npz` をそのまま probe 0 に置く
  (コンパイルは 1 回)。ベースラインの窓幅も親のものになる
  (`task.record_window_ms` とは違いうる)。
- **タイムラインが `record_hours` のリストになった。** 旧: `recovery_hours` +
  `probe_interval_hours` (等間隔)。新: develop と同じ時刻のリスト。
  `baseline_window_ms` は廃止 (ベースラインが親のものになったので指定できない)。
- **出力先が `<親 run>/<切断条件>/` になった** (旧 `outputs/lesion/<条件>/`)。
  `--condition` 引数は無い。ディレクトリ名は切断 spec から機械的に決まる。
- **解析パラメータを task から外した。** `hub_z` / `include_betweenness` /
  `include_clustering` は `src/utils/analysis/graph.py` の既定値が正になった。
  **値は同じ** (2.5 / true / true) なので指標は変わらないが、旧 run の `config.yaml` には
  残っており、新しいコードはそれを読まない。
- **`figures/overview/` に 2 枚増えた。** `weight_trajectories` と
  `weight_distribution_shift` は実装も import もありながら登録簿に載っておらず、
  出ていなかった。

`task.duration` は 3 実験 (develop / akita_soc / lesion) すべてから削除した。
run 長を決めていたのは常に `record_hours` の最大値 + `record_window_ms` で、
`duration` はそれを手計算した値が記録に残るだけのフィールドだった。
**結果は変わらない** (どこもこの値で走っていない) が、旧 run の `config.yaml` には
残っている。

---

## 重み分布図: 横軸 [0, 1] 固定 / 縦軸を確率密度に (develop / akita_soc / lesion)

`weight_distribution` (構造図の `weight_distribution_initial.png` と
`panels/weight_distribution/`) の軸を 3 実験とも変えた。

- **横軸は常に [0, 1]。** 可塑性の Wmax で正規化済みなので、データに合わせると記録時刻
  ごとに軸が動いて分布の移動が読めない。ビン境界も [0, 1] を 80 等分に固定したので、
  **記録時刻をまたいで図を重ねられる**。
- **縦軸は確率密度** (`density=True`)。シナプス本数が違う run どうし、および損傷の
  前後でも形をそのまま比べられる。

`delay_distribution` / `distance_distribution` は変えていない (物理量なのでデータ範囲のまま)。

**範囲の外の値は描かれない。** 重みが 1 を超える設定にした場合、その分は棒として出ない
(左パネルのタイトルの `max=` には出るので、気づけないことはない)。

## lesion の `structure/` `panels/` を develop と同じ一覧に

lesion の図は develop の複製だったが、develop 側の更新が反映されず古い版のままだった。
**`figures/` と `report/{structure,panels}.py` を develop の現行版で置き直した。**

- `panels/` に 3 種類増えた: `weight_distribution/` `weight_network/` `trace/`
- `structure/weight_distribution.png` → **`weight_distribution_initial.png` へ改名**
- `structure/connection_probability.png` が E/I 4 本 + 理論曲線 → **実測 1 本**に
- `structure/network_sample.png` が重みではなく**結合マスク**から描くように (線幅一定)
- 窓のタイトルだけ develop と違う。`0 h` ではなく `baseline (-0.17 h)` / `post (+3.00 h)`
  (損傷実験では時刻だけでは切断の前後が読めないため)

`task.trace_neuron` / `trace_window_s` も develop と同じものを持つようになった
(既定 `null` = 採らない)。ベースラインの probe は**親 run が採っていればその npz を
そのまま引き継ぐ**ので、develop と lesion の契約が 7 → 8 ファイルになった。

## ブロック境界線: 白+黒のサンドイッチをやめ、グレー 1 色に (develop / akita_soc / lesion)

`style.draw_block_boundaries()` が引く線 (重み行列・粗視化結合図) の見た目を変えた。

**旧: 白 1.2pt の上に黒 0.4pt を重ねる。** 黒が白の中央を削るので 1 本の境界が
「白 1px + 黒 1px + 白 1px」になり、境界 15 本に対して白い筋が 30 本描かれていた。
そのうえデータ座標から画素への倍率が整数でないため線ごとに画素内の位置がずれ、
アンチエイリアスがそれを拾って**太さも濃さも 1 本ずつばらついた**。

**新: 中間グレー (`"0.5"`) 1 色、アンチエイリアス無効 + 画素スナップ。** 白 1 色では
粗視化結合図 (白背景) で消え、黒 1 色では重み行列 (viridis の暗い背景) で沈むので、
両方で読める中間グレーにした。線幅は階層レベルごとに 1.0 / 0.6 pt。

**図の見た目が変わるだけで、数値は変わらない。**

## lesion: post probe のスパイク時刻が settle のぶんずれていた (修正)

`run_one.probe()` は GeNN の時計が返すスパイク時刻を**そのまま** npz へ書いていた。
`record_start_ms` は**切断軸** (settle を含まない) の時刻なので、`load_spikes()` が引いた
窓ローカル時刻は `settle_ms` (既定 60,000 ms) ぶん先へずれていた。baseline の probe は
親 run の窓を切断軸へ移して書くので、**post の probe だけ**がずれていた。

影響 (`settle_ms > 0` で走った全 lesion run の post probe):

- **ラスターが真っ白。** 窓ローカル時刻が 60〜660 s に載るのに対し、図の x 範囲は 0〜30 s。
- **`burstiness_index` が壊れる。** `[0, record_window_ms]` を 1 s ビンで切るので、
  先頭 `settle_ms` ぶんのビンが空になり (指標を押し上げる)、末尾へはみ出したスパイクは
  数えられない。
- アバランチ分布・発火率・ISI は時刻の**差**しか見ないので変わらない。

**古い run は再実行するしかない。** npz が持つ時刻そのものが間違っているので、再解析
(`replot.py`) では直らない。`records.save_spikes()` が窓の外のスパイクを弾くようになった
ので、同じずれが静かに通ることは無い。

## develop / lesion: ニューロン間相関の図と指標を追加

Watanabe et al. (2026) *Communications Engineering* 5:150 の "neuronal correlation"
(発火率時系列のゼロラグ Pearson 相関、式 1) を `src/utils/analysis/correlation.py` に
入れ、両実験の `panels/` と `metrics.csv` に出すようにした。

- `panels/correlation/correlation_{tag}.png` が増えた (論文 Fig. 2b に対応)
- `overview/mean_correlation_track.png` が増えた (平均相関の時間発展)。値は
  `metrics.csv` を読み直すので、この図と表と相関行列のタイトルは必ず一致する
- `panels/correlation_network/correlation_network_{tag}.png` が増えた (論文 Fig. 3c 上段:
  相関を体細胞の位置の上に描いた無向グラフ)。描くのは**相関が
  `correlation_network.MIN_CORRELATION` (= 0.4) 以上の全ペア**で、結合の有無も距離も
  見ない。閾値は絶対値で固定 —— 窓ごとに分位点で決めると系全体の相関が落ちても同じ
  本数が残り、切断で同期が失われたことが図に出なくなる
- `panels/weight_network/` が**エリアの境界線を背景に敷くようになった** (`network_sample`
  と同じ)。軸範囲もエリアの境界箱から取るので、`space: area_uniform` の run で
  autoscale に落ちなくなった

これに伴う変更が 2 つ:

- 両実験の `store/views.py` に **`Series.area()` / `Window.area()`** が増えた (config から
  area を組み直す)。`BaseArea` は RNG を消費しないので run の再現性には影響しない
- `figures/network.py` の `_draw_nodes` / `_apply_axis_limits` / `_excitatory_mask` が
  `draw_nodes` / `apply_axis_limits` / `excitatory_mask` へ改名された
  (`correlation_network.py` も呼ぶようになったため)
- `metrics.csv` に `mean_correlation` と `correlation_pairs` の 2 列が増えた。
  lesion では baseline の行にも出るので、切断前後の差は `metrics_delta.csv` の
  `delta_mean_correlation` / `ratio_mean_correlation` が自動で持つ

r(t) のビン幅は **100 ms** (`correlation.RATE_BIN_MS`)。**変えると相関の値が変わる。**

論文のフレームレート (20 fps = 50 ms) には合わせていない。論文の r(t) は GCaMP6s
(減衰 ~1 s) を CASCADE で推定したもので、実効的な時間分解能はフレーム間隔より粗い ——
揃えるべきはサンプリング間隔ではなく「1 回の同期イベントが 1 ビンに入る」ことだと判断した。

`b50_seg1_g002F_U8_W0_20h_sweep` の 8 seed で、baseline と切断 +3 h の窓を掃引した実測
(平均 ± SD):

| bin [ms] | 平均相関 (bef) | Δ 平均相関 | 負のペアの割合 |
|---|---|---|---|
| 25 | 0.046 ± 0.014 | −28.8 ± 18.7 % | 16.0 ± 6.1 % |
| 50 | 0.076 ± 0.023 | −29.8 ± 18.0 % | 13.1 ± 5.4 % |
| **100** | **0.104 ± 0.031** | **−30.6 ± 17.4 %** | **10.8 ± 4.6 %** |
| 250 | 0.113 ± 0.030 | −31.6 ± 15.2 % | 10.6 ± 3.8 % |
| 500 | 0.091 ± 0.019 | −25.9 ± 12.9 % | 15.9 ± 4.8 % |
| 1000 | 0.079 ± 0.014 | −25.4 ± 11.8 % | 21.0 ± 4.1 % |
| 2000 | 0.074 ± 0.014 | −21.4 ± 14.9 % | 25.2 ± 4.7 % |

100–250 ms が 3 つの指標すべてで最良 —— 相関の大きさが最大 (細かすぎるビンは発火が
まばらで推定が減衰する)、切断による低下が最大、推定ノイズで負に落ちるペアが最少
(粗いビンは標本数が減るぶんノイズが増える)。両側でなだらかに悪化するので、
**この範囲の中での 100 か 250 かは結果を変えない**。

切断によるモジュール間相関の消失そのものは 25 ms〜10 s のどのビン幅でも見える
(ビン幅は効果の有無ではなく感度を決める)。

**既存の数値は変わらない。** 増えたのは列と図で、既存 run は `replot.py` を通せば
新しい列と図が付く。

## lesion: probe の番号が `record_hours` の添字と一致するようになった

以前はベースラインが `p000` を取り、post の probe が `p001` から始まっていた。
`record_hours: [0..20]` を指定すると記録は `p001`〜`p021` になり、「3 番目の記録時刻」と
`p003` が食い違っていた。

**新: post は `p000`〜`p020` (= `record_hours` の添字)。ベースラインは番号を持たず
`{kind}_baseline.npz`。** ベースラインはそもそも別種 (親 run 由来で窓幅も別) なので、
番号の並びを共有しない。`Window.index` と `metrics.csv` の `probe` 列は
ベースラインで `-1` になる。

図のファイル名も npz と同じ token に揃えた:

    旧  raster_p000_baseline.png / raster_p004_post.png
    新  raster_baseline.png      / raster_p003.png

**古い run は再解析しても新しい名前にならない** (ファイル名は書いた時点で決まる)。
新旧を並べるときはディレクトリごと分けること。

同時に lesion の `weight_network` のタイトルを `3 h` から `post (+3.00 h)` へ直した
(他の lesion の図と同じ `Window.label`)。切断の前後が読めなかったため。

## develop / lesion: 臨界性と同期の列を metrics.csv へ追加

- **同期イベント** (`src/utils/analysis/synchrony.py`): `event_bin_ms` /
  `event_rate_hz` / `event_participation` / `event_participation_ceiling` /
  `synchronous_fraction` / `active_fraction_median`。
  ビン幅は `correlation.RATE_BIN_MS` (= 100 ms) を共有する。**平均発火率の逆数は
  使わない** —— 1 ビンあたり平均 1 発になる幅だとほぼ全ビンがイベント判定になり
  (実測で 99%)、損傷の前後で参加率が動かなくなる
- **アバランチサイズ** (`avalanche.avalanche_size_columns`): `avalanche_rate_hz` /
  `avalanche_size_mean` / `avalanche_size_median` / `avalanche_size_max`。両実験
- **分岐比の一式**: develop にも `bp_*` (σ_bp・α_size・寿命指数・各種チェック) が
  付くようになった (lesion は以前から持っていた)。**臨界かどうかの判定は σ_bp**。
  サイズ分布の傾きは亜臨界でもべき乗に見える

**既存の数値は変わらない。** 増えたのは列だけで、既存 run は `replot.py` を通せば付く。
解釈と方針は `docs/technical/criticality_notes.md`。

## develop / lesion: ラスターの y 軸がモジュール名になった

並べ替えた後の通し番号には意味が無いので、区切り線と同じ単位のラベル
(`style.outer_block_labels()`、重み行列・相関行列と同じもの) を出すようにした。
最外ブロックが 16 個を超える layout では従来どおり数値目盛りのまま。

**図の見た目が変わるだけで、数値は変わらない。**

## 相関のビン幅 100 ms → 30 ms、lesion のアバランチのビンは baseline 固定

- **`correlation.RATE_BIN_MS` = 30 ms** (旧 100 ms)。`mean_correlation` と
  `panels/correlation*` の値が変わる。基準を「平均相関の最大化」から
  「確実に相関しているペアの検出力」へ変えた結果 —— 分離度 (サロゲートに対する d) は
  30 ms で最大 (35.0 ± 3.2)、100 ms では 27.4 ± 2.1 まで落ちる。
  30 ms は相互相関ヒストグラムの半値全幅とも一致する
- **`synchrony.EVENT_BIN_MS` = 100 ms** を新設し、相関のビンから切り離した。
  「集団の 20% が同時に撃つ」イベントは 50〜100 ms の現象で、30 ms では検出できない
- **lesion の `bp_*` のビンが baseline 固定**になった (`Series.avalanche_bin_ms()`)。
  従来は窓ごとの平均イベント間隔だったので、損傷で発火率が落ちるとビンが伸び、
  それだけで σ_bp が上がって見えていた (0.496 → 0.529)。固定すると 0.496 → 0.441 で
  **変化の符号が逆になる**

**既存 run の `metrics.csv` と相関の図は再解析で値が変わる。** 新旧を並べないこと。
根拠の数値は `docs/technical/criticality_notes.md`。

## lesion: ブリッジ切断がモジュール内へ食い込んでいた

`BridgeSelector` は「軸索がブリッジ part の内側を通ったか」で切っていた。ブリッジ part は
union を連結にするため両端を `bridge_overlap` だけモジュールへ食い込ませてあるので
(`area._bridge_spec`)、**その食い込みをかすめただけの、モジュール内で完結する軸索まで
切断対象に入っていた**。`hierarchical_modular_grid` の既定値 (side 200 / bridge_width 50 /
overlap 50) では、ブリッジ矩形 300 um のうち 100 um = 1/3 がモジュールの内側である。

**新: 判定は「ブリッジ part の内側かつ全モジュール part の外側」= 露出した通路だけ**
(`axons.synapse_crossed_bridges()` / `connectivity.module_part_indices()`)。

実測 (`b50_seg1_g002F_U8_W0_20h_sweep`, `bridge:parts=BX1-3`、切断本数 旧 → 新):

| seed | 全体 | 同一モジュール内 | 同一クラスタ内 |
|---|---|---|---|
| 01 | 101 → 43 | 44 → 2 | 58 → 2 |
| 02 |  91 → 28 | 37 → 2 | 55 → 3 |
| 03 | 156 → 59 | 79 → 3 | 95 → 4 |
| 04 |  90 → 57 | 25 → 1 | 34 → 1 |
| 05 | 139 → 45 | 55 → 0 | 83 → 0 |
| 06 | 131 → 78 | 40 → 5 | 53 → 6 |
| 07 | 123 → 46 | 51 → 1 | 75 → 1 |
| 08 | 208 → 89 | 61 → 1 | 93 → 1 |

**切断本数が 3〜7 割減り、クラスタ間ブリッジの切断なのに同一クラスタ内で完結していた
結合がほぼ消える** (旧は切断の 4〜7 割がそれだった)。新しい判定は旧判定の部分集合。

残る同一クラスタ内の数本は**正しい**: 軸索が露出部へ出て戻ってから接触した結合で、
通路を焼けば実際に切れる。経路の判定範囲が `offsets[pre] .. contact_seg` である
(= 接触より前に通ったものだけ数える) ことは元から正しく、そこは変えていない。

**既存の lesion run はすべて作り直しが必要。** `lesion.json` の `num_synapses_cut` /
`cut_profile` が過大なので、`replot.py` では直らない (切断そのものが違う)。

## 相関のビン幅 30 ms → 50 ms

- **`correlation.RATE_BIN_MS` = 50 ms** (旧 30 ms)。`mean_correlation` と
  `panels/correlation*` の値が変わる (相関は全体に上がり、`correlation_network` の
  閾値 0.4 を超える線が増える)。
- 論文 (Watanabe et al. 2026) のフレームレート 20 fps = 50 ms に合わせた。
  30 ms を選んだ基準 (偶然との分離度 d) で見ても 50 ms は 33.4 ± 2.8 と
  30 ms の 35.0 ± 3.2 から SD の範囲内で、検出率は 96.4 % とむしろわずかに高い。
  どのペアが相関しているか (順位) はビン幅にほぼ依らない
- `synchrony.EVENT_BIN_MS` (100 ms) とアバランチのビンは変わらない

**既存 run の `metrics.csv` と相関の図は `replot.py` を通すと値が変わる。** 新旧を
並べないこと (図のタイトルにビン幅が出るので、そこで見分けられる)。


---

## 結合生成を COO に一本化 (`network.sparse` 削除)。実現値は変わらない

- 結合・重み・遅延の全コンポーネントが COO を直接返すようになり、密な (N, N) で生成する経路と
  `network.sparse` (`auto` / `force` / `off` / `on`) を削除した。コンポーネントの API は
  `generate()` → `(rows, cols)` / `generate(rows, cols)` → 1 結合 1 値 (`generate_sparse` と
  `mask` 引数は無くなった)。
- **同じ config.yaml から同じネットワークができる。** 旧実装 (密経路を強制) と新実装の
  `GlobalCOO` (row / col / weights / delays) と生成後の rng 状態が、`outputs/` と `configs/` の
  seed 固定の 35 config でビット単位で一致した (Akita, axon_growth / axon_growth_fine,
  develop の seed 掃引, distance_based, beggs_plenz, C.elegans, akita_soc 遅延)。全クラス単体でも
  旧密版との一致を確認済みで、`test/models/network/test_coo_generation.py` がその実現値を固定している。
- 保存済み config.yaml にある `network.sparse` は読み込み時に無視される (どちらの値でも結果は同じ)。

---

## 2026-10-02 — 構造図を build の記録から描く (replot は再ビルドしない)。実現値は変わらない

**何が変わったか:**

- build の直後に記録が増えた (3 実験とも。一覧は各 `scripts/<実験>/store/build_record.py`):
  - `data/delays.npz` —— 遅延 [ms]。`connectivity.npz` と index 整合
  - `data/weights_initial.npz` —— build 直後の重み (lesion は切断直後 = 親から復元した値)。
    `record_hours` が 0 を含むかどうかによらず必ず書く
  - lesion の `data/axon_geometry.npz` —— **切断後**の軸索幾何 (以前は書いていなかった)
- `connectivity.npz` を書くタイミングが `sim.setup()` の後から **build の直後**へ移った。
  中身は同じ (`builder.global_coo()` の row / col / shape)。lesion の `connectivity.npz`
  (切断後) も同じ
- 構造図は本番も replot も `series.built()` (記録を読む view) から描く。**replot は
  ネットワークを組み直さない。** 再ビルドして記録と照合するのは `replot --verify` のときだけ。
  ふだんの守りは golden テスト (`test/test_network_golden.py`)
- `GeNNSimulator.synapse_connectivity_coo()` を削除した。結合構造は `builder.global_coo()`
  から取る (`pull_synapse_coo()` の戻り値は変わらない)
- `store/series.py` → `store/views.py`、lesion の `store/parent.py` → `store/parent_run.py` へ改名
- 記録を足す前に走った run でファイルが無いときの `MissingData` のメッセージが
  「この記録を足す前に走った run です」になった (`no_space` の「空間を持たない config です」と
  区別する)

**影響 (実測):** 図も指標も変わらない。develop / akita_soc / lesion の小さな run (CPU) で、
`run_one` と `replot --verify` の出力 (図・CSV・npz) がそれぞれ 40 / 22 / 57 ファイル
すべてバイト一致した。lesion の構造図 8 枚は旧経路 (再ビルドして切り直して描く) とも
バイト一致。builder から直接描いた構造図と記録から描いた構造図は、`axon_growth` (disk) と
`axon_growth_hierarchy` (ブリッジつき) の 2 構成で 8 枚と CSV 2 本がすべてバイト一致した。

**古い run:** replot はそのまま通る。`delays.npz` / `weights_initial.npz` が無いので、
構造図のうち `delay_distribution.png` と `weight_distribution_initial.png` (lesion は
`weight_distribution.png`) が skip される。lesion の旧 run は切断後の軸索幾何も無いので
`axon_network.png` も skip。それ以外の図と `metrics.csv` は以前と同じものが出る。
欲しければ run をやり直すこと。

---

## `delay_by_target` 廃止 → Akita の COO 遅延が実値に

シナプス群ごとの `delay_by_target` を廃止し、遅延プロファイル `akita_soc_axonal`
(`type_based_axonal`。`BaseDelay.axonal = True`) で同じ条件を表すようにした。
軸索遅延経路を使うかは遅延クラスの `axonal` 属性で決まる。

**力学は変わらない。** 旧 `delay_by_target` 版と新版で、GeNN に渡る結合・重み・
`axonal_delay_steps` (E→E 15 / その他 8 @ dt=0.1)・可塑性モデルの全状態が一致することを
`scripts/akita_soc/` の 4 config (`akita_soc` / `_delay` / `_delay_cap` / `_delay_clip`) で確認済み。

変わったのは**記録される遅延**だけ。旧版は GeNN 登録の直前で遅延を上書きしていたので、
`global_coo().delays` (= `connectivity.npz` の遅延) にはダミーの `constant` 1.0 ms が
入っていた。新版は実値 (1.5 / 0.8 ms) が入る。golden の `no_space` の delays ハッシュを更新した。

旧 run の `config.yaml` には `delay_by_target` が残っている。解析 (読み出し) はそのまま通るが、
それを `NetworkBuilder.build()` に渡すとエラーになる (無言で 1.0 ms の別条件で走るのを防ぐため)。
親 run から再シミュレーションする場合 (lesion, `replot --verify`) は、
`delay_by_target` を消して `network.delay` を `akita_soc_axonal` に書き換えること。
