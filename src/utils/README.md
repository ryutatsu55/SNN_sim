# src/utils — 契約 / 解析 / 描画

`src/utils/` を触る前に読む。プロジェクトの全体像はルート `README.md`。

**何に触れてよいか**で分かれている。

- **runview.py** — 実験出力の**読み出し契約**。numpy 以外は何も import しない
  (matplotlib も NetworkBuilder も見ない)。
- **analysis/** — 純粋な計算。matplotlib を決して import しない。
- **plotting/** — 汎用の描画のみ。保存する png / mp4 以外のファイルを読み書きしない。

**`src/utils/` は `src/models` を import しない。** 必要な対象（area、軸索ジオメトリ、
NetworkBuilder）は duck-typing で受け取る。

---

## analysis/

- `avalanche.py` — アバランシェ抽出
- `powerlaw.py` — 離散打ち切り MLE、LLR、経験 PMF
- `criticality.py` — ΔCr、burstiness、bimodality D、相互相関
- `spikes.py` — 発火率、CSV 書き出し
- `correlation.py` — ニューロン間の機能的結合（発火率時系列のゼロラグ Pearson 相関）
- `synchrony.py` — 集団同期の粗さ（イベント頻度・参加率・発火率で決まるその上限）、
  ネットワークバースト（連続するイベントビンを 1 回にまとめた頻度・幅・参加率）、
  群ごとの同期（参加率の時系列、Golomb の χ）、全体イベント（サロゲートで決めた閾値と
  偶然の分を除いた深さで、深くて多くのモジュールに広がるイベントと 1 モジュールだけのイベントを分ける）
- `transfer_entropy.py` — 実効結合（一般化移動エントロピー GTE と、損傷前に固定した結合集合での比 RTE。
  Watanabe et al. 2026 式 2・3）
- `weights.py` — E/I ブロック分解と重み統計 (COO のみ、O(nnz))、および
  **保存済みの重みを別の COO へ載せ直す** (`align_subset_to_coo` / `align_saved_to_coo`)
- `connectivity.py` — **任意の軸**でのグループ間結合確率、および同じ数値を
  **ブリッジを何本跨ぐか**で組み替えたもの
- `isi.py` — スパイク間隔の構造（ニューロンごとの CV、Lv、Fano、沈黙率）
- `graph.py` — ハブ指標（次数/強度、媒介中心性、クラスタ係数、participation coefficient と
  within-module degree z-score = Guimerà-Amaral の組）
- `axons.py` — どのシナプスの軸索がどの領域パーツを横切ったか。ブリッジ損傷実験の入力

**「ブリッジ」の定義はただ1箇所にある** — `connectivity.bridge_part_indices()` =
`allow_soma: false` のパーツ。補集合は `module_part_indices()`。

**ブリッジを「渡った」の判定はパーツの内外判定ではない。** ブリッジのパーツは union を
連結にするため両端を `bridge_overlap` だけモジュールへ食い込ませてあるので、内外判定だけでは
モジュール内で完結する軸索まで拾う。渡ったかどうかは
`axons.synapse_crossed_bridges()` =「ブリッジパーツの内側**かつ**全モジュールパーツの外側」
で、`synapse_crossed_parts()` はその素材となるプリミティブ（`exclude_part_indices` を取る）。

**重みの引き当ては位置ではなく (pre, post) で行う。** 損傷後のネットワークは損傷前の
部分集合なので位置では並ばないし、並びを揃える前に作られた記録は GeNN の格納順で
書かれている。位置で対応づけると気付かないまま別のシナプスに値が乗る。ここに置いてある
のは純粋な COO の索引計算だからで、損傷実験の `figures/` が実験の `analysis/` を
import せずに済む効果もある。

### weights.py と connectivity.py の分担

`weights.py` は*シナプスごとの値*を固定の4つの E/I ブロックへ分解する。
`connectivity.py` は*シナプスが存在するか*を数え、グループ化する軸を引数に取る
（`module` / `layer` / `polarity` がそのまま効く）。

分母は**使えるペア数**（`n_i·n_j`、対角は `allow_self_connections=False` が通例なので
`n_i·(n_i−1)`）で、これが不揃いなグループサイズを比較可能にしている。
`within_probability` / `between_probability` はグループごとの確率を平均するのではなく
分子と分母をプールする — グループが不揃いだと両者は異なり、プールした値の方が
「層内のペアを1つ引いたとき、それは繋がっているか？」に答える。

`bridge_hop_matrix()` / `hop_connection_probability()` はプールされた非対角の値を分解する。
モジュール領域では非対角は一様でない（隣接モジュールはブリッジ1本、対角は2本）ので、
ペア集合を**最短ブリッジ本数**で組み替え、上と全く同じように hop ごとに分子分母をプールする。
hop 0 は対角なので `within_probability` を厳密に再現し、到達不能なペアは `-1` として
独立の行になる。モジュールグラフは area の `parts` から幾何学的に導く: 2つのパーツは
**バウンディングボックスの交差**上のグリッドが両方で `contains()` になる点に当たれば隣接、
そして 0-1 BFS が体細胞不可のパーツへ入るとき 1、それ以外は 0 を課す。
この分類だけが唯一の要件 — ブリッジにも体細胞が居る area (`soma_in_bridge: true`) には
課金すべきパーツが無いので、`bridge_hop_matrix()` は全部 0 hop と報告するのではなく raise する。

---

## runview.py — 読み出し契約

解析・描画関数が「どの実験のデータか」ではなく**この契約**に対して書かれるようにする
ための取り決め。

    RunView          1 つの run を、ある時点から見たもの
    ├── Built        build 直後
    │   └── BuiltNetwork   記録を持たない道具のための実装 (NetworkBuilder から作る)
    │                      実験の実装は各実験の store/views.py (build の記録を読む)
    ├── Window       記録窓 1 つ
    │   └── MemoryWindow   記録を持たない道具のための例外 (docstring を読むこと)
    └── Series       run 全体

基底が持つデータは `run_dir` / `config` / `layout` の 3 つだけ。それ以外は
「その名前で呼べる」ことだけを決めたアクセサで、既定は `MissingData` を投げる。
**実験ごとに違うのは実装 (`scripts/<実験>/store/`) であって、名前と形ではない。**

`MissingData` を投げる側 (契約) と受ける側 (`guard()` / `optional()`) は同じファイルにある。

設計の背景は `docs/architecture/runview_contract.md`。

---

## plotting/

`raster.py`, `traces.py`, `network.py`, `matrices.py`, `distributions.py`, `animation.py`,
`area.py`、加えて `ordering.py`（ID の permutation と軸の選択）と `common.py`、
`model_test.py`。

**全関数が `(view, out_path)` を取る。** `view` は `runview.py` の契約で、必要なデータは
図が自分で取る。持っていないものを要求すると view が `MissingData` を投げ、受けるのは呼び出し側の `guard()`。

例外は 2 つ。`ax` を取るプリミティブ（`draw_area` / `draw_discrete_distribution` /
`draw_block_boundaries`）と、`model_test.py`（モデル単体を手で駆動した結果を描く図。
run ではないので契約を取らない）。

軸範囲・色・点の大きさ・dpi・figsize・乱数 seed は引数ではなく、モジュール先頭の名前付き定数。
並べ替え軸も引数ではなく、`ordering.available_order_axes(layout)` が layout を見て
`("module", "polarity")` か `("polarity",)` を選ぶ。ラスターと粗視化結合図はいつも同じ並びになる。

分割は**その図が何についてのものか**であって、どのパイプラインが作ったかではない:

- `network.py` = 空間に置かれたネットワーク（座標が要る）
- `matrices.py` = 結合そのもの（重み画像、粗視化した密度、距離に対する結合確率 =
  距離で周辺化した行列）
- `distributions.py` = 値の分布（重み、遅延、距離、アバランシェサイズ）

どれにも当てはまらない図は、それが実験固有であるサインで、`scripts/<実験>/figures/` に属する。

シナプスごとのヒストグラムは全て1つの本体 `_synapse_value_distribution()` を共有する
（左パネル = 全シナプス、右 = 同じビン境界上の E/I ブロック）。`delay_distribution` /
`distance_distribution` / `weight_distribution` はその数行のラッパ。

距離に関する2つの図の対応に注意: `distance_distribution` は**距離あたりのシナプス数**を
数え、`empirical_connection_probability` はその距離で**使えるペア数**で割る。
ペア数は距離とともに増えるので、単調減少する確率でもピークを持つカウントを生む。
一方をもう一方のスケール変換として読まないこと。

`common.py` にはいくつかのモジュールが共有するものだけを置く — `save_figure()` と、
E/I ブロック図で EE の色を常に同じに保つ `BLOCK_COLORS` パレット。`ordering.py` の
`draw_block_boundaries()` も同じ理由でそこにある（重み行列と粗視化図の**2 枚が同じ線を
引く**）。

**受け渡しは COO 一本**（`view.coo()` が `row` / `col` / インデックスの揃った1次元配列を返す）。
(N, N) 行列を取る関数は無い。画像が必要なのは `matrices.py` だけなので、そこが内部で
densify する。

### network.py は空間ネットワーク図を2つ持つ

- `network()` — 各シナプスを体細胞間の直線矢印として描く
- `axon_network()` — 同じネットワークを、各シナプスを**それを実際に作った軸索**に沿って
  描く（体細胞 → ポリライン → 接触点、そこから樹状突起のリーチを破線のスタブで）。加えて
  サンプルした全ニューロンの軸索全体を薄い下地として描く。`connection: axon_growth` にのみ
  適用され、ジオメトリを `AxonGrowthTopology.axon_geometry()` から duck-typed で取る
  （`area.py` と同じ規則なので `src/utils/plotting` は `src/models` を何も import しない）。

この2つは**同じ乱数ストリームを同じ順序で消費する**ので、両図が同じニューロンと
同じエッジを示す。seed もサンプル数もモジュール定数 (`SAMPLE_SEED` / `N_SAMPLE`) なのは
そのため — 呼び出し側から片方だけ動かせると、この性質が黙って壊れる。

### area.py は SDF をグリッド評価して描く

`BaseArea` を `area.bounds` 上のグリッドで SDF 評価して描画する（領域は `contourf`、境界は
`contour(levels=[0])`）。`sdf()` が唯一の抽象メソッドである以上、これが唯一の形状非依存な
レンダラである — `patches.Circle`/`Rectangle` では `disk`/`rect` は描けても `CompositeArea` が
描けず、それが合成可能な領域を持つ意味そのもの。

area は duck-typed で受け取る（`sdf` / `bounds` / `is_bounded`、任意で `part_of` /
`part_names` / `area_um2`）。2つの図の分担: `area_figure()` は領域*だけ*を描き、
`network()` は体細胞/結合グラフの下に境界を重ね、軸範囲を `area.bounds` から取る
（これ以前は `space: area_uniform` に `x_range` が無いため autoscale に落ちていた）。
`network()` はエリアを `optional(view.area)` で取る — 無くても図は成立するので、座標と
違って必須にしない。

---

## run ディレクトリの規約

ファイル名の定数がどのモジュールにあるかは `src/core/README.md` の「ファイル別責務」。`src/utils/` はこれを import する。

`AXONS_NAME` (`axon_geometry.npz`) は **connector が軸索ジオメトリを持つときだけ**書かれ
（`axon_growth` 系）、ポリラインと各シナプスの接触点を保持する。どの軸索がどのブリッジを
横切ったかを問う損傷実験に必要な記録である。シリアライズは `AxonGeometry.save()` / `.load()`、
配置場所は呼び出し側の責任（`layout.save_axes()` と同じ）。`scripts/develop/run_one.py` が
実際の呼び出し例で、ジオメトリを duck-typed で触るので、それを持たない connector では no-op になる。

解析用に `NetworkLayout` を復元するのは各実験の `store/views.py::open_run()` の仕事:

```python
ConfigManager().load_resolved(run_dir / CONFIG_NAME)   # → AppConfig
NetworkLayout.from_config(...)                          # → 自動軸のみ
layout.load_axes_file(data_dir / AXES_NAME)             # → 外部軸
```

**area は記録に残らないので config から組み直す** (`Series.area()`)。`BaseArea` は RNG を
保持も消費もしないので、読み出しの途中で作っても
run の再現性には影響しない。モジュールの隣接関係 (`analysis/connectivity.bridge_hop_matrix()`)
を要求する図は、これを通して area を取る。
