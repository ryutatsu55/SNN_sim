# src/core — 設計と不変条件

`src/core/` を触る前に読む。プロジェクトの全体像はルート `README.md`。

---

## 1. Config → Pydantic → Registry パイプライン

3 者の役割と、コアを変えずにコンポーネントを足せる理由はルート `README.md` の
「アーキテクチャ — 4 つの不変条件」。追加の手順は `docs/architecture/adding_components.md`。

---

## 2. NetworkLayout: 正準グローバルID空間 + 任意軸

`layout.py` がニューロンのインデックス付けを所有する。単一の正準グローバルID空間 `0..N-1`
だけが参照系であり、他はすべてその上に貼られた**長さ N のラベル配列**である。

ニューロンの2つのビュー:

- **グローバルインデックス**: 結合行列・座標・解析で使う
- **population ローカルインデックス**: GeNN が population 内でニューロンを指す方法

`NetworkLayout` が両者を変換する（`split_global_to_local`, `merge_local_to_global`,
`local_to_global`）。simulator とデータローダはこれに委譲する。**不変条件:** 各 population の
`global_indices` は昇順である — GeNN はシナプスをソースニューロン順に格納し、
`_pair_coo()` がグローバル COO から切り出した並びがそのまま GeNN の並びになるのはこれに依存する。

### 軸 (Axes)

軸とは、各グローバルIDを値（カテゴリ名、またはソート用の数値）へ写す長さ N のラベル配列。
分類とソートは軸に対して汎用に書かれている。

```python
layout.ids_by("layer")                                # {値: グローバルID配列}
layout.ids_where(layer="IN2", polarity="excitatory")  # 多軸交差
layout.order_by("layer", "depth")                     # 並べ替え permutation (文字列軸と数値軸を混在可)
```

- **自動軸** `population` / `mode` / `polarity` は `from_config()` が構築する。これらは
  config だけから導出できる軸そのものであり、だから永続化されない。
- **外部軸** (`layer`, `module`, …) は `add_axis()` で外から押し込む。`NetworkLayout` 自身が
  データファイルや座標を読むことは決してない。

特権的な軸は存在しない: layer / module は連続性を要求されないただの軸であり、
スライスはすべて fancy-indexing (`np.ix_`) で行う。

### assignment

`config.layout.assignment` は population 所属をグローバルID空間へどう写すかを選ぶ:

- `sequential` — config の宣言順。population ごとに連続ブロックを占める。既存 run と
  比較可能に保ちたいときに使う。
- `random` (**既定**) — 全体比率に従って散布、`simulation.seed` から決定的。
  既定である理由は、ID順自体が意味を持つ場合に population と ID順を脱相関させるため。
  例: C. elegans ではグローバルID順が解剖学的な層順なので、`sequential` だと抑制性
  ニューロンが全部最後の層に固まってしまう。

これが変えるのはどのニューロンが E/I かであって、**コネクトーム自体ではない**。

`simulation.seed` と同様、`assignment` は **`ConfigManager` が材料化する**: 省略すると
`resolve()` が既定値を返り値の config に書き込むので、`config.yaml` は常に実際に使われた値を
記録する。したがって既定値は `config_manager.DEFAULT_ASSIGNMENT` にのみ存在する —
`NetworkLayout.from_config()` は推測せず raise し、`load_resolved()` は補完しない
（保存済み config は記録であり、`assignment` の欠落は検証エラー）。

`NetworkLayout` が持つのは ID空間・変換表・軸だけ。config の内容（ニューロンのパラメータ、
宣言された個数）は**持たない** — それらは config から直接読むこと。`__init__` は config を
受け取らず、`from_config()` だけが config に触れる。

---

## 3. 結合はすべて COO。密な (N, N) はホスト側に作らない

結合・重み・遅延のコンポーネントは最初から COO で値を返す: `BaseConnection.generate()` が
`(rows, cols)`、`BaseWeight.generate(rows, cols)` / `BaseDelay.generate(rows, cols)` が結合 1 本に
1 値の配列。`_generate_global_matrices()` がそれを単一の `GlobalCOO(row, col, weights, delays,
shape)` にまとめ、以降は GeNN 登録・`src/utils`・`scripts/`・テストのどこもこの形しか見ない。
GeNN への登録も常に `matrix_type="SPARSE"`。

- **不変条件: `GlobalCOO` は row-major ソート済み・重複なし**（row 昇順、row 内で col 昇順）。
  各 connector の `generate()` が保証する。`_pair_coo()` がこれに依存する: `global_indices` が
  昇順なので global→local の写像は単調で、GeNN が期待する順序（ソースニューロン順）を保つ。
- 全ペアに乱数を引く connector (`constant_prob` 系・`distance_based`・`gaussian_distance_type`) は
  `_row_block_coo()` で行ブロックごとに引く。(N, N) を一括で引いたのと同じ乱数が同じ (i, j) に
  対応するので、メモリは行ブロック分で済み、実現値は N² を一括で引く実装と変わらない。
- 「シナプスが存在する」の判定基準は**重みではなく COO に載っていること**。重み 0 のシナプスも
  シナプスである。
- **結合を密な (N, N) に戻す場所はただ1つ、`plotting/matrices.py::densify()`** で、`imshow` の
  直前である（C. elegans の connector / weight は配布データが 81×81 の CSV なので読み込みに
  行列を使うが、出口は COO）。行列の画像は本当に密行列を必要とするため。これは描画ローカルな関心事であり
  (`DENSE_RENDER_LIMIT` を自前で持つ)、交換形式ではない。

---

## 4. ファイル別責務

- **config_manager.py**: YAML をマージし、Pydantic で検証し、`AppConfig` ルートモデルを提供する。
  - `resolve()` は **`simulation.seed` を材料化する**: 欠落または `null` なら OS エントロピーから
    乱数シードを引き、返り値の config に書き込む。これで全 run が具体的で記録されたシードを持つ。
  - `save_resolved(config, save_dir)` は2ファイルを書く — `config.yaml`（解決済み、実シード。
    解析CLIが読むのはこちら）と `source_config.yaml`（入力 YAML の逐語コピー、`seed: null` と
    コメントを保持）。
  - `load_resolved()` はシードを発明しない。保存済み config は記録なので、そこの `null` は
    警告のみ。
  - `resolve()` は同様に **`simulation.backend` を材料化する**: `cuda` | `cpu`、または `auto`
    (`AUTO_BACKEND_NEURON_THRESHOLD` = 400 ニューロンで解決)。`auto` が検証済み `AppConfig` に
    残ることはない — GeNN のデバイス RNG はバックエンドごとに異なり、run は *(seed, backend)* の
    組でしか特定できないので、記録は常に具体的なバックエンド名を持つ。既定は `cuda`
    (`DEFAULT_BACKEND`)、これが置き換えた旧 `NetworkBuilder.USE_GPU = True` の挙動を保つため。
- **registry.py**: 全コンポーネント種別のグローバルレジストリ (NEURON_MODELS, SYNAPSE_MODELS, …)
- **layout.py**: `NetworkLayout` — 上記2節。config だけから `from_config()` で再構築可能、GeNN 不要。
- **NetworkBuilder.py**: config + 登録済みコンポーネントから GeNN モデルを作るファクトリ。
  モデルは double 精度で作る (時刻も double)
  - `_generate_global_matrices()`: 空間・結合・重み・遅延コンポーネントを呼び、
    **結果を単一の `GlobalCOO` にまとめる**
  - `global_coo() -> GlobalCOO`: builder の外からグローバル結合を読む**唯一の**方法
  - `_inject_axes()`: 各コンポーネントの `describe_axes()` を layout へ押し込む
  - `_build_neuron_populations()` / `_build_synapses()`: GeNN への登録
  - 生成後も `self.area` **と `self.connection`** を保持する。行列に到達しない副産物
    (`AxonGrowthTopology` の軸索ポリライン) を可視化から参照できるようにするため。
- **simulator.py**: GeNN モデルのラッパ。`push()`, `step()`, `pull()`, `reset()` を扱う。
  `pull_synapse_coo()` / `pull_synapse_flat()` が実際の読み手。`pull_synapse()`（密 N×N）は
  `pull_synapse_coo()` の薄い**ビュー**で、`DENSE_PULL_LIMIT_BYTES` で守られている。
- **run ディレクトリのファイル名**: 専用モジュールは持たず、**読み書きするモジュールが持つ**。
  `config_manager.py` が `CONFIG_NAME` / `SOURCE_CONFIG_NAME`、`layout.py` が `AXES_NAME`、
  `NetworkBuilder.py` が `GENN_CODE_DIR`。`data/` と記録ファイル名は実験の
  `scripts/<実験>/store/` にある。
