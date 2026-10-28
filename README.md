# SNN_sim

PyGeNN (GPU-enhanced neuronal networks) の上に構築した、モジュール式のスパイキングニューラルネットワーク (SNN)
シミュレータ。カスタムモデルである PQN (Piecewise Quadratic Neuron) の実装（浮動小数点/固定小数点）を含む。

- **PyGeNN Backend**: GeNN を利用した高速な SNN シミュレーション
- **Flexible Configuration**: YAML ベースの設定ファイルをモジュール化 (`components/`) し、`Pydantic` で型チェックとバリデーションを行う
- **Registry Pattern**: ニューロンモデル、シナプス、トポロジー、データローダーなどをレジストリで動的に登録・読み込みできる

**設定 (YAML) / 検証 (Pydantic) / コンポーネント発見 (Registry)** を分離しているので、
コアを変更せずに新しいモデルを追加できる。

この README はプロジェクトの仕様書で、使い方・構成・設計の不変条件をまとめる。
各層の詳細は `src/core/`・`src/models/network/`・`src/utils/`・`test/` のそれぞれの `README.md` にある。

---

## 初期設定

順番に remote SSH の設定、GitHub の設定、Python の環境構築、GeNN のインストール (環境構築時にインストールされる) が必要になる。
remote SSH については、黒木または先輩に直接聞いてください。セキュリティーの観点で GitHub には共有しない。
引継ぎ用に `docs/manual/github_setup.md`, `docs/manual/python_env_setup.md` を用意したのでそれを参照して初期設定してちょうだい。
GeNN については最新版の 5.4をforkし、独自に編集したものを使用している。環境構築のマニュアル通りにやればインストールされるはず。

---

## 基本的な使い方

```bash
python -m scripts.tools.pipeline_check   # configs/test.yaml でフルパイプライン実行
pytest test/ -q                     # 自動テスト全件
pytest test/ -q --ignore=test/test_axon_growth.py   # 所要時間の 9 割を占める 1 本を除く
python test/models/plasticity/STDPtest/test_multi_spike.py  # 手動テスト (GeNN ビルド必須)
python -m src.core.config_manager   # 解決済み config を Pydantic が見る形で表示（デバッグ用）
```

テストの件数・所要時間と、自動/手動の一覧は `test/README.md`。
実験の回し方は各実験の `scripts/<実験>/README.md`。

---

## 処理の流れ

`scripts/tools/pipeline_check.py` が動く実例:

1. `ConfigManager.resolve()` — 読み込み・検証・`seed`/`backend`/`assignment` の材料化
2. `NetworkBuilder.build()` — ニューロン確保 → グローバル結合生成 → GeNN 登録
3. `GeNNSimulator.setup()` — GeNN モデルのコンパイル、GPU メモリ確保
4. `DataLoader.generate()` — trial 入力とメタデータを yield
5. ループ: `push()` → `step()` → `pull()` → 次の trial のため `reset()`
6. 可視化: `src/utils/plotting` の図は `(view, out_path)` を取る。記録を持たない道具は
   builder をそのまま `BuiltNetwork.from_builder()` に載せれば、実験の構造図と同じ関数が
   呼べる (実験は build の記録を書いてから `series.built()` で読み直す)

---

## ディレクトリ構成

| 場所 | 中身 |
|---|---|
| `src/core/` | ConfigManager / Registry / NetworkLayout / NetworkBuilder / simulator |
| `src/models/neurons/` | `BaseNeuronModel` の実装。`@NEURON_MODELS.register()` |
| `src/models/synapses/` | `BaseSynapse` の実装。GeNN 互換パラメータ |
| `src/models/plasticity/` | `BasePlasticity` の実装（学習則） |
| `src/models/network/` | 領域・空間配置・結合マスク・重み分布・遅延分布 |
| `src/models/readouts/` | 出力層（Ridge 回帰デコーダ等） |
| `src/data/` | `BaseDataLoader` と具体ローダ (`spatial_loader.py`, `test_data.py`) |
| `src/utils/analysis/` | 純粋な計算。matplotlib を import しない |
| `src/utils/plotting/` | 汎用の描画のみ。**全関数が `(view, out_path)` を取る**（例外は `ax` を取るプリミティブと `model_test.py`）|
| `src/utils/runview.py` | 実験出力の**読み出し契約** (`RunView` / `Built` / `Window` / `Series`)。`src/utils/plotting/` はこれに対して書かれている |
| `configs/` | **実験に属さない** config のみ。道具が使うメイン config (`test.yaml`) と構造確認・ベンチ用プロファイル、および全体で共有する `components/*.yaml`。実験のメイン config は `scripts/<実験>/` にある |
| `scripts/` | 実行スクリプト。**1 実験 = 1 ディレクトリ**。どの実験も `report/`(いつ何を出すか) `store/`(読み書き) `analysis/`(何を測るか) `figures/`(どう描くか) の 4 層 + 3 つの入口 (`__main__.py` / `run_one.py` / `replot.py`) |
| `scripts/develop/` | 空間構造の上での発達実験 (軸索伸長 + モジュール領域)。手順は `scripts/develop/README.md` |
| `scripts/akita_soc/` | Ikeda-Akita-Takahashi (2023) Fig.2 の再現 (N=100, `no_space`)。手順は `scripts/akita_soc/README.md` |
| `scripts/lesion/` | 育った run を引き継いで結合を切り、回復を追う。手順は `scripts/lesion/README.md` |
| `scripts/tools/` | 実験に依存しない道具 (`visualize_network_structure.py` / `c_elegans.py` / `spike_animation.py` / `pipeline_check.py`)。実験から**一方向に** import してよい |
| `test/` | `test/models/{network,neurons,plasticity}/` が `src/models/` の対応する対象を鏡写しにする。一覧と自動/手動の区別は `test/README.md` |
| `docs/` | 設計の詳細 `docs/architecture/`・技術メモ `docs/technical/`・環境構築 `docs/manual/` |

`src/core/`, `src/models/network/`, `src/utils/`, `test/` にはそれぞれ `README.md` があり、
その層の構成と仕様を書いてある。

### 実験スクリプト

実験用スクリプトは `scripts/<実験名>/` に 1 実験 1 ディレクトリで置いてある
(`scripts/develop/` / `scripts/akita_soc/` / `scripts/lesion/`)。どの実験も同じ構成になっている:

| 置き場所 | 中身 |
|---|---|
| `__main__.py` | 入口。`python -m scripts.<実験名>` で run ディレクトリを作り、`run_one.py` を回す |
| `run_one.py` | 1 つの run を最後まで回す |
| `replot.py` | 記録から図と表を作り直す (シミュレーションは回さない) |
| `task.yaml` / `*.yaml` | 記録プロトコルと、その実験のメイン config |
| `report/` | いつ何を出すかの一覧 |
| `store/` | 記録の読み書き |
| `analysis/` | 何を測るか |
| `figures/` | 図の描き方 |

`scripts/lesion/` にはこのほか、run をまたいで比べる `compare.py` と、切断の手順をまとめた `protocol/` がある。

出力は `outputs/<実験名>/<条件>/` に出る (`scripts/lesion/` だけは親 run の下)。
引数・出力・回し方は各実験の `README.md` (`scripts/develop/README.md` など) を参照。
層の分け方の詳細は `docs/architecture/experiment_structure.md`。

**既存の実験とは別のことをしたいときは、既存の実験に足さず、それ用のディレクトリ `scripts/<新しい実験名>/` を新しく作る。**
既存の実験ディレクトリの構造 (`report/` `store/` `analysis/` `figures/` の 4 層と 3 つの入口) が活用できそうなら、
コピーして始めるとよい。

---

## アーキテクチャ — 4 つの不変条件

### 1. Config → Pydantic → Registry

「設定ファイル (YAML)」「型バリデーション (Pydantic)」「動的クラス生成 (Registry)」の 3 者が協調して動作することで、
モデルやその条件を独立に管理している。

1. **`configs/` (YAML)**: 実験パラメータや、使用するクラスの「名前（文字列）」を階層的に定義する。
   メイン設定 (実験のものは `scripts/<実験>/<名前>.yaml`、道具が使うものは `configs/*.yaml`) と
   モジュール別設定 (`components/*.yaml`) に分割して記述する。どの config がどこにあるかの一覧は
   `docs/architecture/configuration.md`。
2. **`src/core/config_manager.py` (Pydantic)**: 読み込まれた YAML を結合し、型チェックと値のバリデーションを行ってオブジェクト化する。
3. **`src/core/registry.py`**: YAML で指定された「文字列」を、実際の「Python クラス」に紐付ける。

基底クラスを継承して `@NEURON_MODELS.register("名前")` を付け、YAML からその名前を参照すれば、
NetworkBuilder も simulator も変更せずに拡張できる。

### 2. NetworkLayout — 正準グローバルID空間 + 任意軸

`src/core/layout.py` が唯一の参照系である `0..N-1` を所有し、それ以外は全部その上に貼られた
**長さ N のラベル配列（軸）**。`ids_by()` / `ids_where()` / `order_by()` が汎用API。
特権的な軸は無い。

```python
layout.ids_by("layer")
layout.ids_where(layer="IN2", polarity="excitatory")
layout.order_by("layer", "depth")
```

**不変条件:** 各 population の `global_indices` は昇順（GeNN がソースニューロン順に格納するため）。

### 3. コンポーネント階層

すべての拡張点はインタフェースを定める抽象基底クラスを持つ:
`BaseNeuronModel` / `BaseSynapse` / `BasePlasticity` / `BaseConnection` / `BaseWeight` /
`BaseDelay` / `BaseDataLoader` / `BaseSpace` / `BaseArea`。

構築順は **area → space → connection → weight → delay**。
`BaseSpace` / `BaseConnection` / `BaseWeight` / `BaseDelay` は
`describe_axes() -> {軸名: 長さN配列}` を返せ、NetworkBuilder がそれを layout に注入する。
`BaseArea` にはこのフックが無い（座標より前に構築されるため）。

### 4. 結合はすべて COO。密な (N, N) はホスト側に作らない

結合・重み・遅延のコンポーネントは最初から COO で値を返し、`_generate_global_matrices()` が
それを単一の `GlobalCOO(row, col, weights, delays, shape)` にまとめる。GeNN への登録も
常に `matrix_type="SPARSE"`。**不変条件: COO は row-major ソート済み。**

これは `NetworkBuilder.global_coo()` だけでなく **GeNN から取り出す側
(`simulator.pull_synapse_coo()`) も同じ**。GeNN はシナプスを SynapseGroup ごとに持つので
格納順は「集団ごとのブロック連結」だが、simulator は値を `global_coo()` の位置へ散布して
返す (row/col は `global_coo()` そのもの)。おかげで **COO の並びはプロジェクト全体で 1 つ**
になり、builder 由来と記録由来を位置で対応づけてよい。結合構造だけが要るときは
`global_coo()` を直接使う (実験は build の直後にこれを `connectivity.npz` へ書く)。

### Core モジュールの役割 (`src/core/`)

コアモジュールはシミュレータの心臓部。
**基本的にここは変更しないで済むように設計されてるはず**。詳細は `src/core/README.md`。

* **`config_manager.py`**: YAML の読み込み、コンポーネントの結合、Pydantic による構造化。
    ちなみにこれをメインファイルとして実行すると最終的に完成される config オブジェクトを確認できる
* **`registry.py`**: コンポーネント (Class) をグローバルに登録・取得する仕組み。
    ここに登録された Class のうち yaml で指定した名前で登録されている Class が動的に呼び出される
* **`NetworkBuilder.py`**: 設定と Class 部品からネットワークトポロジーを構築するファクトリ。
* **`simulator.py`**: PyGeNN へのロード、GPU 転送、ステップ実行を管理。

---

## 設定ファイル (YAML) の書き方

### メイン config の最小形

```yaml
simulation:
  dt: 0.1                  # [ms]
  seed: 42                 # null なら OS エントロピーから引いて config.yaml に記録される
  backend: cuda            # "cpu" | "cuda" | "auto"

layout:                    # 省略可。省略時は既定が記録される
  assignment: random       # "random" (既定) | "sequential"

neurons:
  Layer_Exc:
    type: akita_escape_lif # @NEURON_MODELS.register("akita_escape_lif")
    mode: excitatory       # 力学のモード (components/neurons.yaml のキー)
    polarity: excitatory   # E/I。mode とは独立
    num: 80

network:
  area: disk               # 必須。制約が無い場合は "no_space"
  connection: constant_prob
  weight: normal_broad
  space: ...
  delay: ...
```

パラメータは「Network 系（領域・空間・結合・重み・遅延）」と「Neuron/Synapse 系」で書き方が異なる。

### Network 系 (領域・空間・結合・重み・遅延)

メイン YAML では「使用するプロファイル名」のみを指定し、具体値はコンポーネント YAML に記述する。

**コンポーネント設定 (例: `configs/components/connections.yaml`)**
ここに書かれた設定値が connection クラス (`constant_prob`) の中で `self.config.p` のような形式でアクセスできる。
```yaml
constant_prob:
  p: 0.5
  allow_self_connections: false
```

### Neuron / Synapse 系

メイン YAML で「モデルクラス (`type`)」と「動作モード (`mode`)」を指定する。
そのモードのパラメータはコンポーネント側に記述する。

**コンポーネント設定 (例: `configs/components/neurons.yaml`)**
```yaml
akita_escape_lif:
  excitatory:
    tau_m: 30.0
    v_rest: -74.0
    v_th: -54.0
    ...
```

詳細 (メイン config の置き場所、プロファイルの上書き規則、`configs/` の一覧) → `docs/architecture/configuration.md`

---

## 新しいコンポーネントの追加

1. 対応する基底クラス (`BaseNeuronModel` / `BaseConnection` / ...) を継承したクラスを書く
2. クラス定義の直上に `@<カテゴリ>.register("YAMLで使用する名前")` を付ける
3. `configs/components/` にプロファイルを書き、メイン YAML からその名前を参照する

初期化時に渡される `self.config` は辞書ではなく **Pydantic モデルのインスタンス**なので、
YAML で定義したパラメータは `self.config.base_value` のように属性としてアクセスする。

各基底クラスが実装を求めるもの・軸の追加・area プロファイルの追加 → `docs/architecture/adding_components.md`
