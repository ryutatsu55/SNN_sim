# 04. モデル構築 API

[← 03 クイックスタート](03_quickstart.md) ｜ [次: 05 組み込みモデル →](05_builtin_models.md)

---

ここでは `GeNNModel` とポピュレーション追加 API を詳しく説明します。
シグネチャは `pygenn/genn_model.py` 準拠です。

## 4.1 GeNNModel コンストラクタ

```python
GeNNModel(precision="float",
          model_name="GeNNModel",
          backend=None,
          time_precision=None,
          genn_log_level=PlogSeverity.WARNING,
          code_gen_log_level=PlogSeverity.WARNING,
          transpiler_log_level=PlogSeverity.WARNING,
          runtime_log_level=PlogSeverity.WARNING,
          backend_log_level=PlogSeverity.WARNING,
          **preference_kwargs)
```

| 引数 | 説明 |
|------|------|
| `precision` | `scalar` 型の精度。`"float"` か `"double"` |
| `model_name` | モデル名（生成コードのディレクトリ名に使用） |
| `backend` | `"cuda"` / `"hip"` / `"single_threaded_cpu"`。`None` で自動選択 |
| `time_precision` | 時刻 `t` の型。`None` なら `precision` と同じ |
| `*_log_level` | 各サブシステムのログレベル |
| `**preference_kwargs` | バックエンド固有設定（例: CUDA の `manual_device_id=0`） |

主要プロパティ:

- `model.dt`（float, ms）: タイムステップ。**ポピュレーション追加前に設定**するのが基本。
- `model.batch_size`（int）: バッチ数。
- `model.t`（float, ms）/ `model.timestep`（int）: 現在時刻 / 現在ステップ。
- `model.timing_enabled`（bool）: 性能計測の有効化（build 前に設定）。
- 内部辞書: `model.neuron_populations`, `model.synapse_populations`, `model.current_sources`,
  `model.custom_updates`, `model.custom_connectivity_updates`（名前→グループの dict）。

## 4.2 ニューロン群の追加: add_neuron_population

```python
add_neuron_population(pop_name, num_neurons, neuron, params={}, vars={}) -> NeuronGroup
```

| 引数 | 説明 |
|------|------|
| `pop_name` | 一意な名前 |
| `num_neurons` | ニューロン数 |
| `neuron` | 組み込みモデル名（文字列、例 `"LIF"`）または `create_neuron_model(...)` のインスタンス |
| `params` | パラメータ値の dict（population 内で均一な定数） |
| `vars` | 変数初期値の dict（定数 / numpy 配列 / `init_var(...)` スニペット） |

```python
lif_params = {"C": 1.0, "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0,
              "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 2.0}
pop = model.add_neuron_population("E", 800, "LIF", lif_params,
                                  {"V": -65.0, "RefracTime": 0.0})
```

変数初期化の3つの方法:

```python
{"V": -65.0}                  # 定数（バックエンドが埋める）
{"V": np.arange(800.0)}       # Python 配列をコピー
{"V": init_var("Uniform", {"min": -70.0, "max": -50.0})}  # 初期化スニペット
```

## 4.3 シナプス群の追加: add_synapse_population

```python
add_synapse_population(pop_name, matrix_type, source, target,
                       weight_update_init, postsynaptic_init,
                       connectivity_init=None) -> SynapseGroup
```

シナプス群は **重み更新モデル(weight update)** と **後シナプスモデル(postsynaptic)** の2つで
挙動が決まります。前者は「シナプスでどんな計算が起き、前後ニューロンへ何を出力するか」、
後者は「シナプス入力をどう入力電流（等）へ変換するか」を定義します。

```python
sg = model.add_synapse_population(
    "EE", "SPARSE",
    src_pop, tgt_pop,
    init_weight_update("StaticPulseConstantWeight", {"g": 1.0}),
    init_postsynaptic("ExpCurr", {"tau": 5.0}),
    init_sparse_connectivity("FixedProbability", {"prob": 0.1}))
```

### matrix_type（結合の格納形式）

`SynapseMatrixType`（文字列でも指定可）:

| 値 | 意味 |
|----|------|
| `DENSE` | 密行列（全 pre×post）。重み変数を初期化スニペットで GPU 初期化可 |
| `DENSE_PROCEDURALG` | 密だが重みを手続き的に生成 |
| `SPARSE` | 疎（ragged 形式）。`init_sparse_connectivity` で初期化／`set_sparse_connections` で明示指定 |
| `BITMASK` | ビットマスク疎形式 |
| `PROCEDURAL` | 結合をその場で手続き生成（メモリ最小） |
| `PROCEDURAL_KERNELG` | 手続き結合 + カーネル重み |
| `TOEPLITZ` | 畳み込み的な構造化結合（`init_toeplitz_connectivity`） |

- `DENSE` / `DENSE_PROCEDURAL` は重み初期化スニペットだけで GPU 初期化できる。
- `SPARSE` / `BITMASK` / `PROCEDURAL` の結合は `init_sparse_connectivity(...)` で生成。
- `TOEPLITZ` は `init_toeplitz_connectivity(...)`。
- `connectivity_init` を省略すると「未初期化の疎結合」になる（あとで `set_sparse_connections` で埋める用途）。

### init_weight_update / init_postsynaptic / init_sparse_connectivity

```python
init_weight_update(snippet, params={}, vars={}, pre_vars={}, post_vars={},
                   pre_var_refs={}, post_var_refs={}, psm_var_refs={})

init_postsynaptic(snippet, params={}, vars={}, var_refs={})

init_sparse_connectivity(snippet, params={})
init_toeplitz_connectivity(snippet, params={})
init_var(snippet, params={})
```

- `snippet` は組み込みモデル名（文字列）か `create_*` インスタンス。
- 重み更新モデルは per-synapse の `vars`、pre/post ニューロン単位の `pre_vars`/`post_vars`、
  および各種 `*_var_refs`（他ポピュレーション変数への参照）を持てる（→[08_advanced.md](08_advanced.md)）。

### 疎結合を明示的に与える（SNN_sim 方式）

`init_sparse_connectivity` で確率的に生成するのではなく、自前で計算した pre/post インデックス対を
渡すこともできます。SNN_sim はこの方式です。

```python
sg = model.add_synapse_population("A_to_B", "SPARSE", src, tgt,
        init_weight_update(...), init_postsynaptic(...))   # connectivity_init は省略
sg.set_sparse_connections(pre_indices, post_indices)        # 明示的に結合を設定
sg.max_dendritic_delay_timesteps = max_delay_steps          # 樹状突起遅延の最大値
```

## 4.4 電流源の追加: add_current_source

```python
add_current_source(cs_name, current_source_model, pop,
                   params={}, vars={}, var_refs={}) -> CurrentSource
```

ニューロン群へ外部電流を注入します。

```python
model.add_current_source("Noise", "GaussianNoise", pop,
                         {"mean": 0.0, "sd": 1.0})
```

- `current_source_model`: 組み込み名（`"DC"`, `"GaussianNoise"`, `"PoissonExp"`）か
  `create_current_source_model(...)`。
- `var_refs`: ターゲットニューロンの変数への参照（電圧依存の注入など）。

## 4.5 build と load

```python
build(path_to_model="./", always_rebuild=False, never_rebuild=False)
load(num_recording_timesteps=None)
```

- `build()`: コード生成（lazy）＋コンパイル。`always_rebuild=True` で強制再生成、
  `never_rebuild=True` で再生成を抑止（並列実行で生成物の競合を避ける用途）。
- `load()`: GPU/CPU へロード。**スパイク記録を使う場合は `num_recording_timesteps` が必須**。
- `model.unload()` でロードを解除しメモリ解放。

## 4.6 カスタム更新・カスタム結合更新（概要）

毎ステップではなく任意タイミングで GPU 上の処理を走らせる仕組み。詳細は
[08_advanced.md](08_advanced.md)。

```python
add_custom_update(cu_name, group_name, custom_update_model,
                  params={}, vars={}, var_refs={}, egp_refs={})
add_custom_connectivity_update(cu_name, group_name, syn_group,
                  custom_conn_update_model, ...)
# 実行（同名 group_name の更新がまとめて走る）
model.custom_update(group_name)
```

---

[← 03 クイックスタート](03_quickstart.md) ｜ [次: 05 組み込みモデル →](05_builtin_models.md)
