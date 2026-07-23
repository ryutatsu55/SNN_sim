# 10. SNN_sim での GeNN 連携

[← 09 API リファレンス](09_api_reference.md) ｜ [README に戻る](README.md)

---

この章は SNN_sim プロジェクトが GeNN/pygenn を**どう使っているか**を、実コードに基づいて解説します。
パスはすべて `kuroki/SNN_sim/` 起点です。

## 10.1 全体アーキテクチャ

SNN_sim は「**設定（YAML）駆動 + レジストリによる動的モデル選択**」の上に GeNN を載せた構成です。

```
configs/*.yaml ──▶ ConfigManager(Pydantic検証) ──▶ AppConfig
                                                      │
                              ┌───────────────────────┘
                              ▼
   NetworkBuilder ── pygenn.GeNNModel を組み立て（neuron/synapse/current source）
                              │  Registry から各コンポーネント実装を名前で取得
                              ▼
   GeNNSimulator ── build/load/step/push/pull/reset でシミュレーション実行
                              ▼
   results/spikes ──▶ src/utils/visualize で可視化、outputs/ へ保存
```

主なディレクトリ:

| パス | 役割 |
|------|------|
| [src/core/](../../src/core/) | パイプライン中核（Config・Registry・NetworkBuilder・Simulator・出力管理） |
| [src/models/](../../src/models/) | ニューロン/シナプス/可塑性/ネットワーク(空間・結合・重み・遅延)/readout の実装 |
| [src/data/](../../src/data/) | 試行データ生成（spatial / audio / test） |
| [src/utils/](../../src/utils/) | 評価・可視化 |
| [configs/](../../configs/) | YAML 設定（メイン + components/） |
| [scripts/](../../scripts/) | 実行スクリプト（test.py など） |

## 10.2 GeNN モデルの生成（NetworkBuilder）

[src/core/NetworkBuilder.py](../../src/core/NetworkBuilder.py) がすべての pygenn 呼び出しを担います。

```python
# NetworkBuilder.__init__
self.genn_model = pygenn.GeNNModel("double", model_name, time_precision="double")
self.genn_model.dt = self.config.simulation.dt
```

- 精度は **double**、`time_precision="double"`（樹状突起遅延などの時間精度のため）。
- バックエンドは未指定（=自動選択）。**GPU が無い環境で動かすには**、ここに
  `backend="single_threaded_cpu"` を渡す改修が必要です（→[01_overview.md](01_overview.md#14-バックエンド)）。

### ニューロン群の登録

[`_build_neuron_populations`](../../src/core/NetworkBuilder.py) は、Registry から取得した
ニューロンクラスのインスタンスを `add_neuron_population` に渡します。

```python
NeuronClass = NEURON_MODELS.get(params.type)        # 例: "LIF", "PQN_float", "akita_escape_lif"
neuron_instance = NeuronClass(params, self.config.simulation.dt)
self.genn_model.add_neuron_population(
    pop_name=group_name,
    num_neurons=num_neurons,
    neuron=neuron_instance.model_class,   # pygenn のモデル定義オブジェクト
    params=neuron_instance.params,        # 定数 dict
    vars=neuron_instance.vars)            # 初期値 dict
```

### シナプス群の登録（疎結合 + 樹状突起遅延）

[`_build_synapses`](../../src/core/NetworkBuilder.py) は可塑性（重み更新）モデルと
シナプス（後シナプス）モデルを Registry から取得し、`init_weight_update` / `init_postsynaptic`
で初期化してから `add_synapse_population` します。**結合は確率生成ではなく、自前で計算した
インデックス対**を `set_sparse_connections` で与えます。

```python
weight_init = pygenn.genn_model.init_weight_update(
    snippet=plas_instance.snippet, params=plas_instance.params,
    vars=plas_instance.vars, pre_vars=plas_instance.pre_vars,
    post_vars=plas_instance.post_vars,
    pre_var_refs=plas_instance.pre_var_refs,
    post_var_refs=plas_instance.post_var_refs,
    psm_var_refs=plas_instance.psm_var_refs)

post_init = pygenn.genn_model.init_postsynaptic(
    snippet=syn_instance.snippet, params=syn_instance.params,
    vars=syn_instance.vars, var_refs=syn_instance.var_refs)

sg = self.genn_model.add_synapse_population(
    pop_name=f"{src_name}_to_{tgt_name}", matrix_type="SPARSE",
    source=src_pop, target=tgt_pop,
    weight_update_init=weight_init, postsynaptic_init=post_init)

sg.set_sparse_connections(local_src_idx, local_tgt_idx)           # 明示的結合
max_delay_steps = int(np.max(delays_flat)) + 1
sg.max_dendritic_delay_timesteps = max_delay_steps               # 樹状突起遅延の最大値
```

> 遅延は `custom_Akita` 等の重み更新変数 `d` として保持され、`addToPostDelay(w, d)` で適用されます
> （→[08_advanced.md](08_advanced.md#81-シナプス遅延)）。

### 電流源・スパイク記録

```python
# _build_input_ports: GaussianNoise を組み込み電流源として各群に付与
cs = self.genn_model.add_current_source(
    cs_name=cs_name, current_source_model="GaussianNoise",
    pop=self.genn_model.neuron_populations[pop_name],
    params={"mean": ..., "sd": ...}, vars={})

# _build_output_ports: スパイク記録を有効化
pop.spike_recording_enabled = rec_spike
```

### `_component_lifeline` の役割（重要）

`NetworkBuilder` は生成したモデルインスタンスを `self._component_lifeline` に保持します。
これは GeNN のコード生成が終わるまで Python 側のオブジェクトが **GC されないようにする**ためで、
カスタムモデルを動的生成する設計では必須のパターンです。

## 10.3 レジストリと Base クラス抽象

[src/core/registry.py](../../src/core/registry.py) はデコレータ方式の登録機構です。

```python
@NEURON_MODELS.register("LIF")
class LIF(BaseNeuronModel):
    @property
    def model_class(self):  return pygenn.genn_model.neuron_models.LIF()
    @property
    def params(self):       return {"C": ..., "TauM": ..., ...}
    @property
    def vars(self):         return {"V": ..., "RefracTime": 0.0}
```

各レジストリ: `NEURON_MODELS`, `SYNAPSE_MODELS`, `PLASTICITY_MODELS`, `SPATIAL_MODELS`,
`CONNECTION_MODELS`, `WEIGHT_MODELS`, `DELAY_MODELS`, `DATA_LOADERS`。

Base クラス（[BASE_neuron.py](../../src/models/neurons/BASE_neuron.py) ほか）が
`model_class` / `params` / `vars` などの統一インターフェースを定義し、GeNN へ渡す形を抽象化します。
これにより YAML には**文字列名**だけを書き、実装クラスはデコレータで自動的に解決されます。

## 10.4 設定駆動（YAML + Pydantic）

[configs/test.yaml](../../configs/test.yaml) の例:

```yaml
simulation:
  N: 150
  dt: 0.1            # ms
  seed: null
neurons:
  Layer_Exc: { type: akita_escape_lif, mode: excitatory, num: 80 }
  Layer_Inh: { type: akita_escape_lif, mode: inhibitory, num: 20 }
synapses:
  from_Exc:
    source: Layer_Exc
    plasticity: { type: custom_Akita, mode: e-stdp }   # PLASTICITY_MODELS の名前
    synapse:    { type: ExpCond,      mode: excitatory } # SYNAPSE_MODELS の名前
network:
  space: block_2d            # SPATIAL_MODELS
  connection: distance_based # CONNECTION_MODELS
  weight: lognormal_broad    # WEIGHT_MODELS
  delay: distance_based      # DELAY_MODELS
```

- `type` フィールドの文字列が、そのまま各 Registry のキーになります。
- 詳細な数値パラメータは [configs/components/](../../configs/components/) 配下の
  `neurons.yaml` / `synapses.yaml` / `weights.yaml` 等に分離され、`mode` で選択されます。
- [src/core/config_manager.py](../../src/core/config_manager.py) が Pydantic でスキーマ検証し、
  `AppConfig` を返します（タイプミスや欠落を早期に検出）。

## 10.5 シミュレーション実行（GeNNSimulator）

[src/core/simulator.py](../../src/core/simulator.py) が実行を担当します。

```python
def setup(self):
    self.model.build()                                   # コード生成 + コンパイル
    self.model.load(num_recording_timesteps=self.max_timesteps)

def push(self, global_data, target_var="Iext"):          # 入力注入（CPU→GPU）
    for pop_name, data in self._split_global_to_local(global_data).items():
        pop = self.model.neuron_populations[pop_name]
        pop.vars[target_var].view[:] = data
        pop.vars[target_var].push_to_device()

def step(self, duration_steps=1):
    for _ in range(duration_steps):
        self.model.step_time()

def pull(self, var_name):                                # 状態取得（GPU→CPU）
    out = {}
    for pop_name in self.model.neuron_populations:
        pop = self.model.neuron_populations[pop_name]
        pop.vars[var_name].pull_from_device()
        out[pop_name] = np.copy(pop.vars[var_name].view)
    return self._merge_local_to_global(out)

def get_global_spikes(self):
    self.model.pull_recording_buffers_from_device()
    times, local_ids = self.model.neuron_populations[pop_name].spike_recording_data[0]
    ...
```

ポイント:

- SNN_sim は変数アクセスに `pop.vars[name].view` を使います（汎用ドキュメントの `current_view` に相当）。
- `_split_global_to_local` / `_merge_local_to_global` で「グローバル N ニューロン配列」と
  「ポピュレーションローカル配列」を相互変換します（global↔local インデックス対応は
  `group_info` が保持）。
- `reset()` は各群・各シナプスの変数を numpy 操作で初期状態へ戻し `push_to_device()`、
  試行ベースの実験（複数 trial）に対応します。

## 10.6 プロジェクト独自モデル

| 種別 | 名前 | 実装 | 概要 |
|------|------|------|------|
| ニューロン | `LIF` | [lif.py](../../src/models/neurons/lif.py) | GeNN 組み込み LIF のラッパ |
| ニューロン | `PQN_float` | [pqn_float.py](../../src/models/neurons/pqn_float.py) | 区分二次ニューロン（浮動小数、`create_neuron_model` で C コード生成） |
| ニューロン | `PQN_int` | [pqn_int.py](../../src/models/neurons/pqn_int.py) | 固定小数点版 PQN（int64 演算で量子化誤差回避） |
| ニューロン | `akita_escape_lif` | [akita_escape_lif.py](../../src/models/neurons/akita_escape_lif.py) | 逃避課題向けカスタム LIF |
| シナプス | `ExpCurr` / `ExpCond` | [standard_models.py](../../src/models/synapses/standard_models.py) | GeNN 組み込み後シナプスのラッパ（`ExpCond` は `create_var_ref(pop,"V")` を使用） |
| 可塑性 | `custom_Akita` | [custom_Akita.py](../../src/models/plasticity/custom_Akita.py) | STDP + 短期可塑性のカスタム重み更新（`pre_vars`/`post_vars` でトレース保持、`d` で遅延） |
| 可塑性 | `static` | [standard_models.py](../../src/models/plasticity/standard_models.py) | 学習なしの固定重み |

ネットワークトポロジ（[src/models/network/](../../src/models/network/)）も Registry 化されています:
空間 `space.py`（no_space/grid_2d/random_2d/block_2d）、結合 `connectors.py`
（constant_prob/distance_based/prob_based_block）、重み `weights.py`
（constant/normal_broad/lognormal_broad/distance_dependent）、遅延 `delays.py`
（constant/distance_based）。これらは GeNN に渡す前のグローバル行列（mask/weight/delay/coords）を生成し、
`NetworkBuilder` がそれを疎インデックス＋per-synapse 変数へ変換します。

## 10.7 出力と可視化

`scripts/test.py` のメインフロー:

```python
manager = ConfigManager()
config = manager.resolve(config_src, TASK_NAME)
builder = NetworkBuilder(config)
genn_model, group_info = builder.build(rec_spike=True)
sim = GeNNSimulator(genn_model, config, builder); sim.setup()

for trial_inputs, meta in data_loader.generate():
    for inputs, duration_steps in trial_inputs:
        sim.push(inputs, target_var="Iext")
        for _ in range(duration_steps):
            sim.step()
            results[step, :] = sim.pull("V")
    trial_results = sim.get_global_spikes()
    sim.reset()

manager.save_resolved(config, save_dir=output_dir)
visualize.neuron_test(results, I_in, trial_results["times"], trial_results["ids"], config, save_path=output_dir)
visualize.network(weights=builder.global_weights, coords=builder.global_coords, config=config, save_path=output_dir)
```

- 出力先は [src/core/output_manager.py](../../src/core/output_manager.py) の
  `create_run_output_dir(TASK_NAME)` がタイムスタンプ付きで作成（`outputs/<task>/<timestamp>/`）。
- 可視化は [src/utils/visualize/](../../src/utils/visualize/)（ラスター、ネットワーク図、重み追跡など）。

## 10.8 GeNN 汎用ドキュメントとの対応表

| やりたいこと | 汎用ドキュメント | SNN_sim での実装箇所 |
|--------------|------------------|----------------------|
| モデル生成 | [04](04_model_building.md#41-gennmodel-コンストラクタ) | `NetworkBuilder.__init__` |
| ニューロン追加 | [04](04_model_building.md#42-ニューロン群の追加-add_neuron_population) | `_build_neuron_populations` |
| シナプス追加 | [04](04_model_building.md#43-シナプス群の追加-add_synapse_population) | `_build_synapses` |
| カスタムモデル | [06](06_custom_models.md) | `pqn_*.py`, `custom_Akita.py` |
| 遅延 | [08](08_advanced.md#81-シナプス遅延) | `max_dendritic_delay_timesteps` + `d` |
| 実行・push/pull | [07](07_simulation_recording.md) | `GeNNSimulator.push/step/pull` |
| スパイク記録 | [07](07_simulation_recording.md#72-スパイク記録) | `spike_recording_enabled`, `get_global_spikes` |

---

[← 09 API リファレンス](09_api_reference.md) ｜ [README に戻る](README.md)
