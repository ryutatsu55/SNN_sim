# 09. API リファレンス（早見表）

[← 08 高度な機能](08_advanced.md) ｜ [次: 10 SNN_sim 連携 →](10_snn_sim_integration.md)

---

シグネチャは `pygenn/genn_model.py` / `genn_groups.py` / `__init__.py`（GeNN 5.4.0）準拠。

## 9.1 GeNNModel メソッド

```python
GeNNModel(precision="float", model_name="GeNNModel", backend=None,
          time_precision=None, **preference_kwargs)

add_neuron_population(pop_name, num_neurons, neuron, params={}, vars={}) -> NeuronGroup
add_synapse_population(pop_name, matrix_type, source, target,
                       weight_update_init, postsynaptic_init,
                       connectivity_init=None) -> SynapseGroup
add_current_source(cs_name, current_source_model, pop,
                   params={}, vars={}, var_refs={}) -> CurrentSource
add_custom_update(cu_name, group_name, custom_update_model,
                  params={}, vars={}, var_refs={}, egp_refs={}) -> CustomUpdate
add_custom_connectivity_update(cu_name, group_name, syn_group,
                  custom_conn_update_model, params={}, vars={},
                  pre_vars={}, post_vars={}, var_refs={},
                  pre_var_refs={}, post_var_refs={}, egp_refs={})

build(path_to_model="./", always_rebuild=False, never_rebuild=False)
load(num_recording_timesteps=None)
step_time()
custom_update(name)
unload()
pull_recording_buffers_from_device()
```

主なプロパティ: `dt`, `t`, `timestep`, `batch_size`, `precision`, `time_precision`, `name`,
`timing_enabled`, `neuron_update_time`, `presynaptic_update_time`, `postsynaptic_update_time`,
`synapse_dynamics_time`, `init_time`, `init_sparse_time`。
辞書: `neuron_populations`, `synapse_populations`, `current_sources`, `custom_updates`,
`custom_connectivity_updates`。

## 9.2 モデル定義ファクトリ（create_*）

```python
create_neuron_model(class_name, params=None, vars=None, derived_params=None,
    sim_code=None, threshold_condition_code=None, reset_code=None,
    extra_global_params=None, additional_input_vars=None,
    auto_refractory_required=False)

create_weight_update_model(class_name, params=None, vars=None,
    pre_vars=None, post_vars=None,
    pre_neuron_var_refs=None, post_neuron_var_refs=None, psm_var_refs=None,
    derived_params=None,
    pre_spike_syn_code=None, post_spike_syn_code=None,
    pre_event_syn_code=None, post_event_syn_code=None, synapse_dynamics_code=None,
    pre_event_threshold_condition_code=None, post_event_threshold_condition_code=None,
    pre_spike_code=None, post_spike_code=None,
    pre_dynamics_code=None, post_dynamics_code=None, extra_global_params=None)

create_postsynaptic_model(class_name, params=None, vars=None,
    neuron_var_refs=None, derived_params=None, sim_code=None, extra_global_params=None)

create_current_source_model(class_name, params=None, vars=None,
    neuron_var_refs=None, derived_params=None, injection_code=None,
    extra_global_params=None)

create_custom_update_model(class_name, params=None, vars=None, derived_params=None,
    var_refs=None, update_code=None, extra_global_params=None,
    extra_global_param_refs=None)

create_custom_connectivity_update_model(class_name, params=None, vars=None,
    pre_vars=None, post_vars=None, derived_params=None,
    var_refs=None, pre_var_refs=None, post_var_refs=None,
    row_update_code=None, host_update_code=None,
    extra_global_params=None, extra_global_param_refs=None)

create_var_init_snippet(class_name, params=None, derived_params=None,
    var_init_code=None, extra_global_params=None)

create_sparse_connect_init_snippet(class_name, params=None, derived_params=None,
    row_build_code=None, col_build_code=None,
    calc_max_row_len_func=None, calc_max_col_len_func=None,
    calc_kernel_size_func=None, extra_global_params=None)

create_toeplitz_connect_init_snippet(class_name, params=None, derived_params=None,
    diagonal_build_code=None, calc_max_row_len_func=None,
    calc_kernel_size_func=None, extra_global_params=None)
```

## 9.3 初期化ヘルパ（init_*）

```python
init_var(snippet, params={})
init_sparse_connectivity(snippet, params={})
init_toeplitz_connectivity(snippet, params={})
init_postsynaptic(snippet, params={}, vars={}, var_refs={})
init_weight_update(snippet, params={}, vars={}, pre_vars={}, post_vars={},
                   pre_var_refs={}, post_var_refs={}, psm_var_refs={})
```

`init_weight_update` の戻り値は `(WeightUpdateInit, vars, pre_vars, post_vars)` のタプル、
`init_postsynaptic` は `(PostsynapticInit, vars)` のタプルで、`add_synapse_population` に渡します。

## 9.4 変数参照・EGP 参照ヘルパ

```python
create_var_ref(pop, var_name)
create_psm_var_ref(sg, var_name)
create_wu_var_ref(source_sg, var_name[, target_sg, target_var_name])
create_wu_pre_var_ref(sg, var_name)
create_wu_post_var_ref(sg, var_name)
create_spike_time_var_ref(pop)
create_prev_spike_time_var_ref(pop)
create_out_post_var_ref(sg)
create_den_delay_var_ref(sg)
create_src_spike_count_var_ref(sg)
create_egp_ref(pop, egp_name)
create_psm_egp_ref(sg, egp_name)
create_wu_egp_ref(sg, egp_name)
```

## 9.5 主要な列挙型

### VarLocation（変数の確保場所）

| 値 | 意味 |
|----|------|
| `HOST_DEVICE` | ホストとデバイス両方（**既定**） |
| `DEVICE` | デバイスのみ（ホストメモリ節約。Python から不可視） |
| `HOST_DEVICE_ZERO_COPY` | zero-copy 共有メモリ（Jetson 等の組み込み向け） |

### VarAccess（neuron/wu/cs/ccu モデル変数）

`READ_WRITE`, `READ_ONLY`, `READ_ONLY_DUPLICATE`, `READ_ONLY_SHARED_NEURON`,
`REDUCE_BATCH_SUM`, `REDUCE_BATCH_MAX`, `REDUCE_NEURON_SUM`, `REDUCE_NEURON_MAX`

### VarAccessMode（変数参照のアクセス）

`READ_ONLY`, `READ_WRITE`, `BROADCAST`, `REDUCE_SUM`, `REDUCE_MAX`

### CustomUpdateVarAccess（custom update モデル変数）

`READ_ONLY`, `READ_WRITE`, `REDUCE_BATCH_SUM`, `REDUCE_BATCH_MAX`,
`REDUCE_NEURON_SUM`, `REDUCE_NEURON_MAX`

### SynapseMatrixType

`DENSE`, `DENSE_PROCEDURALG`, `SPARSE`, `BITMASK`, `PROCEDURAL`, `PROCEDURAL_KERNELG`, `TOEPLITZ`

### ParallelismHint

`PRESYNAPTIC`, `WORD_PACKED_BITMASK`（ほか）

## 9.6 グループクラスの主なプロパティ／メソッド

### NeuronGroup
- `vars`, `extra_global_params`
- `spike_recording_enabled`, `spike_event_recording_enabled`
- `spike_recording_data[b]`
- `num_neurons`
- `set_param_dynamic(name)`, `set_dynamic_param_value(name, value)`

### SynapseGroup
- `src`, `trg`
- `vars`（per-synapse）, `pre_vars`, `post_vars`, `psm_vars`
- `extra_global_params`, `psm_extra_global_params`
- `axonal_delay_steps`, `back_prop_delay_steps`, `max_dendritic_delay_timesteps`
- `set_sparse_connections(pre_inds, post_inds)`,
  `get_sparse_pre_inds()`, `get_sparse_post_inds()`
- `pull_connectivity_from_device()`, `push_connectivity_to_device()`
- `pre_spike_event_recording_data[b]`, `post_spike_event_recording_data[b]`
- `out_post`

### CurrentSource
- `vars`, `extra_global_params`
- `set_param_dynamic(name)`, `set_dynamic_param_value(name, value)`

### 変数オブジェクト（vars["X"] 等）
- `pull_from_device()`, `push_to_device()`
- `current_values`（整形済みコピー）, `values`（全遅延ステップ）, `current_view`（直接ビュー）
- EGP は `set_init_values(arr)`（load 前）も持つ

---

[← 08 高度な機能](08_advanced.md) ｜ [次: 10 SNN_sim 連携 →](10_snn_sim_integration.md)
