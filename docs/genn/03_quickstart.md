# 03. クイックスタート

[← 02 インストール](02_installation.md) ｜ [次: 04 モデル構築 →](04_model_building.md)

---

## 3.1 GeNN の基本ワークフロー（12 ステップ）

GeNN のプログラムはほぼ常に同じ流れです。

1. `GeNNModel(precision, name[, backend=...])` でモデル生成
2. `model.dt = ...` でタイムステップ（ms）設定
3. （任意）`model.batch_size = N` でバッチ設定
4. `model.add_neuron_population(...)` でニューロン群を追加
5. `model.add_synapse_population(...)` でシナプス群を追加
6. （任意）`model.add_current_source(...)` で電流源を追加
7. （任意）`pop.spike_recording_enabled = True` でスパイク記録を有効化
8. （任意）EGP の初期値設定（`pop.extra_global_params["X"].set_init_values(...)`）
9. `model.build()` でコード生成 + コンパイル
10. `model.load(num_recording_timesteps=...)` でロード（記録を使うなら引数必須）
11. `while model.timestep < T: model.step_time()` で実行
12. `model.pull_recording_buffers_from_device()` / `var.pull_from_device()` で結果取得

## 3.2 最小の動く例（SpikeSourceArray → LIF）

入力スパイク列（`SpikeSourceArray`）を LIF ニューロンに `StaticPulse` シナプス＋指数電流
（`ExpCurr`）で接続し、出力スパイクを記録する完全な例です。GPU が無くても
`backend="single_threaded_cpu"` で動きます。

```python
import numpy as np
from pygenn import GeNNModel, init_weight_update, init_postsynaptic

# 1. モデル生成（GPU が無ければ CPU バックエンドを指定）
model = GeNNModel("float", "quickstart", backend="single_threaded_cpu")
model.dt = 1.0  # ms

# --- 入力: SpikeSourceArray（10 ニューロン、各々が決まった時刻に発火） ---
NUM_INPUT = 10
# 各入力ニューロン i を時刻 (i+1) ms に1回発火させる
spike_ids = np.arange(NUM_INPUT)
spike_times = (np.arange(NUM_INPUT) + 1).astype(np.float32)

# SpikeSourceArray は「ソート済みスパイク配列」と各ニューロンの開始/終了インデックスを使う
end_spike = np.cumsum(np.bincount(spike_ids, minlength=NUM_INPUT))
start_spike = np.concatenate(([0], end_spike[:-1]))
# ニューロンid→時刻 の順にソート
spike_times = spike_times[np.lexsort((spike_times, spike_ids))]

inp = model.add_neuron_population(
    "Input", NUM_INPUT, "SpikeSourceArray", {},
    {"startSpike": start_spike, "endSpike": end_spike})
inp.extra_global_params["spikeTimes"].set_init_values(spike_times)

# --- 出力: LIF ニューロン 5 個 ---
lif_params = {"C": 1.0, "TauM": 20.0, "Vrest": -65.0, "Vreset": -65.0,
              "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 2.0}
lif_init = {"V": -65.0, "RefracTime": 0.0}
out = model.add_neuron_population("Output", 5, "LIF", lif_params, lif_init)

# スパイク記録を有効化
out.spike_recording_enabled = True

# --- シナプス: 全結合(DENSE)、固定重み、指数電流 ---
model.add_synapse_population(
    "Input_Output", "DENSE",
    inp, out,
    init_weight_update("StaticPulse", {}, {"g": 1.0}),  # g は per-synapse 変数
    init_postsynaptic("ExpCurr", {"tau": 5.0}))

# 2. build → load
model.build()
T = 100
model.load(num_recording_timesteps=T)

# 3. 実行
while model.timestep < T:
    model.step_time()

# 4. 結果取得
model.pull_recording_buffers_from_device()
out_times, out_ids = out.spike_recording_data[0]  # [0] = バッチ0
print(f"出力スパイク数: {len(out_ids)}")
print("時刻:", out_times[:10])
print("ニューロンid:", out_ids[:10])
```

ポイント:

- `init_weight_update("StaticPulse", {}, {"g": 1.0})` の第2引数は **params**、第3引数は **vars**。
  `StaticPulse` は重み `g` を per-synapse 変数として持つ（→[05_builtin_models.md](05_builtin_models.md)）。
- `ExpCurr` は時定数 `tau` の指数減衰電流（params のみ）。
- 記録を使うので `load(num_recording_timesteps=T)` に総ステップ数を渡す。
- `spike_recording_data[b]` は `(spike_times, spike_ids)` のタプルを返す。

## 3.3 膜電位など変数の時系列を取りたい場合

スパイクではなく状態変数（例: LIF の `V`）を毎ステップ取得するには、`pull_from_device()` で
GPU→CPU 転送してから値を読みます（→[07_simulation_recording.md](07_simulation_recording.md)）。

```python
V_trace = np.empty((T, out.num_neurons))
while model.timestep < T:
    model.step_time()
    out.vars["V"].pull_from_device()
    V_trace[model.timestep - 1] = out.vars["V"].current_values
```

> 毎ステップ pull すると GPU と同期するため遅くなります。大規模・高速化が必要なら
> スパイク記録機構や、間引いた pull を検討してください。

## 3.4 次のステップ

- 各 API の詳細 → [04_model_building.md](04_model_building.md)
- 使える組み込みモデル → [05_builtin_models.md](05_builtin_models.md)
- 自作モデル → [06_custom_models.md](06_custom_models.md)

---

[← 02 インストール](02_installation.md) ｜ [次: 04 モデル構築 →](04_model_building.md)
