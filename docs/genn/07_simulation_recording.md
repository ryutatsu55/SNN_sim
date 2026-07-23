# 07. シミュレーションの実行・変数アクセス・記録

[← 06 カスタムモデル](06_custom_models.md) ｜ [次: 08 高度な機能 →](08_advanced.md)

---

## 7.1 実行ループ

`build()` → `load()` のあと、`step_time()` をループで呼びます。

```python
model.build()
model.load()

while model.timestep < 100:
    model.step_time()
```

- `model.timestep`（int）: 現在の整数ステップ。setter で設定も可能。
- `model.t`（float, ms）: 現在時刻。
- GPU バックエンドでは、このループは各ステップのカーネルを**非同期発行**するだけで、
  CPU と同期しません（変数を pull する時などに同期）。

## 7.2 スパイク記録

スパイクは疎なので、GeNN は **専用の高効率記録機構** を持ちます。GPU 上に複数ステップ分の
イベントを溜めてから一括でホストへ転送します。

```python
# 1. build 前に記録を有効化
pop.spike_recording_enabled = True
# spike-like event を使うなら:
# pop.spike_event_recording_enabled = True

# 2. load 時に記録ステップ数を確保（必須）
model.build()
model.load(num_recording_timesteps=100)

# 3. 実行
while model.timestep < 100:
    model.step_time()

# 4. GPU→ホストへ転送して取得
model.pull_recording_buffers_from_device()
spike_times, spike_ids = pop.spike_recording_data[0]   # バッチ0
```

- `spike_recording_data[b]` は `(spike_times, spike_ids)` のタプル。
- シナプス群の pre/post イベントは
  `sg.pre_spike_event_recording_data[b]` / `sg.post_spike_event_recording_data[b]`。
- `num_recording_timesteps` は「溜められる最大ステップ数」。長時間シミュレーションでは
  区切りごとに pull してバッファを消化する設計にします。

## 7.3 変数アクセス（push / pull / view / values）

状態変数は各グループの `vars` 辞書（名前→変数オブジェクト）に入っています。
シナプス群はさらに `pre_vars` / `post_vars` / `psm_vars` を持ちます。
既定では変数は GPU とホストの両方に確保され、ホスト側から読めます
（`VarLocation.DEVICE` にすると Python から読めなくなります）。

### push / pull

```python
pop.vars["V"].pull_from_device()   # GPU → ホスト
pop.vars["V"].push_to_device()     # ホスト → GPU
```

> CPU バックエンドでは push/pull は何もしませんが、全バックエンドで透過的に動くよう
> **残しておくことが推奨**されています。

### values と view

```python
# 読み取り（コピー、ユーザ向けに整形済み）
v = pop.vars["V"].current_values           # 現在の遅延ステップ分・疎行列の並び替え済み
np.save("V.npy", pop.vars["V"].current_values)

# 全遅延ステップを見たい場合
all_v = pop.vars["V"].values

# 直接メモリビュー（書き込み用）
pop.vars["V"].current_view[:] = 1.0
pop.vars["V"].push_to_device()
```

- `current_values`: GeNN 所有データの**コピー**。疎行列は構築順に並べ替え、遅延付き変数は
  現在の遅延ステップ分を抽出する等の整形が入る。
- `current_view`: GeNN 所有メモリへの**直接ビュー**。値の設定に使い、その後 `push_to_device()`。

典型パターン:

```python
# 読む
pop.vars["V"].pull_from_device()
np.save("V.npy", pop.vars["V"].current_values)

# 書く
pop.vars["V"].current_view[:] = -65.0
pop.vars["V"].push_to_device()
```

## 7.4 Extra Global Parameters（EGP）

任意サイズの配列。**load 前に確保・初期化が必要**です。

```python
# 確保 + 初期値設定（load 前）
pop.extra_global_params["X"].set_init_values(np.zeros(100))
model.build()
model.load()

# load 後は変数同様にアクセス
pop.extra_global_params["X"].current_view[:] = 1.0
pop.extra_global_params["X"].push_to_device()
```

- ニューロン群: `pop.extra_global_params`
- シナプス群: 重み更新モデル用 `sg.extra_global_params`、後シナプスモデル用 `sg.psm_extra_global_params`
- 例: `SpikeSourceArray` の `spikeTimes`。

## 7.5 バッチ実行

```python
model.batch_size = 64
```

- パラメータ・疎結合はバッチ間で共有。状態変数は `VarAccess` 次第で複製/共有
  （→[06_custom_models.md](06_custom_models.md#63-変数アクセスvaraccess)）。
- バッチ依存データはインデックスで取得: `pop.spike_recording_data[b]`。

## 7.6 動的パラメータ（dynamic parameter）

パラメータは既定で読み取り専用ですが、`set_param_dynamic` で **実行時変更可能** にできます。

```python
# build 前
pop.set_param_dynamic("tau")

model.build()
model.load()

# 実行中に値を変更
tau = np.arange(0, 100, 10)
while model.timestep < 100:
    if (model.timestep % 10) == 0:
        pop.set_dynamic_param_value("tau", tau[model.timestep // 10])
    model.step_time()
```

> 注意: dynamic param を変えても、それに依存する **derived_params は更新されません**。
> 減衰率などを derived_params で事前計算している場合は注意が必要です。

## 7.7 性能プロファイリング

```python
model = GeNNModel("float", "profiled")
model.timing_enabled = True   # build 前に設定

# ... ポピュレーション追加 ...
model.build(); model.load()
for _ in range(1000):
    model.step_time()

print(f"neuron_update:      {model.neuron_update_time:.6f}s")
print(f"presynaptic_update: {model.presynaptic_update_time:.6f}s")
print(f"postsynaptic_update:{model.postsynaptic_update_time:.6f}s")
print(f"synapse_dynamics:   {model.synapse_dynamics_time:.6f}s")
print(f"init / init_sparse:  {model.init_time:.6f}s / {model.init_sparse_time:.6f}s")
```

各値は秒単位でモデル生存期間にわたり累積します。custom update の時間は
`model.get_custom_update_time(name)` 等で取得します。

> GPU では timing 有効化は同期オーバーヘッドを加えるため、性能計測時のみ有効にします。

---

[← 06 カスタムモデル](06_custom_models.md) ｜ [次: 08 高度な機能 →](08_advanced.md)
