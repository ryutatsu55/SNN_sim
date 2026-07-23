# 08. 高度な機能（遅延・カスタム更新・変数参照）

[← 07 シミュレーション・記録](07_simulation_recording.md) ｜ [次: 09 API リファレンス →](09_api_reference.md)

---

## 8.1 シナプス遅延

GeNN には 3 種類の遅延があります。**どのコードが遅れるか** が異なるので要注意です。

| 遅延 | 設定 | 効果 |
|------|------|------|
| 軸索遅延 | `sg.axonal_delay_steps = n` | プレスパイクがシナプスに届くのを n ステップ遅延 → `pre_spike_syn_code` が n ステップ後に実行 |
| 逆伝播遅延 | `sg.back_prop_delay_steps = n` | ポストスパイクの到達を n ステップ遅延 → `post_spike_syn_code` が n ステップ後に実行 |
| 樹状突起遅延 | `addToPostDelay(inc, d)` + `sg.max_dendritic_delay_timesteps` | 後ニューロンへの**電流付与だけ**をシナプスごと（異種可）に遅延 |

```python
sg.axonal_delay_steps = 5            # 全シナプス一律の軸索遅延（5ステップ）
sg.back_prop_delay_steps = 3         # 逆伝播遅延

# 異種の樹状突起遅延（StaticPulseDendriticDelay 等）
# 重み更新コード内: addToPostDelay(g, d);   d は per-synapse 変数（uint8_t 等）
sg.max_dendritic_delay_timesteps = 16   # d が取りうる最大値+1
```

- 樹状突起遅延は「電流の到達」だけを遅らせ、コードの実行自体は遅れません。
  異種（シナプスごとに異なる）遅延が必要なときに使います。
- 重み更新コードでは `postVar[delay]` で後シナプス変数/参照に遅延アクセスもできます。
- 樹状突起遅延を使うと後シナプス出力は `max_dendritic_delay_timesteps` ステップ分バッファされ、
  custom update でリセットする場合は `create_den_delay_var_ref(sg)` を使います。

> **SNN_sim では**: `NetworkBuilder` が `sg.set_sparse_connections(...)` の後に
> `sg.max_dendritic_delay_timesteps = max_delay_steps` を設定し、`custom_Akita` 等の重み更新で
> 遅延変数 `d` を `addToPostDelay` に使います（→[10_snn_sim_integration.md](10_snn_sim_integration.md)）。

### 補足: `pre_spike_code` を「特定ステップ遅らせて」呼べるか

これは直接の質問への回答です。結論:

- **`pre_spike_syn_code` / `post_spike_syn_code`（シナプス単位のスパイク処理）** なら、
  標準の `axonal_delay_steps` / `back_prop_delay_steps` で実行タイミングを N ステップ遅延できます。
  「プレスパイクの効果を N ステップ遅らせたい」だけならこれで十分です。
- **`pre_spike_code` / `post_spike_code`（ニューロン単位の pre/post 変数更新）** そのものを遅延実行
  させる標準機能は **ありません**。これらは実際にニューロンが発火したステップで走ります
  （`axonal_delay_steps` は pre 変数のリングバッファをスライドさせるだけで、`pre_spike_code` の
  実行自体は遅らせない）。
- どうしてもニューロン単位の更新ロジックを遅延させたい場合は、`pre_vars` にシフトレジスタ（長さ N の
  配列や `pre_spike_time` を使った時刻比較）を作り、`pre_dynamics_code`（毎ステップ実行）で
  N ステップ経過した分を処理する、という手動実装が必要です。

## 8.2 カスタム更新（Custom Update）

ニューロン群・シナプス群・電流源は毎ステップ自動更新されますが、**時々だけ**走らせたい処理
（刺激後の状態リセット、勾配に基づく重み更新、変数の転置、バッチ間リダクション等）は
**custom update** で記述します。

```python
from pygenn import create_custom_update_model, create_var_ref

reset_model = create_custom_update_model(
    "reset",
    var_refs=[("V", "scalar"), ("RefracTime", "scalar")],
    update_code="""
        V = -65.0;
        RefracTime = 0.0;
    """)

model.add_custom_update(
    "pop_reset",          # 一意な名前
    "Reset",              # group_name（同名グループはまとめて実行される）
    reset_model, {}, {},
    {"V": create_var_ref(pop, "V"),
     "RefracTime": create_var_ref(pop, "RefracTime")})

# 実行（任意のタイミングで）
model.custom_update("Reset")
```

- `group_name` が同じ custom update はまとめて実行されます。
- リダクション: 変数を `CustomUpdateVarAccess.REDUCE_BATCH_SUM` 等にするとバッチ/ニューロン方向の集約が可能。
- 転置: `create_wu_var_ref(sg, "g", back_sg, "g")` で「順方向」変数の更新を「転置」変数へ同時反映
  （DENSE 結合のみ）。

## 8.3 カスタム結合更新（Custom Connectivity Update）

custom update と同様に手動トリガしますが、変数ではなく **結合トポロジ自体** を更新します
（SPARSE 結合のみ）。構造的可塑性（シナプスの生成・削除）などに使います。

```python
add_custom_connectivity_update(cu_name, group_name, syn_group,
    custom_conn_update_model, params={}, vars={}, pre_vars={}, post_vars={},
    var_refs={}, pre_var_refs={}, post_var_refs={}, egp_refs={})
```

`row_update_code` 内では `for_each_synapse { ... }` で行内シナプスを走査し、
`add_synapse(...)` / `remove_synapse()` で結合を増減できます。`host_update_code` は
GPU 行更新の前に CPU 側で走ります。

## 8.4 変数参照（Variable References）

あるモデルから**別のポピュレーションの変数**を参照する仕組み。後シナプスモデルから後ニューロンの
変数を見たり、custom update を特定の変数に「取り付ける」のに使います。

| 関数 | 参照先 |
|------|--------|
| `create_var_ref(pop, "V")` | ニューロン群の per-neuron 変数 |
| `create_psm_var_ref(sg, "x")` | シナプス群の後シナプスモデル変数 |
| `create_wu_pre_var_ref(sg, "x")` | 重み更新モデルの pre 変数 |
| `create_wu_post_var_ref(sg, "x")` | 重み更新モデルの post 変数 |
| `create_wu_var_ref(sg, "g"[, back_sg, "g"])` | per-synapse 重み変数（転置リンク可） |
| `create_spike_time_var_ref(pop)` | スパイク時刻 |
| `create_prev_spike_time_var_ref(pop)` | 直前スパイク時刻 |
| `create_out_post_var_ref(sg)` | シナプス群の後シナプス出力（inSyn 蓄積先） |
| `create_den_delay_var_ref(sg)` | 樹状突起遅延バッファ |
| `create_src_spike_count_var_ref(sg)` | ソース側スパイクカウント |

スパイク時刻系は初期化に合わせ、リセット時は大きな負値
（`np.finfo(np.float32).min`）に戻します。

### 転置の例

```python
wu_transpose_var_ref = {"R": create_wu_var_ref(sg, "g", back_sg, "g")}
```

`back_sg` は `sg` と転置の次元を持つシナプス群（pre/post が入れ替わったもの）。
update 後、順方向 `g` への変更が転置側にも反映されます（DENSE のみ）。

## 8.5 EGP 参照（Extra Global Parameter References）

複雑な custom update / custom connectivity update 間で EGP のデータを共有する仕組み。

```python
create_egp_ref(pop, "egp_name")        # ニューロン群の EGP
create_psm_egp_ref(sg, "egp_name")     # 後シナプスモデルの EGP
create_wu_egp_ref(sg, "egp_name")      # 重み更新モデルの EGP
```

---

[← 07 シミュレーション・記録](07_simulation_recording.md) ｜ [次: 09 API リファレンス →](09_api_reference.md)
