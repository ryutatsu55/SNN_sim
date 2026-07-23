# 06. カスタムモデル（GeNNCode）

[← 05 組み込みモデル](05_builtin_models.md) ｜ [次: 07 シミュレーション・記録 →](07_simulation_recording.md)

---

GeNN 最大の特徴は、ニューロン・シナプス・初期化などの挙動を **GeNNCode**（C 風の文字列）で
自由に定義できることです。組み込みモデルもこの仕組みで作られています。

## 6.1 GeNNCode 言語仕様

GeNNCode は基本的に **C99** ですが、以下の違いがあります。

- プリプロセッサなし
- `printf` 用の最小限の文字列サポートのみ（`strstr` 等は無い）
- ユーザコード内で関数・typedef・構造体を定義できない（構造体は一切不可）
- 一部の難解な C99 機能（8 進整数・16 進浮動小数リテラル等）は非対応
- **アドレス演算子 `&` は非対応**（GPU のローカル変数はレジスタ前提でアドレス不可）。
  EGP の部分配列は `const int *sub = egp + offset;` のように書く（`&egp[offset]` は不可）
- C++ 同様の関数オーバーロード対応（`sin(30.0f)` は float 版に解決）
- **浮動小数リテラルの型**: サフィックス無し `30.0` は `scalar`（モデルの precision）、
  `30.0f` は常に float、`30.0d` は常に double
- LP64 データモデル（`int`=32bit, `long`=64bit）
- 使用可能な標準ライブラリ関数（抜粋）:
  `cos, sin, tan, asin, acos, atan, atan2, exp, expm1, exp2, pow, log, log2, log10, log1p,
  sqrt, cbrt, hypot, ceil, floor, fmod, round, rint, trunc, fabs, fmax, fmin, fma, erf,
  tgamma, lgamma, copysign, min, max, abs, printf` ほか

### 乱数生成

GeNNCode 内で使える乱数関数:

| 関数 | 説明 |
|------|------|
| `gennrand()` | 32bit 符号なし整数 |
| `gennrand_uniform()` | [0,1] 一様 |
| `gennrand_normal()` | 平均0・標準偏差1 の正規 |
| `gennrand_exponential()` | λ=1 の指数 |
| `gennrand_log_normal(mean, std)` | 対数正規 |
| `gennrand_gamma(alpha)` | ガンマ（形状 alpha） |
| `gennrand_binomial(n, p)` | 二項 |

### 各コードで使える組み込み変数

- 共通: `dt`（タイムステップ）, `t`（現在時刻, ms）
- ニューロン系: `id`（ニューロン index）, `num_neurons`, `Isyn`（合計シナプス入力）
- 後シナプス系: `inSyn`（重み更新からの入力）, `id`, `num_neurons`
- 重み更新（シナプス）系: `id_pre`, `id_post`, `num_pre`, `num_post`, `st_pre`, `st_post`
- 初期化系: 書き込み先 `value`、ニューロン初期化なら `id`/`num_neurons`、シナプス初期化なら
  `id_pre`/`id_post`/`num_pre`/`num_post`

## 6.2 パラメータと派生パラメータ

`params` は均一な定数。`derived_params` は params から計算する定数で、効率のために使います
（例: 時定数から減衰率を事前計算）。

```python
params=["tau"],
derived_params=[("ExpTC", lambda pars, dt: np.exp(-dt / pars["tau"]))]
```

> 注意: dynamic param（実行時に変更可能なパラメータ）を変えても、それに依存する
> derived_params は更新されません（→[07_simulation_recording.md](07_simulation_recording.md)）。

## 6.3 変数アクセス（VarAccess）

バッチ実行時、各変数がバッチ間で **複製されるか共有されるか** を `VarAccess` で指定します。
共有変数は読み取り専用でなければなりません。実際に使われる値:

| VarAccess | 意味 |
|-----------|------|
| `READ_WRITE` | 読み書き（バッチごとに複製、既定相当） |
| `READ_ONLY` | 読み取り専用（バッチ間共有可） |
| `READ_ONLY_DUPLICATE` | 読み取り専用だがバッチごとに別値 |
| `READ_ONLY_SHARED_NEURON` | ニューロン間で共有される読み取り専用 |
| `REDUCE_BATCH_SUM` / `REDUCE_BATCH_MAX` | バッチ方向のリダクション（custom update 等） |
| `REDUCE_NEURON_SUM` / `REDUCE_NEURON_MAX` | ニューロン方向のリダクション |

```python
vars=[("V", "scalar", VarAccess.READ_WRITE),
      ("tauInv", "scalar", VarAccess.READ_ONLY)]
```

custom update 専用には `CustomUpdateVarAccess`（`READ_ONLY`, `REDUCE_BATCH_SUM/MAX`,
`REDUCE_NEURON_SUM/MAX` 等）を使います。

## 6.4 create_neuron_model

```python
create_neuron_model(class_name,
    params=None, vars=None, derived_params=None,
    sim_code=None, threshold_condition_code=None, reset_code=None,
    extra_global_params=None, additional_input_vars=None,
    auto_refractory_required=False)
```

- `sim_code`: 毎ステップの状態更新（微分方程式の積分など）
- `threshold_condition_code`: 発火条件（真偽式）
- `reset_code`: 発火後の処理

```python
from pygenn import create_neuron_model

if_model = create_neuron_model(
    "IF",
    params=["Vthresh"],
    vars=[("V", "scalar")],
    sim_code="V += Isyn * dt;",
    threshold_condition_code="V >= Vthresh",
    reset_code="V = 0.0;")
```

## 6.5 create_weight_update_model（最重要）

```python
create_weight_update_model(class_name,
    params=None, vars=None, pre_vars=None, post_vars=None,
    pre_neuron_var_refs=None, post_neuron_var_refs=None, psm_var_refs=None,
    derived_params=None,
    pre_spike_syn_code=None, post_spike_syn_code=None,
    pre_event_syn_code=None, post_event_syn_code=None,
    synapse_dynamics_code=None,
    pre_event_threshold_condition_code=None,
    post_event_threshold_condition_code=None,
    pre_spike_code=None, post_spike_code=None,
    pre_dynamics_code=None, post_dynamics_code=None,
    extra_global_params=None)
```

### 各コードブロックの違い（混同しやすい）

| コード | 実行タイミング | アクセス可能 |
|--------|----------------|--------------|
| `pre_spike_syn_code` | プレ発火がシナプスに到達した時、**各シナプス**ごと | synapse vars, pre/post |
| `post_spike_syn_code` | ポスト発火がシナプスに到達した時、各シナプス | synapse vars, pre/post |
| `pre_event_syn_code` | プレ「イベント」発生時、各シナプス（閾値は下記） | 同上 |
| `post_event_syn_code` | ポストイベント発生時、各シナプス | 同上 |
| `synapse_dynamics_code` | **毎ステップ**、各シナプス（時間駆動） | synapse vars |
| `pre_spike_code` | プレ発火時、**プレニューロンごとに1回**（pre 変数更新用） | pre vars のみ |
| `post_spike_code` | ポスト発火時、ポストニューロンごとに1回 | post vars のみ |
| `pre_dynamics_code` | 毎ステップ、プレニューロンごと | pre vars のみ |
| `post_dynamics_code` | 毎ステップ、ポストニューロンごと | post vars のみ |

> `*_syn_code` は **シナプス単位**（O(N²) になりうる）。`pre_vars`/`post_vars` と
> `pre_spike_code`/`pre_dynamics_code` を使うと、プレ/ポストニューロン単位（O(N)）で
> トレース等を保持でき、メモリと計算を節約できます。

### シナプスからの出力関数

| 関数 | 効果 |
|------|------|
| `addToPost(inc)` | 後シナプスニューロンの `inSyn` に加算（→ 後シナプスモデル経由で入力電流に） |
| `addToPostDelay(inc, delay)` | 樹状突起遅延 `delay`（ステップ）付きで後ニューロンへ加算 |
| `addToPre(inc)` | 逆方向（プレニューロンの入力変数）へ加算 |
| `postVar[delay]` | 後シナプス変数/参照へ遅延アクセス |

### イベントシナプス（spike-like event）

「スパイク」ではなく任意条件で発火するイベントを定義できます。`pre_event_threshold_condition_code`
が真になった pre ニューロンについて `pre_event_syn_code` が走ります。組み込みの `StaticGraded`
がこの仕組みの例です（`V > Epre` のとき `addToPost(...)`）。

### 自作 STDP の例

```python
stdp = create_weight_update_model(
    "stdp_additive",
    params=["tauPlus", "tauMinus", "aPlus", "aMinus", "wMin", "wMax"],
    vars=[("g", "scalar")],
    pre_spike_syn_code="""
        addToPost(g);
        const scalar dt = t - st_post;
        if (dt > 0) {
            const scalar timing = exp(-dt / tauMinus);
            g = fmin(wMax, fmax(wMin, g - (aMinus * timing)));
        }
    """,
    post_spike_syn_code="""
        const scalar dt = t - st_pre;
        if (dt > 0) {
            const scalar timing = exp(-dt / tauPlus);
            g = fmin(wMax, fmax(wMin, g + (aPlus * timing)));
        }
    """)
```

## 6.6 create_postsynaptic_model

```python
create_postsynaptic_model(class_name,
    params=None, vars=None, neuron_var_refs=None,
    derived_params=None, sim_code=None, extra_global_params=None)
```

`sim_code` は毎ステップ走り、`injectCurrent(x)` で対象ニューロンへ電流注入、`inSyn` を更新します。

```python
exp_curr = create_postsynaptic_model(
    "MyExpCurr",
    params=["tau"],
    derived_params=[("expDecay", lambda p, dt: np.exp(-dt / p["tau"]))],
    sim_code="""
        injectCurrent(inSyn);
        inSyn *= expDecay;
    """)
```

## 6.7 create_current_source_model

```python
create_current_source_model(class_name,
    params=None, vars=None, neuron_var_refs=None,
    derived_params=None, injection_code=None, extra_global_params=None)
```

```python
cs = create_current_source_model(
    "MyNoise",
    params=["sd"],
    injection_code="injectCurrent(gennrand_normal() * sd);")
```

## 6.8 create_custom_update_model / create_custom_connectivity_update_model

任意タイミングで GPU 処理を走らせるモデル。詳細は [08_advanced.md](08_advanced.md)。

```python
create_custom_update_model(class_name,
    params=None, vars=None, derived_params=None, var_refs=None,
    update_code=None, extra_global_params=None, extra_global_param_refs=None)

create_custom_connectivity_update_model(class_name,
    params=None, vars=None, pre_vars=None, post_vars=None, derived_params=None,
    var_refs=None, pre_var_refs=None, post_var_refs=None,
    row_update_code=None, host_update_code=None, ...)
```

## 6.9 create_var_init_snippet / create_sparse_connect_init_snippet

```python
create_var_init_snippet(class_name, params=None, derived_params=None,
                        var_init_code=None, extra_global_params=None)

create_sparse_connect_init_snippet(class_name, params=None, derived_params=None,
    row_build_code=None, col_build_code=None,
    calc_max_row_len_func=None, calc_max_col_len_func=None,
    calc_kernel_size_func=None, extra_global_params=None)
```

- 変数初期化: `var_init_code` 内で書き込み先 `value` に値を代入。
- 疎結合初期化: `row_build_code` 内で `addSynapse(id_post)`、`col_build_code` 内で `addSynapse(id_pre)`。
  `calc_max_row_len_func=lambda num_pre, num_post, pars: ...` で行の最大長を返す。

```python
my_uniform = create_var_init_snippet(
    "MyUniform",
    params=["min", "max"],
    var_init_code="value = min + (gennrand_uniform() * (max - min));")
```

---

[← 05 組み込みモデル](05_builtin_models.md) ｜ [次: 07 シミュレーション・記録 →](07_simulation_recording.md)
