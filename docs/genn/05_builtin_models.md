# 05. 組み込みモデル一覧

[← 04 モデル構築](04_model_building.md) ｜ [次: 06 カスタムモデル →](06_custom_models.md)

---

GeNN には実用的な組み込みモデルが用意されています。すべて文字列名で参照でき、内部実装は
`include/genn/genn/*.h` にあります（自作モデルの良い手本にもなります）。
**params** は population 内で均一な定数、**vars** はニューロン/シナプスごとの状態変数です。

## 5.1 ニューロンモデル（`neuron_models`）

| モデル | params | vars | 概要 |
|--------|--------|------|------|
| `LIF` | `C, TauM, Vrest, Vreset, Vthresh, Ioffset, TauRefrac` | `V, RefracTime` | 漏れ積分発火。不応期つき指数減衰 |
| `Izhikevich` | `a, b, c, d` | `V, U` | Izhikevich モデル（閾値 `V>=29.99`、発火後 `V=c, U+=d`） |
| `IzhikevichVariable` | （なし） | `V, U, a, b, c, d` | `a,b,c,d` をニューロンごとの変数にした版（異種パラメータ可） |
| `Poisson` | `rate`(Hz) | `timeStepToSpike` | 指数分布による Poisson 発火。派生パラメータ `isi` |
| `SpikeSourceArray` | （なし） | `startSpike, endSpike` | 外部スパイク時刻配列を再生。EGP `spikeTimes`(scalar*) を使用 |
| `TraubMiles` | `gNa, ENa, gK, EK, gl, El, C` | `V, m, h, n` | Hodgkin-Huxley（Traub & Miles）。内部で 25 分割積分 |
| `RulkovMap` | `Vspike, alpha, y, beta` | `V, preV` | 1 次元離散写像ニューロン（`DT=0.5ms` 前提） |

### LIF の実装（参考）

```cpp
// sim_code
if (RefracTime <= 0.0) {
  scalar alpha = ((Isyn + Ioffset) * Rmembrane) + Vrest;
  V = alpha - (ExpTC * (alpha - V));
}
else {
  RefracTime -= dt;
}
// threshold:  RefracTime <= 0.0 && V >= Vthresh
// reset:      V = Vreset; RefracTime = TauRefrac;
```

派生パラメータ: `ExpTC = exp(-dt/TauM)`, `Rmembrane = TauM/C`。
`Isyn` は後シナプスモデルから注入される合計シナプス入力です。

### SpikeSourceArray の使い方

決まった時刻に発火する入力に使います。`spikeTimes`（EGP）にソート済みの全スパイク時刻を入れ、
各ニューロンの開始/終了インデックスを `startSpike` / `endSpike` に設定します。

```python
end_spike = np.cumsum(np.bincount(spike_ids, minlength=N))
start_spike = np.concatenate(([0], end_spike[:-1]))
spike_times = spike_times[np.lexsort((spike_times, spike_ids))]

ssa = model.add_neuron_population("SSA", N, "SpikeSourceArray", {},
                                  {"startSpike": start_spike, "endSpike": end_spike})
ssa.extra_global_params["spikeTimes"].set_init_values(spike_times)
```

## 5.2 重み更新モデル（`weight_update_models`）

| モデル | params | vars | 動作 |
|--------|--------|------|------|
| `StaticPulse` | （なし） | `g`(READ_ONLY) | プレ発火時 `addToPost(g)`。異種重み |
| `StaticPulseConstantWeight` | `g` | （なし） | 重み均一版の `addToPost(g)` |
| `StaticPulseDendriticDelay` | （なし） | `g`(READ_ONLY), `d`(uint8, READ_ONLY) | `addToPostDelay(g, d)`。異種の樹状突起遅延 |
| `StaticGraded` | `Epre, Vslope` | `g`(READ_ONLY) | アナログ（傾斜電位）シナプス。pre 電位 `V` を参照 |
| `STDP` | `tauPlus, tauMinus, Aplus, Aminus, Wmin, Wmax` | `g` | 加算型 STDP（最近傍ペアリング、ハード境界） |

### `StaticPulse` の実装

```cpp
// pre_spike_syn_code
addToPost(g);
```

`addToPost(x)` は後シナプスニューロンの `inSyn` に `x` を加算する関数です。
逆方向は `addToPre(x)`、遅延付き加算は `addToPostDelay(x, delay)`。

### `STDP` の実装（学習則の手本）

```cpp
// pre_spike_syn_code（プレ発火時）
addToPost(g);
scalar dt = t - st_post;            // st_post = 直近のポスト発火時刻
if (dt > 0) {
    scalar timing = exp(-dt / tauMinus);
    scalar newWeight = g - (Aminus * timing);
    g = fmax(Wmin, fmin(Wmax, newWeight));   // 抑圧（depression）
}

// post_spike_syn_code（ポスト発火時）
scalar dt = t - st_pre;             // st_pre = 直近のプレ発火時刻
if (dt > 0) {
    scalar timing = exp(-dt / tauPlus);
    scalar newWeight = g + (Aplus * timing);
    g = fmax(Wmin, fmin(Wmax, newWeight));   // 増強（potentiation）
}
```

`st_pre` / `st_post` は GeNN がスパイク時刻を追跡する組み込み変数です。
自作の学習則は [06_custom_models.md](06_custom_models.md) を参照。

## 5.3 後シナプスモデル（`postsynaptic_models`）

シナプス入力 `inSyn` を入力電流（等）へ変換します。`injectCurrent(x)` で対象ニューロンへ注入。

| モデル | params | var_refs | 動作 |
|--------|--------|----------|------|
| `ExpCurr` | `tau` | — | 電流として指数減衰。`injectCurrent(init*inSyn); inSyn*=expDecay;` |
| `ExpCond` | `tau, E` | `V`(ニューロン電位, READ_ONLY) | コンダクタンス型。`injectCurrent(inSyn*(E-V))` |
| `DeltaCurr` | （なし） | — | 瞬時注入。`injectCurrent(inSyn); inSyn=0;` |

`ExpCurr` の派生パラメータ: `expDecay = exp(-dt/tau)`, `init = tau*(1-exp(-dt/tau))/dt`。
`ExpCond` は逆転電位 `E` を使う電位依存（オームの法則的）シナプスで、後シナプスニューロンの
電圧 `V` への参照が必要です（`init_postsynaptic(..., var_refs={"V": create_var_ref(pop,"V")})`）。

## 5.4 電流源モデル（`current_source_models`）

| モデル | params | vars | 動作 |
|--------|--------|------|------|
| `DC` | `amp` | — | 定常電流 `injectCurrent(amp)` |
| `GaussianNoise` | `mean, sd` | — | 正規分布ノイズ `injectCurrent(mean + gennrand_normal()*sd)` |
| `PoissonExp` | `weight, tauSyn, rate` | `current` | Poisson スパイク列＋指数シナプスを 1 対 1 結合相当で注入 |

`PoissonExp` は「Poisson ニューロン群を 1 対 1 で繋ぐ」代わりの効率的な実装です。
派生パラメータ: `ExpDecay = exp(-dt/tauSyn)`, `Init = weight*(1-exp(-dt/tauSyn))*(tauSyn/dt)`,
`ExpMinusLambda = exp(-(rate/1000)*dt)`。

## 5.5 変数初期化スニペット（`init_var_snippets`）

`init_var("名前", {params})` の形で変数初期化に使います。

| スニペット | params | 説明 |
|-----------|--------|------|
| `Uninitialised` | — | 初期化しない |
| `Constant` | `constant` | 定数で埋める |
| `Uniform` | `min, max` | 一様乱数 |
| `Normal` | `mean, sd` | 正規乱数 |
| `NormalClipped` | `mean, sd, min, max` | [min,max] に収まるまで再サンプリングした正規乱数 |
| `NormalClippedDelay` | `mean, sd, min, max`(ms) | 遅延用。ms をステップに換算し丸める |
| `Exponential` | `lambda` | 指数分布 |
| `Gamma` | `a, b` | ガンマ分布（形状 a, スケール b） |
| `Binomial` | `n, p` | 二項分布 |
| `Kernel` | （EGP `kernel`） | 外部カーネル配列から初期化（畳み込み用） |

```python
{"g": init_var("Normal", {"mean": 0.5, "sd": 0.1})}
```

## 5.6 疎結合初期化スニペット（`init_sparse_connectivity_snippets`）

`init_sparse_connectivity("名前", {params})` の形で `SPARSE` / `BITMASK` / `PROCEDURAL` 結合を生成します。

| スニペット | params | 説明 |
|-----------|--------|------|
| `OneToOne` | — | 1 対 1（対角）結合 |
| `FixedProbability` | `prob` | 各 pre-post を確率 `prob` で結合 |
| `FixedProbabilityNoAutapse` | `prob` | 自己結合（`id_pre==id_post`）を除外 |
| `FixedNumberPostWithReplacement` | `num` | 各 pre が `num` 個の post へ（重複あり） |
| `FixedNumberPreWithReplacement` | `num` | 各 post が `num` 個の pre から（重複あり） |
| `FixedNumberTotalWithReplacement` | `num` | 全体で `num` 本のシナプス（多項分布で行に配分） |
| `Conv2D` | `conv_kh, conv_kw, conv_sh, conv_sw, conv_padh, conv_padw, conv_ih, conv_iw, conv_ic, conv_oh, conv_ow, conv_oc` | 2D 畳み込み結合 |

```python
init_sparse_connectivity("FixedProbability", {"prob": 0.1})
```

> 畳み込み的な結合は `TOEPLITZ` 行列型＋`init_toeplitz_connectivity` でも実現できます。

---

[← 04 モデル構築](04_model_building.md) ｜ [次: 06 カスタムモデル →](06_custom_models.md)
