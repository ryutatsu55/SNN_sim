# コンダクタンスの単位系と g_max スケーリング

Beggs & Plenz (2003) 再現 (`configs/beggs_plenz.yaml`, N=40000) で平均発火率が **52.85 Hz** と
生物学的にありえない値になった問題の調査記録。原因は「結合数 K が config 間で大きく変わったのに
g_max を据え置いた」ことで、**同時にシミュレーション速度の問題の原因でもあった**。

調査日: 2026-07-20 / 対象コミット: 9c533a3 以降の作業ツリー

---

### 単発 EPSP の換算表

`w=0.5, x_release=U=0.4, tau_syn=2 ms, tau_m=30 ms, v_rest=-74, v_th=-54, E_exc=0` で数値積分した値。
Δg = U·w·g_max_無次元 (物理版では Δg = U·w·g_max_物理·R_m)。

| g_max (無次元) | g_max 物理 [µS] (R_m=100MΩ) | Δg | 単発 EPSP |
|---|---|---|---|
| 4.0 (無次元版の既定) | 0.0400 (40 nS) | 0.800 | **3.18 mV** |
| 2.5 | 0.0250 (25 nS) | 0.500 | 2.01 mV |
| 1.24 | 0.0124 (12.4 nS) | 0.248 | 1.00 mV |
| 1.056 | 0.0106 (10.6 nS) | 0.211 | 0.86 mV |
| **0.62** | **0.0062 (6.2 nS)** | 0.124 | **0.50 mV** ← 物理版の既定 |
| 0.279 | 0.0028 (2.8 nS) | 0.056 | 0.23 mV |

閾値までの距離は 20 mV なので、無次元版の既定 `g_max=4.0` は
**1シナプスで閾値の 16% を埋めている**(飽和時 w=1.0 なら 6.21 mV = 31%)。
物理版の既定 `g_max=0.0062 µS` は単発 EPSP 0.5 mV で in vivo の実測レンジに入る
(`configs/components/plasticity.yaml` の `e-stdp_physical` / `i-stdp_physical`)。

---

## 2. 文献値

### 単発 EPSP (paired recording)

| 文献 | 系 | 振幅 |
|---|---|---|
| [Markram et al. 1997, J Physiol](https://consensus.app/papers/details/6d6bca67a0a65737b50f47b5a2080256/) | ラット L5 thick tufted ペア | 1.3 ± 1.1 mV (0.15–5.5) |
| [Feldmeyer et al. 1999, J Physiol](https://consensus.app/papers/details/4bc6b19eadaa5a05bdd9b69990d41245/) | ラット L4→L4 | 1.59 ± 1.51 mV |
| [Feldmeyer et al. 2002, J Physiol](https://consensus.app/papers/details/4744556aa82554bc8de4f2ca2f9cbbaa/) | ラット L4→L2/3 | 0.7 ± 0.6 mV |
| [Feldmeyer et al. 2006, J Physiol](https://consensus.app/papers/details/52998c50e5c859e3bbf7cccc2ac43e0d/) | ラット L2/3→L2/3 | 1.0 ± 0.7 mV |
| [Jouhanneau et al. 2015, Cell Reports](https://consensus.app/papers/details/1e2c44e4d7c0555ca3b726df2afb270f/) | マウス L2→L2 **in vivo** | 大半が **<0.5 mV** |
| [Angulo et al. 1999, J Neurophysiol](https://consensus.app/papers/details/8d088b15cc515af79063467835a6d9b6/) | pyr→FS 介在細胞 | 2.1 ± 1.3 mV |

まとめ: **slice で 0.7–1.6 mV、in vivo(高コンダクタンス状態)では大半が <0.5 mV**。
E→I は E→E より大きい (2.1 mV) — config の `p0_ei > p0_ee` と整合的。
Angulo は「FS 細胞を発火させるには 8 ± 5 個の同時入力が必要」と見積もっており、
単発で閾値の 16% を埋める現状の設定はこれと明らかに矛盾する。

### 単発 IPSP (FS→錐体)

抑制側は g_I を EPSP と**独立に**較正するための目標値。PubMed で単発抑制**コンダクタンス**
[nS] を直接報告する論文(下記)は全文が有料で数値を抽出できなかったため、**IPSP 振幅**を target にした。

| 文献 | 系 | 報告 |
|---|---|---|
| Thomson et al. 1996 | ラット新皮質 FS→錐体 | 単発 IPSP ≈ **2.0 mV @ 保持 −55 mV**(config が採用する target) |
| [Gupta, Wang & Markram 2000, Science](https://doi.org/10.1126/science.287.5451.273) | ラット体性感覚野 介在細胞3型 | GABA 単発コンダクタンスを報告(**有料・値未抽出**) |
| [Xiang, Huguenard & Prince 2002, J Neurophysiol](https://doi.org/10.1152/jn.2002.88.2.740) | ラット視覚野 L5 FS→錐体 | 単発 IPSC と GABA_A コンダクタンスをモデル推定(**有料・値未抽出**)。pathway が最も適合 |

**E と I で較正 handle の質が根本的に違う**:

- **EPSP**: 駆動力 `V−E_exc = 74 mV` が大きく安定 → g_E は**一点に確定**(0.7 mV → 0.0086 µS)。
- **IPSP**: 駆動力 `V−E_inh` が小さく保持電位と E_Cl の両方に依存 → g_I は**帯**にしかならない。
  保持 −55 で 25 mV だが**静止 −74 では 6 mV**(E_inh=−80 に接近し、ほぼ shunting で振幅が消える)。

したがって IPSP 振幅目標がそのまま g_I を決める。**抑制シナプスの減衰 tau_syn=4 ms**
(興奮の 2 ms ではない。ExpCond inhibitory)を使い、EPSP と同じ換算規約 U·W=0.2, R_m=100 MΩ で数値積分:

| IPSP 目標 @ −55 | g_I [µS] | **g_I/g_E** |
|---|---|---|
| 0.2 mV | 0.0041 | 0.5 |
| 1.0 mV | 0.0209 | 2.4 |
| **2.0 mV (採用)** | **0.0429** | **5.0** |
| 3.5 mV | 0.0781 | 9.1 |

**g_I/g_E = 5.0 は押し付けた補正ではなく、EPSP/IPSP を独立に較正した結果として出てくる比**である
(2026-07-21 ユーザ確認)。ただし §3 で見るように、これでも構造的 K_E/K_I=8.10 には届かず、
N=40000 での総コンダクタンス比は I/E ≈ 0.62 と**興奮優位**のまま。臨界点探索では下の `gi*` を掃引。

### スケーリング則

- [**Barral & Reyes 2016, Nature Neuroscience**](https://consensus.app/papers/details/1e37edd5df095c21a99a3ce989281ac7/) — **最重要**。
  培養系(= Beggs & Plenz と同じ標本系)で「シナプス強度は結合数 K に対して **~1/√K** でスケールする」
  ことを実測し、理論の理想値に近いと報告。E/I バランスと in vivo 様の活動が保たれる。
- [Girardi-Schappo et al. 2019](https://consensus.app/papers/details/bd0ecaa31f665baeb9f5c3bfea660999/) /
  [2020](https://consensus.app/papers/details/c3dbf52007185072b14f9bcb396dfada/) —
  E/I 重み比 g_c ≈ 4 が二次相転移点に対応し、そこでべき乗アバランシェが出る。
- [Morrison et al. 2007, Neural Computation](https://consensus.app/papers/details/e8423d59af83576aab6caa7dd1e37c3a/) —
  balanced random network + STDP が非同期不規則発火と両立する条件。

### 較正の目標値

- [Beggs & Plenz 2004, J Neurosci](https://consensus.app/papers/details/da86a243a65852c69fec148a9f21fd5c/) —
  培養は **4,736 ± 2,769 アバランシェ/時 (≈1.3/s)**、寿命は数 ms。
  → 我々の N=1000 スモークは収束後でも **283/s** で **200倍多い**。
- 皮質の自発発火率は一般に 0.1–5 Hz。**52.85 Hz は 1〜2桁過大**。

---

## 3. fan-in の実測

`layout.ids_by_mode()` と実結合 COO から算出した、興奮性ニューロン1個が受ける入力本数。

| config | N | E→E fan-in | I→E fan-in | **K_E/K_I** | 総シナプス |
|---|---|---|---|---|---|
| `akita_soc.yaml` | 100 | 79.0 | 20.0 | 3.95 | 9,900 |
| `beggs_plenz_n1000.yaml` | 1,000 | 144.5 | 37.9 | 3.81 | 189,614 |
| `beggs_plenz.yaml` | 40,000 | **1,134.4** | 140.1 | **8.10** | 48,090,437 |

**問題は 2 つある。**

1. **総入力が 14.4 倍過剰** — g_max 据え置きなので E 入力が K_E に比例して増える。
2. **E:I バランスが崩れている** — K_E/K_I が 3.95 → 8.10。
   σ_ie(175 µm) が σ_ee(300 µm) より局所的なため、箱が σ を超えて広がると
   興奮だけが遠方から集まる。**g_max の一律スケーリングでは直らない**。

   akita_soc 相当の比を回復するには g_max_I を **2.05 倍**する必要がある。

**独立較正した g_I/g_E=5.0 では届かない**: §2 で EPSP/IPSP から独立に較正した比は 5.0 だが、
構造的 `K_E/K_I = 8.10` の方が大きいため、総コンダクタンス比は

```
(g_I·K_I) / (g_E·K_E) = (0.0429·140.1) / (0.0086·1134.4) = 6.01 / 9.76 = 0.62
```

で **N=40000 では総抑制が総興奮の 62% しかない = 興奮優位**。単発 IPSP が単発 EPSP より
大きい(2.0 vs 0.7 mV → 比 5.0)ことは E:I 不均衡を緩和するが、σ_ie < σ_ee による構造的な
興奮偏重(K_E/K_I=8.10)を打ち消すには足りない。これが §5 で「暴走」する主因であり、
臨界に持ち込むには `gi*` 掃引で g_I をさらに上げる(比 8 前後)必要があることを示す。

---

## 4. 平均場による診断: 低発火の定常解が存在しない

ExpCond の平均コンダクタンスは `Δg × rate × tau` なので、定常膜電位は

```
gE = U·w·g_max_E · r · tau_E · K_E,   gI = U·w·g_max_I · r · tau_I · K_I
V_ss = (v_rest + gE·E_exc + gI·E_inh) / (1 + gE + gI)
```

| config | r=1 Hz | r=5 Hz | r=50 Hz |
|---|---|---|---|
| akita_soc N=100 | −66.5 | −51.0 ⚠ | −31.4 ⚠ |
| beggs N=1000 | −61.9 | −44.4 ⚠ | −30.0 ⚠ |
| **beggs N=40000** | **−33.7 ⚠** | −20.6 ⚠ | −16.4 ⚠ |

(閾値 −54 mV。⚠ = 閾値超え)

**N=40000 は r=1 Hz でも既に閾値を超えており、低発火の定常解が数学的に存在しない。**
一方 N=1000 は r=1 Hz で −61.9 と閾値下にあり、実際にスモークでは 1.19 Hz に収束した。
平均場の予測と実測が一致している。

---

## 5. スケーリング候補の定量比較 (N=40000)

> **注記**: この表は無次元単位・「E/I 補正係数 2.05」枠組みでの初期解析。§2 の独立較正
> (g_E=0.0086, g_I=0.0429 µS, g_I/g_E=5.0)は生物学的な出発点を与えるが、総 I/E≈0.62 と
> 興奮優位で、下表が言う「1/√K のみ」に近い暴走側に位置する。臨界へ持ち込むには `gi*` 掃引で
> g_I をさらに上げる必要がある。V_ss の相図としての含意は今も有効なので残す。

akita_soc (K_E=79, g_max=4.0) を基準とする。E/I 補正係数 = 2.05(= K_E/K_I の N 依存分)。

| 案 | g_E | g_I | 単発EPSP | V_ss@1Hz | V_ss@5Hz | 評価 |
|---|---|---|---|---|---|---|
| 現状 | 4.000 | 4.000 | 3.18 mV | −33.7 ⚠ | −20.6 ⚠ | 暴走 |
| 1/√K のみ | 1.056 | 1.056 | 0.86 mV | −52.2 ⚠ | −30.4 ⚠ | 暴走 |
| 1/√K + E/I補正 | 1.056 | 2.165 | 0.86 mV | **−54.3** | −37.1 | 閾値の 0.3 mV 下 = 臨界の縁 |
| 1/K のみ | 0.279 | 0.279 | 0.23 mV | −66.1 | −48.3 | 安定 |
| 1/K + E/I補正 | 0.279 | 0.572 | 0.23 mV | −66.5 | −51.0 | 安定 (akita_soc の動作点と一致) |

### 考察

- **1/√K は Barral & Reyes の実測則**であり、単発 EPSP 0.86 mV も slice の実測範囲 (0.7–1.6 mV) に入る。
  ただし E/I 補正なしでは暴走する。1/√K が成立するには「平均入力の増大を抑制が打ち消す」強い
  E/I キャンセルが前提だが、本モデルの E:I 比は空間 σ で決まってしまい、その前提を満たさない。
- **1/√K + E/I補正は V_ss が閾値の 0.3 mV 下**という際どい点に落ちる。これは相転移の縁であり、
  **臨界アバランシェが期待される場所そのもの**。escape noise(現在は §6 の `b: auto`、
  v=v_th で発火確率 1.0)のソフト閾値と併せると Beggs&Plenz の臨界状態に対応する可能性がある。
- **1/K + E/I補正は akita_soc の動作点を厳密に再現する** (独立に逆算した g_max が 0.279 に一致)。
  単発 EPSP 0.23 mV は in vivo の実測範囲。安全側で subcritical。

### 実装上の注意

既存の `normalize_gmax_by_fan_in: true` は `calculate_gmax_scale` で **集団ごとに 1/fan_in** を掛ける
([custom_Akita.py](../src/models/plasticity/custom_Akita.py))。これは 1/K スケーリングを自動化する一方で、
E 総コンダクタンスも I 総コンダクタンスも g_max に等しくなるため **E:I 比を強制的に 1:1 にしてしまう**。
akita_soc の比 (3.95) を保ちたい場合は g_max_E/g_max_I を別々に与える必要がある。
1/√K を使いたい場合は指数を選べるパラメータ (例 `gmax_fan_in_exponent`: 0 / 0.5 / 1.0) の追加が要る。

### 推奨

1/K + E/I 補正で安定動作させ、そこから g_max を 1/√K 方向へ上げていって臨界点を探す
(Akita 実験と同じ g_max スイープ手法)。1/√K 近傍が臨界の候補領域である。



## 6. escape gain b の自動計算 (`b: auto`)

§4・§5 は定常膜電位 `V_ss` を閾値 `v_th` と突き合わせて動作点を論じているが、
このニューロンモデルの発火は**硬い閾値ではなく滑らかな escape noise** であり、
そのままでは `v_th` 到達が発火をほとんど意味しない。発火確率は

```
SpikeProb(V) = CScale · exp((V - v_th) / b),
CScale       = f_rest · 0.001 · dt · exp(-(v_rest - v_th) / b)     (Eq.S6)
```

なので `V = v_th` でも `SpikeProb = CScale`。既定 `b=4.0` / `f_rest=0.4 Hz` / `dt=0.1 ms` では
`CScale ≈ 6.5e-4` にすぎず、閾値に達しても 1 ステップの発火確率は 0.065% でしかない。
つまり `V_ss` を `v_th` と比べる議論は、**閾値がその温度で実際に効いている**ことを暗黙に仮定していた。

これを一貫させるため、`b` に `auto` を指定できるようにした
([akita_escape_lif.py](../src/models/neurons/akita_escape_lif.py) の `resolve_escape_gain`)。
`auto` は `V = v_th` で `SpikeProb = CScale = 1.0` となる `b` を選ぶ:

```
b = (v_th - v_rest) / ln(1 / (f_rest · 0.001 · dt))
```

既定値では `b ≈ 1.975 mV`(従来の 4.0 より鋭い)。これで「閾値到達 = ほぼ確実に発火」となり、
§4・§5 の `V_ss` vs `v_th` の比較がそのまま発火の議論として読める。両ニューロンモデル
(無次元版・物理版)で使え、物理版 config は既定で `b: auto`。数値を明示すれば従来どおり固定 `b`。

**臨界探索への含意**: `b` は閾値近傍の発火の鋭さそのものなので、g_max スイープとは独立な
「もう 1 本の軸」になる。`b` を小さくするほどハード閾値に近づき、`V_ss` が閾値をまたぐ点での
発火感受性が上がる。§5 で「臨界の縁」とした `1/√K + E/I補正`(`V_ss` が閾値の 0.3 mV 下)の
アバランシェ挙動は `b` に強く依存するはずで、スイープ時は `b: auto` に固定して比較条件を揃えるのがよい。


## 7. 再現手順

本ドキュメントの §1・§3・§4・§5 の数値はすべて次のスクリプトで再現できる
(fan-in は実際にネットワークを構築して実測、EPSP は数値積分、V_ss は平均場)。

```bash
python scripts/analyze_gmax_scaling.py             # fan-in を実測 (数十秒)
python scripts/analyze_gmax_scaling.py --skip-fanin  # 既知の fan-in を使い即座に表示
```

その他:

```bash
# 単位系のテスト (無次元版 / 物理版の等価性)
pytest test/core/test_unit_systems.py -v

# GeNN 実機での再現性と単位換算 (コンパイルを伴うため既定ではスキップ)
RUN_GENN_TESTS=1 pytest test/core/test_genn_reproducibility.py -v

# ネットワーク構築の決定性 (seed -> 実現) の golden hash
pytest test/core/test_network_reproducibility.py -v

# スモーク実行 (N=1000, 生物時間2分) — 発火率と実行速度を体感する
python scripts/beggs_plenz_avalanche.py \
    --config configs/beggs_plenz_n1000.yaml --task-name beggs_plenz_smoke
```

§6 の性能値は使用 GPU (RTX 5070 Ti) に依存する。再測定するときは
`sim.step(n)` と `sim.flush_recording()` を分けて計測すること
(まとめて測ると転送を含んだ値になる。ただし実測では転送の寄与は 0.002% だった)。
