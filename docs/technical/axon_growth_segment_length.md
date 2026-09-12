# `axon_growth` の `segment_length` を変えるときの再計算ルール

`segment_length`（以下 Δs）は**積分の刻み幅**であって、モデルのパラメータではない。
値を変えるときは、モデルの意味を決めている量が保たれるように他のパラメータを合わせる
必要がある。ここにその規則と、変更後の妥当性確認の手順をまとめる。

出典は Orlandi et al. (2013) Nat. Phys. 9:582（`docs/private/nphys2686.pdf`）と
Sumi et al. (2025) Front. Neurosci. 19:1570783 §2.8（`docs/private/damage.pdf`）。
論文の値は Δs = 0.1 mm / σ_θ = 0.1 rad で、これが `axon_growth` プロファイル。

> **結論 (ルート `CLAUDE.md` から移設した一段落):**
> Δs を変えるときは `angle_sigma` を `0.1 * sqrt(Δs / 100)` として再計算すること。
> 転回ウォークはステップ単位なので persistence length は `Lp = 2Δs/σ²`。論文の (100 µm, 0.1 rad)
> は 20 mm で、平均軸索長 1.1 mm の 18 倍 — これが「quasi-straight」の数値的な意味である。
> Δs だけ下げると `Lp` も一緒に落ちる（Δs=10 で 2 mm = 軸索長と同程度にくねる）ので、
> それは別のモデルになる。`mean_axon_length` / `dendrite_radius` / `connection_prob` は
> そのまま流用してよい（最後のものは *invasion* ごとなので構成上 Δs 不変:
> `p_within` は Δs 100→2 で 0.196–0.215 に留まる）。自然に動くのは
> `floor(L/Δs)` の切り捨て（実効弧長 1043 → 1078 µm、Δs 100 → 10。出次数も 25.4 → 26.5 と追随）と、
> 細い回廊がそもそも存在するか（4.5 µm のブリッジに侵入する軸索は Δs=100 で 4.5%、Δs=10 で 89.6%）。
> `axon_growth_fine` が実例。
>
> 境界処理・閉じ込め・invasion 抽選は別ファイル →
> [`axon_growth_confinement.md`](axon_growth_confinement.md)

> **モデルそのものと実装アルゴリズムの逐条解説は巻末の
> [付録: `axon_growth` のモデルとアルゴリズム (詳細)](#付録-axon_growth-のモデルとアルゴリズム-詳細) にある。**
> 「何をモデル化しているのか」「`_grow_axons` / `generate_sparse` が実際に何をしているか」
> 「乱数消費の順序」「論文との差分」を知りたいときはそちら。本編 (§1〜§4) は
> Δs を変更する人向けの実務規則に絞ってある。

---

## 1. 唯一の必須の再計算: `angle_sigma`

```
sigma_new = sigma_ref * sqrt(dS_new / dS_ref)          # 基準は Δs=100 um, σ=0.1 rad
```

### なぜか

`_grow_axons()` の方向はステップごとの角度ランダムウォーク `theta += N(0, sigma)` で、
n ステップ後の方向相関は

```
<cos(theta_n - theta_0)> = exp(-n * sigma^2 / 2)
```

弧長 `s = n * dS` で書き直すと `exp(-s / Lp)`、すなわち

```
Lp = 2 * dS / sigma^2        # 持続長 [um]
```

`Lp` が「軸索がどれだけ進むと元の向きを忘れるか」＝ Orlandi の言う **"quasi-straight"**
の実体で、軸索がソマからどれだけ遠くへ届くかを決める。論文の値では

```
Lp = 2 * 100 / 0.1^2 = 20,000 um = 20 mm
```

で、平均軸索長 1.1 mm の **18 倍**。全長にわたる向きの揺らぎは `sigma * sqrt(n) = 0.33 rad
= 19°` にしかならない。`sigma` を据え置いて Δs だけ下げると `Lp` が比例して落ち、
別のモデルになる（Δs=10 で `Lp = 2 mm`、軸索長と同程度＝クネクネした軌跡）。

### 早見表（Lp = 20 mm を保つ値）

| Δs [µm] | `angle_sigma` [rad] | 1 セグメントあたりの折れ角 | Lp [mm] |
|---|---|---|---|
| 200 | 0.1414214 | 8.10° | 20 |
| **100**（`axon_growth`, 論文値） | **0.1** | 5.73° | 20 |
| 50 | 0.0707107 | 4.05° | 20 |
| 25 | 0.05 | 2.86° | 20 |
| 20 | 0.0447214 | 2.56° | 20 |
| **10**（`axon_growth_fine`） | **0.0316228** | 1.81° | 20 |
| 5 | 0.0223607 | 1.28° | 20 |
| 2 | 0.0141421 | 0.81° | 20 |
| 1 | 0.01 | 0.57° | 20 |

### 検証済み（3 mm 円板、N=2827、壁の影響がないので σ スケーリングの純粋な対照群）

軸索の直線度（端点間距離 ÷ 弧長）が保たれていれば正しい:

| Δs | σ | 直線度 |
|---|---|---|
| 100 | 0.1 | 0.9336 |
| 50 | 0.0707 | 0.930 |
| 25 | 0.05 | 0.929 |
| 10 | 0.0316 | 0.9298 |
| 10 | **0.1（補正なし＝誤り）** | **0.854** |

補正すれば小数点 2 桁まで一致し、補正を忘れると 8% 劣化する。**新しい Δs を入れたら
まずこれを測ること。**

---

## 2. 変えてはいけないパラメータ

| パラメータ | 理由 |
|---|---|
| `mean_axon_length` | 軸索長は Rayleigh 分布・平均 1.1 mm という**生物側の量**。Δs とは無関係 |
| `connection_prob` | 樹状突起円への**侵入 1 回**あたりの確率。侵入は「連続セグメントの極大区間」なので定義上 Δs 不変。実測で Δs を 100→2（50 倍）にしても `p_within` は 0.196〜0.215 で動かない |
| `dendrite_radius` | 樹状突起の広がりという生物側の量 |
| `boundary` | 境界の扱い方の選択であって刻み幅と無関係 |

---

## 3. Δs を変えると勝手に動く量（ノブではないので触らない。ただし把握しておく）

### (a) `floor(L / Δs)` による軸索長の切り捨て

ステップ数は `n_seg = floor(L / Δs)` なので、公称の全長は平均 **Δs/2 だけ短くなる**。
Δs を細かくするとこのバイアスが減り、論文値 1100 µm に近づく：

| Δs | 実現される平均弧長 [µm] | k_out（3 mm 円板） |
|---|---|---|
| 100 | 1043 | 25.4 |
| 50 | 1060 | 25.9 |
| 25 | 1071 | 26.2 |
| 10 | 1078 | 26.5 |

**k_out は弧長にほぼ比例する**（+3.4% の弧長に対し +3.8% の出次数）。これは歪みではなく
バイアス除去なので、旧値に合わせ込むために `mean_axon_length` をいじらないこと。
「Δs を変えたら出次数が数 % 動いた」の出所は毎回これ。

### (b) 線分内外判定の分解能 = `Δs / _SEGMENT_SAMPLES`

`BaseArea._SEGMENT_SAMPLES = 16` は**セグメント長に対する相対値**なので、分解能は
Δs に比例する（Δs=100 なら 6.25 µm、Δs=10 なら 0.63 µm）。**Δs を下げる方向なら自動的に
細かくなるので何もしなくてよい。上げる方向のときだけ**、扱う形状の最小構造（ブリッジ幅、
部分領域の空隙）より粗くならないか確認すること。

### (c) 細い通路を通れるかどうか — Δs を変える主な動機

長さ Δs の直線弦が幅 w の通路に収まる角度の許容範囲は `asin(w / Δs)`：

| Δs | w = 20 µm | w = 4.5 µm |
|---|---|---|
| 100 µm | ±11.5° | ±2.6° |
| 50 µm | ±23.6° | ±5.2° |
| 25 µm | 制約なし | ±10.4° |
| 10 µm | 制約なし | ±26.7° |

`hierarchical_modular_grid2`（side 100 / bridge 4.5 µm、Watanabe et al. のデバイス寸法）で
同一 seed・同一エリアで比べた実測：

| | Δs=100 | Δs=10 |
|---|---|---|
| ブリッジに入った軸索 | 3/67 (4.5%) | 60/67 (89.6%) |
| 2 モジュール以上を訪れた軸索 | 0 (0.0%) | 32 (47.8%) |
| シナプス数 | 50 | 85 |

**Δs が通路の寸法と同程度以上だと、その通路は幾何として存在しないのと同じになる。**
狭い通路を持つエリアを使うなら、通路幅の数倍以下の Δs を選ぶこと。

### (d) `AxonGrowthTopology._MAX_SLIDE_ITERS`

壁沿いの滑り直しに必要な回数はジオメトリ依存。実測: 円板 3 回、
`hierarchical_modular_grid`(side 200) が Δs=100 で **105 回**、Δs=10 で 12 回。
既定 256 は上記いずれでも binding しないが、**新しいエリアでは確認すること**
（上げて結果が変わるなら効いてしまっている）。

### (e) 計算コスト

セグメント数は 1/Δs に比例する。実測のビルド時間（`_generate_global_matrices()` のみ）:

| config | Δs=100 | Δs=10 |
|---|---|---|
| `axon_growth.yaml`（円板 N=2827） | 0.5 s | 2.2 s |
| `axon_growth_grid.yaml`（N=256、36 part の合成エリア） | 9.0 s | 27.9 s |

`CompositeArea` は 1 回の SDF 評価で全 part を舐めるので、part 数の多いエリアほど効く。

---

## 4. 新しい Δs のプロファイルを足す手順

1. `configs/components/connections.yaml` にプロファイルを追加し、`angle_sigma` を
   §1 の式で計算した値にする（コメントに `= 0.1 / sqrt(dS/100)` の形で残すこと）。
   `mean_axon_length` / `dendrite_radius` / `connection_prob` は写すだけ。
2. `src/models/network/connectors.py` に**同名の空サブクラス**を登録する。
   `profile_name` は YAML のキーであると同時にレジストリのキー
   （`NetworkBuilder._component_classes()` が `CONNECTION_MODELS.get()` を引く）なので、
   これが無いと解決できない。`AxonGrowthFineTopology` が見本。
3. 検証: 3 mm 円板（`configs/axon_growth.yaml`）でプロファイルを差し替えて
   **直線度が 0.93 前後であること**（§1）と、**実弧長/公称が 100%** であることを確認。
   円板は凸で滑らかなので、ここで劣化するなら σ かコードの問題。
4. 目的のエリアで `p_within` が `connection_prob` 前後から動いていないこと（§2）と、
   通路の通過率（§3c）を測る。

**注意: Δs を変えると同じ seed でも別のネットワークになる。** 乱数の消費数がステップ数に
依存するため、既存の実行結果とは比較できない。

---

## 関連

- `CLAUDE.md` 注記 16 / 16b / 16c — 壁沿いの伸長、`contains()` の許容差、角の不動点
- `src/models/network/connectors.py::AxonGrowthTopology` — 実装本体
- `test/test_axon_growth.py::TestInvasionLottery::test_degree_barely_depends_on_segment_length`
  — 侵入ルールの Δs 不変性を pin しているテスト

---
---

# 付録: `axon_growth` のモデルとアルゴリズム (詳細)

ここまでは「Δs を触るときの規則」だけを扱った。以下は**モデルそのものと、
`src/models/network/connectors.py::AxonGrowthTopology` の実装が実際に何をしているか**の
逐条解説。読む順は A → B → C → D → E で、F 以降は契約・コスト・限界の確認用。

---

## A. 何をモデル化しているのか

### A.1 出典と、原文が定めている規則

- Orlandi, Soriano, Alvarez-Lacalle, Teller, Casademunt (2013)
  *Noise focusing and the emergence of coherent activity in neuronal cultures*,
  Nat. Phys. 9:582 (`docs/private/nphys2686.pdf`)
- Sumi et al. (2025) Front. Neurosci. 19:1570783 §2.8 (`docs/private/damage.pdf`)

原文が決めているのは次の 5 点だけで、実装はこれを 1 対 1 に写している。

| # | 原文 | 実装での対応 |
|---|---|---|
| 1 | ニューロンは 2 次元基板上にランダム配置 | `space: area_uniform` (結合モデルの外) |
| 2 | 軸索は soma から**ランダムな向き**で出る | `theta[0] ~ U(0, 2π)` |
| 3 | 全長 ℓ は平均 1.1 mm の **Rayleigh 分布** | `rng.rayleigh(mean / sqrt(pi/2), N)` |
| 4 | 長さ 0.1 mm のセグメントを連ね、各ステップで向きが `p(θi) ∝ exp(−(θi−θi−1)²/2σθ²)`、σθ = 5.73° = 0.1 rad だけ揺らぐ ("pseudo-straight") | `theta += N(0, angle_sigma)` の角度ランダムウォーク |
| 5 | 樹状突起は soma 中心・半径 150 µm の**円**。ニューロン i の軸索が j の樹状突起円に**侵入したら 20% の確率で** i→j を張る | セグメント-soma 距離 ≤ `dendrite_radius`、侵入 1 回につき `connection_prob` で 1 回抽選 |

原文の (4) は「セグメント i の**絶対**方向が、1 つ前の絶対方向を中心とするガウス分布」で
あって、増分に対する分布ではない。したがって `theta` に正規乱数を足し込む素直な
ランダムウォークが原文どおりの実装になる (§1 の持続長の議論はここから来る)。

**確率を距離の関数として与えるモデルではない**のが要点。`gaussian_distance` 系は
「距離 d のペアが結合する確率 f(d)」を仮定するが、こちらは軸索という 1 次元の物体を
実際に空間に置き、幾何的に交差したものだけを結合にする。したがって

- 結合確率は距離だけの関数にならない (軸索が「どっちを向いて出たか」に依る)
- 壁・通路・空隙といった**領域の形が、そのまま結合構造に効く**
- 出次数は軸索長に比例し、入次数は周囲の軸索密度で決まる (両者は別の分布になる)

### A.2 実装の 3 相構成

```
  generate_sparse()
    ├─ 第 1 相  _grow_axons(p)          軸索を折れ線として伸ばす      → セグメント列
    ├─ 第 2 相  KD-tree + 幾何判定       樹状突起円との接触を全部拾う  → 接触セグメント列
    └─ 第 3 相  侵入への畳み込み + 抽選   1 侵入 1 ドローで結合を決める → COO (rows, cols)
```

第 1 相は**乱数を引きながら逐次**進む (向きが前のステップに依存する)。
第 2 相は**乱数を一切引かない純粋な幾何**。第 3 相は乱数を 1 回だけ、
`rng.random(侵入の総数)` としてまとめて引く。この分離のおかげで、性能都合の
ブロック分割 (`_SEGMENT_BLOCK`) や KD-tree の返す順序が実現に影響しない (§F)。

### A.3 パラメータ

`_params()` が YAML から読み、既定値と検証を与える。

| 名前 | 既定 | 単位 | 意味 | 検証 |
|---|---|---|---|---|
| `mean_axon_length` | 1100.0 | µm | Rayleigh 分布の**平均** (scale ではない) | > 0 |
| `segment_length` | 100.0 | µm | 1 ステップの弧長 Δs。**離散化の刻み** (§1) | > 0 |
| `angle_sigma` | 0.1 | rad | 1 ステップあたりの向きの揺らぎ σθ | > 0 |
| `dendrite_radius` | 150.0 | µm | 樹状突起円の半径。全ニューロン共通 | > 0 |
| `connection_prob` | 0.2 | — | **侵入 1 回**あたりの結合確率 | [0, 1] |
| `boundary` | `deflect` | — | 壁に当たったときの挙動: `deflect` / `reflect` / `stop` | 3 値のいずれか |
| `allow_self_connections` | False | — | 自分の樹状突起円への侵入を数えるか | — |

`_params()` は `SimpleNamespace` を返すだけで状態を持たない。**呼ばれるたびに YAML を
読み直す**ので、`_grow_axons(p)` のように引数で引き回している (実装のどこかで
`self.config` を直接読むと検証を素通りする)。

---

## B. 外部インタフェース

### B.1 `NetworkBuilder` から見た呼ばれ方

```
area   = areaClass(...)                       # 乱数を引かない (CLAUDE.md 注記 12)
space  = spaceClass(..., rng, area=area)      # soma 座標。rng を消費する
conn   = AxonGrowthTopology(cfg, N, coords, rng, layout=..., area=area)
rows, cols = conn.generate_sparse()           # ← ここ
weight = weightClass(..., rng, ...)           # 続きの rng を消費
delay  = delayClass(..., rng, ...)
```

`rng` は `NetworkBuilder.rng` **ただ 1 本**を空間 → 結合 → 重み → 遅延の順に共有する。
つまり **`axon_growth` が引く乱数の個数が変われば、重みと遅延の実現もまるごとずれる**。
Δs を変えると同じ seed でも別ネットワークになる (§4 の注意) のは、直接には
「セグメント数が変わる → 第 1 相の `normal()` の消費数が変わる」からで、
第 3 相の抽選数の変化がさらにその後段をずらす。

必要な入力は `coords` (2 次元以上) と `area` の 2 つで、どちらも欠けると
`_validate_inputs()` が明示的に落とす。3 次元座標を渡しても**先頭 2 列しか見ない**
(`coords[:, :2]`) — このモデルは平面培養の再現なので、z は落として構わない。

### B.2 出力は 3 つ

| 出力 | 型 | 経路 | 用途 |
|---|---|---|---|
| 結合 | `(rows, cols)` int32 COO、`(pre, post)` 昇順 | `generate_sparse()` の戻り値 | GeNN 登録、解析すべて |
| 軸索の記録 | `AxonGeometry` | `axon_geometry()` (生成前は `None`) | 描画 (`plotting/network.py::axon_network`)、損傷実験。`axon_geometry.npz` に保存 |
| 軸索長 | `{"axon_length": (N,) float}` | `describe_axes()` | `layout.order_by("axon_length")`、軸索長 vs 出次数の解析 |

`generate()` (密) は `generate_sparse()` を呼んで `mask[rows, cols] = 1` するだけなので、
**密と疎は定義上一致する**。`GaussianDistanceTypeTopology` が両経路で乱数列を手で
揃えているのとは対照的で、こちらは揃える対象がそもそも 1 つしかない。

---

## C. 第 1 相: 軸索を伸ばす (`_grow_axons`)

### C.1 全長とステップ数

```python
lengths = rng.rayleigh(mean_axon_length / sqrt(pi/2), N)   # 平均 = scale*sqrt(pi/2)
n_seg   = floor(lengths / segment_length)                  # 0 も許す
theta   = rng.uniform(0, 2*pi, N)
alive   = n_seg > 0
```

`n_seg == 0` の軸索は 1 本もセグメントを持たない = **出力を持たないニューロン**になる。
論文もそれを排除していないので、実装も許す (3 mm 円板 N=2827 で 15 本、0.5%)。
`floor` による切り捨ての平均 Δs/2 が §3(a) のバイアスの正体。

### C.2 ステップの外側ループ: 「ステップ index で反復し、ニューロン方向にベクトル化」

境界処理があるため向きは逐次依存で、`cumsum` にはできない。そこで

```python
for k in range(max_steps):                 # ステップ index (平均 10 回、最大でも数十)
    act = nonzero(alive & (k < n_seg))[0]  # このステップでまだ伸びているニューロン (ID 昇順)
    theta[act] += rng.normal(0, angle_sigma, act.size)   # ← 乱数はここだけ
    ...
```

と、**外側がステップ・内側 (ベクトル) がニューロン**になっている。`act` は
`np.nonzero` なのでグローバル ID 昇順に固定され、正規乱数の割り当ても ID 昇順に決まる
(ここが崩れると同じ seed で別のネットワークになる)。

### C.3 1 ステップの内側: advance-and-slide

1 ステップは「弧長 Δs ぶん進む」であって「Δs の直線を 1 本引く」ではない。壁に当たった
ら、**当たった場所まで進み、そこでの法線で向きを変え、残りの長さを壁沿いに使う**。
擬似コードにすると:

```
base      = 現在位置
u         = cos/sin(theta)          # 進行方向 (単位ベクトル)
remaining = segment_length          # このステップで使い切るべき弧長
on_wall   = False
advanced  = False

repeat (最大 _MAX_SLIDE_ITERS = 256 回):
    if remaining <= eps: break                       # eps = 1e-6 * Δs
    cand = base + remaining * u                      # 候補終点

    if on_wall:                                      # C.4 曲面への引き戻し
        d = area.sdf(cand)
        if d > 0: cand ← cand - d * area.normal(cand)  (移動量は remaining で頭打ち)

    if area.segment_inside(base, cand):              # 壁に当たらない
        セグメント (base → cand) を出力
        remaining -= |cand - base|                   # 引き戻したぶん端数が残る
        base = cand;  advanced = True;  break 相当 (remaining ≈ 0)
    else:                                            # 壁に当たる
        t = area.first_exit(base, cand)              # 収まる最大の割合
        x = base + t * (cand - base)
        if t * remaining > eps:                      # 進めたぶんだけセグメントを出す
            セグメント (base → x) を出力
            base = x;  remaining *= (1 - t);  advanced = True
        if boundary == "stop": このステップを終了
        on_wall = True
        n  = area.normal(x)                          # ★ 候補終点ではなく**交点**での法線
        u  = u - (u·n) n        (deflect)            # 壁に食い込む成分だけ消す
           = u - 2(u·n) n       (reflect)
        u ← u / |u|   (|u| ≈ 0 なら n を 90° 回した向きへ逃がす)
        if 進めず、かつ向きも変わらなかった: このステップを終了 (状態が不変 = 以降 no-op)
```

設計上の分かれ目は次の 3 つ。

1. **法線を取るのは「候補終点」ではなく「交点」。** 候補終点で取ると、壁に着く前に
   向きが変わってしまう。矩形の壁で旧実装は折れ角の中央値が 90° になり、軌跡が階段状に
   なっていた (`test_turns_are_gentle_in_a_smooth_area` がこれを pin する)。
2. **残りの長さを引き継ぐ (`remaining *= 1 - t`)。** 壁に当たっても 1 ステップの弧長は
   保存される — 曲がるだけで速度は落ちない。旧実装は `base` から**全長**を引き直して
   いたため、曲面 (円) の接線が領域外へ出て「偏向したのに外」となり、半径 400 µm の円で
   全長の 28% が失われていた (`test_convex_walls_never_truncate_an_axon`)。
3. **1 ステップが複数セグメントに割れる。** だから `arc_length` は
   `本数 × segment_length` では**ない** (§C.6)。逆に 1 本のセグメントが Δs を超えることは
   ない (`test_a_segment_never_exceeds_the_step_length`)。

### C.4 曲面への引き戻し (`on_wall` のときだけ走る補正)

壁の上に立って接線方向へ一歩出すと、**壁が曲がっていれば即座に外へ出る** (円の接線は
接点以外すべて外側)。そのままだと `first_exit` が t ≈ 0 を返し続けて軸索が貼り付く。
そこで候補点の SDF を見て、正なら法線方向へ SDF ぶん引き戻す — Newton 1 ステップの
表面への射影で、結果として「曲面に沿って滑る」動きになる。直線の壁なら `sdf == 0` の
ままなので何も起きない。

2 つの安全弁が付いている。

- **引き戻しの頭打ち。** `CompositeArea` の SDF は union = min なので真の距離の**下界**
  でしかなく、引き戻しでかえって遠のくことがある。移動量が `remaining` を超えないよう
  クランプする。
- **`on_wall` を落とさない。** 引き戻して着地した点は壁の上なので、次の反復も
  「壁沿いの一歩」として扱う (`on_wall[g] = snapped[clear]`)。ここを False にすると
  次の反復で接線がまた外へ出て、1 反復ぶん空回りする。

引き戻したぶん弦は名目より少し短い (R=1500 µm の円・Δs=100 µm で 0.02 µm)。その端数は
`remaining` に残って次の反復で使われるので、弧長としては失われない。

### C.5 打ち切り条件は 3 種類あり、意味が違う

| 名前 | 条件 | 効き方 | 意味 |
|---|---|---|---|
| `frozen` | 進めず (`moved == False`) かつ向きも変わらない (`u_new·u1 > 1−1e−12`) | **そのステップ**を終了 | (位置, 向き, 残り) が完全に不変 = 以降何回まわしても同じ。多角形の角 (射影が冪等) がここに落ちる。**1 回目の偏向は向きが変わるので frozen にならない** — 正面衝突した軸索が次の反復で壁沿いに進む経路は残る |
| `stuck` | そのステップで**一度も** eps 以上進めなかった (`~advanced`) | **軸索を永久に**終了 (`alive = False`) | 本物の袋小路。「数 µm 足りなかった」を袋小路と同一視すると全長が目減りするので、少しでも進めたら生かす |
| `boundary: stop` | 壁に当たった | そのステップを終了 | 壁**まで**伸ばして止まる。ステップごと捨てるわけではない |

`stop` の挙動は誤解しやすいので明示しておく。`stop` は `live` (ステップ内のフラグ) しか
落とさないので、壁に着いた軸索は次のステップで**新しい角度を引き直して**再挑戦する。
たいていは新しい向きも外を向くので `t = 0` → `~advanced` → `stuck` で死ぬが、内側を
向けば伸長が再開する (実測: 半径 300 µm の円板・200 本で再開したのは 1 本、0.5%。
平均弧長は 242 µm で、`deflect` の 1030 µm に対して 1/4 以下)。
`_MAX_SLIDE_ITERS` は循環に対する保険にすぎず、正しい停止条件は `frozen` の側にある
(実測の必要回数: 円板 3、`hierarchical_modular_grid` 105 → §3(d))。

### C.6 正準順序と `arc_length`

セグメントは「ステップごと」に積まれるので、生成直後は `(step, owner)` 順に並んでいる。
これを `lexsort` で **`(owner, step, 壁で割れた順)`** に並べ替える。並べ替えキーは
`k * (_MAX_SLIDE_ITERS + 2) + sub` と 1 本の整数に畳んである (この桁取りが
`(k, sub)` の辞書順と一致する)。この順序は 3 つのことを同時に保証する:

- 折れ線としての順序 (`seg_start[j+1] == seg_end[j]` が同一 owner 内で成り立つ)
- `AxonGeometry.offsets` が単なる累積和で作れること
- **第 3 相の「連続セグメント index = 同じ侵入」という判定が成立すること** (§E.1)

`arc_length` は `bincount(seg_owner, weights=|seg_end − seg_start|)`、すなわち
**実際の折れ線長の総和**。壁沿いに割れたセグメントは Δs より短いので、本数からは
復元できない。実測 (3 mm 円板 N=2827、Δs=100): セグメント総数 41,789、1 本あたり
14.78 本に対し平均弧長 1029.6 µm — 1 セグメントの平均長は 69.7 µm で、Δs=100 を大きく
下回る。差は壁沿いの分割ぶん。

---

## D. 第 2 相: 樹状突起円との接触を拾う (乱数を引かない)

セグメント列が確定したら、あとは純粋な幾何。セグメントを `_SEGMENT_BLOCK = 8192` 本ずつ
処理する (メモリのため。結果には影響しない → §E.2, §F)。

### D.1 KD-tree で候補を絞る

セグメントと soma の距離が `dendrite_radius` 以下であるための**必要条件**は、
セグメント中点から soma までが `Δs/2 + dendrite_radius` 以内であること。これを
`cKDTree(soma).query_ball_point(midpoints, query_r)` で一括に取る。

- 壁で割れた短いセグメントに対しても `query_r` は名目の Δs で計算する。半径が過大に
  なるだけで、候補は真の集合の**上位集合**なので取りこぼさない。
- **`query_ball_point` が返す順序は未定義**。そのままだと再現性が壊れるので、直後に
  `lexsort((flat_j, seg_idx))` で固定する。

### D.2 点-線分の厳密距離

候補ペア (セグメント, soma) について、線分上の最近接点を閉形式で求める:

```python
t    = clip( (ap·ab) / (ab·ab), 0, 1 )      # 線分上のパラメータ
perp = ap - t * ab
hit  = |perp|² <= dendrite_radius²
```

`t` は後で接触点を復元するのに使うので、以降のフィルタでも常に 3 本 (`seg_idx`,
`flat_j`, `t`) の対応を崩さずに絞る。**接触点は「円に入った点」ではなく「最近接点」**で
あることに注意 (`AxonGeometry.contact_t` の意味もこれ)。

### D.3 自己結合の除去は**抽選より前**

`own[seg_idx] != flat_j` で落とす。密経路が疎経路に委譲する構成なので、
`GaussianDistanceTypeTopology` のように「対角のドローを消費してから捨てる」必要がない。

### D.4 見通し判定 (line of sight) — ユークリッド距離だけでは足りない

樹状突起も領域の外へは出られない。距離だけで判定すると、**半径 150 µm より狭い空隙の
向こう側にある soma に届いてしまう** (`modular_4` では孤立した円とブリッジの間が 40 µm)。
そこで接触点から soma までの線分が領域内に収まることを要求する:

```python
hit[hit] &= area.segment_inside(contact[hit], soma[flat_j[hit]])
```

これも**抽選より前**に落とす。「そもそも接触していない」ものに乱数を消費させないため
(自己結合と同じ方針)。この判定があるので、非連結な部分領域の間には結合が 1 本もできない
(`test_no_synapses_between_isolated_components`)。

---

## E. 第 3 相: 侵入への畳み込みと抽選

### E.1 「侵入 (invasion)」の定義

> 侵入 = 同じ (軸索 owner, 相手 post) に対して、**連続したセグメント index** で触れて
> いる一続き。

円を一度出てから入り直せば index が飛ぶので、**別の侵入として数える**。判定は
ソート済み配列の隣接比較 1 発:

```python
order = lexsort((g_seg, flat_j, g_own))      # (owner, post, セグメント index) 昇順
start_of_run[1:] = (g_own[1:] != g_own[:-1])
                 | (flat_j[1:] != flat_j[:-1])
                 | (g_seg[1:]  != g_seg[:-1] + 1)   # ← index が飛んだら新しい侵入
```

これが成立するのは §C.6 でセグメントを `(owner, step, 割れた順)` に並べ替えてあるから。
並べ替え前の `(step, owner)` 順では「index が連続 = 折れ線上で隣」が成り立たない。

**なぜ侵入単位でなければならないか。** セグメントごとに抽選すると、同じ 1 回の侵入を
刻んだ本数だけ試行することになり、実効確率が `1 − (1−p)^(刻み数)` に化ける。半径 150 µm
の円は Δs=100 なら ~3 セグメントぶんの幅があるので出次数が約 2.2 倍、Δs=25 なら
約 3.9 倍に膨らむ。つまり**離散化が結合密度を決めてしまう**。侵入単位なら Δs を 50 倍
変えても `p_within` は 0.196〜0.215 で動かない (§2、`test_degree_barely_depends_on_segment_length`)。

### E.2 ブロック境界の `carry`

侵入がブロックの切れ目をまたぐと、後半が「新しい侵入」に見えてしまう。そこで

- 各ブロックの**末尾セグメント**に触れている侵入を `owner * N + post` に符号化して
  `carry` に持ち越す
- 次のブロックで、**先頭セグメント**から始まる run のうち `carry` に含まれるものは
  `start_of_run = False` に落とす (前ブロックで抽選済み)
- `carry` はループの先頭で必ず空にしてから作り直す。途中の `continue` で抜けたブロックが
  古い `carry` を持ち越さないため

結果として `_SEGMENT_BLOCK` は**純粋な性能ノブ**になる。値を変えても同じ seed で
ビット単位に同じネットワークが出る (`test_block_size_does_not_change_the_network`)。

### E.3 正準順序 → 1 侵入 1 ドロー

全ブロックの侵入を集め終えてから、**まとめて 1 回**引く:

```python
order = lexsort((c_seg, post, pre))          # (pre, post, 侵入開始セグメント) 昇順
accept = rng.random(pre.size) < connection_prob
```

ブロック順のまま引くと `_SEGMENT_BLOCK` を変えただけで乱数の割り当てが変わってしまう。
正準順に並べ直してから引くことで、ブロック分割は実現に影響しなくなる。
**`self.rng` を消費するのはこの 1 行だけ**なのが第 3 相の設計。

### E.4 重複の畳み込みと接触点の確定

同じ相手に 2 回侵入して 2 回とも当たれば同じペアが 2 行できる。並びは既に
`(pre, post, seg)` 昇順なので、隣接比較で先頭だけ残せばよい:

```python
keep[1:] = (pre[1:] != pre[:-1]) | (post[1:] != post[:-1])
```

残るのは**最初に当たった侵入**なので、`AxonGeometry.contact_seg` / `contact_t` は
「そのシナプスを作った最も早いセグメントと、その上の位置」になる
(`test_contact_is_the_earliest_touching_segment`)。

結合行列は原文どおり **binary** (`A = {a_ij}`)。同じペアに何回侵入しても重みは増えない
— 重みは後段の weight コンポーネントが決める。

### E.5 実測 (3 mm 円板、N=2827、Δs=100、seed=42)

| 量 | 値 |
|---|---|
| セグメント総数 | 41,789 (14.78 / 軸索) |
| 平均弧長 | 1029.6 µm |
| 軸索を持たないニューロン (`n_seg == 0`) | 15 本 (0.5%) |
| **侵入の総数 = 抽選のドロー数** | 357,553 (126.5 / 軸索) |
| 採択 (× 0.2) | 71,510 |
| 重複を畳んだ後のシナプス数 | **71,328** (k_out = 25.23) |
| 畳み込みで消えたぶん | 182 (0.26%) |
| `generate_sparse()` の実時間 | 0.56 s |

重複が 0.26% しかないのは、同じ相手の円へ 2 回侵入する軸索が稀 (軸索がほぼ直線なので、
一度抜けた円へ戻ってこない) だから。壁で何度も曲がる形状ではこの比率は上がる。

---

## F. 乱数消費の全順序 (再現性の契約)

`self.rng` から引くのは以下がすべてで、この順序と個数が同じなら結果は同一。

| 順 | 呼び出し | 個数 | 相 |
|---|---|---|---|
| 1 | `rng.rayleigh(scale, N)` | N | C.1 全長 |
| 2 | `rng.uniform(0, 2π, N)` | N | C.1 初期方向 |
| 3 | `rng.normal(0, σ, act.size)` を `k = 0..max_steps−1` で | Σ_k \|act_k\| | C.2 各ステップの向き |
| 4 | `rng.random(侵入の総数)` | 侵入数 | E.3 抽選 |

**境界処理は乱数を引かない**ので、壁の形が変わっても 3 の個数は `n_seg` と
`stuck` 判定だけで決まる。逆に言うと、壁で軸索が早く死ねば 3 の消費が減り、その先
(重み・遅延) までずれる。

依存関係を整理すると:

- 消費数が変わる要因: `segment_length` (→ `n_seg`)、`mean_axon_length`、`N`、
  エリアの形 (→ `stuck` と侵入数)、`dendrite_radius` (→ 侵入数)
- 消費数が変わらない要因: `_SEGMENT_BLOCK`、`connection_prob` (個数は同じで閾値だけ動く)、
  `_MAX_SLIDE_ITERS` (binding していない限り)

---

## G. 計算量

| 相 | 支配項 | 備考 |
|---|---|---|
| C (伸長) | O(総セグメント数 × SDF 評価) | SDF は `segment_inside` で最大 `_SEGMENT_SAMPLES`=16 点、`first_exit` はさらに `_BISECT_STEPS`=12 回。壁に当たった行だけ後者を払う |
| D (接触) | O(総セグメント数 × 1 セグメントあたり候補数) | 候補数 ≈ 密度 × π(Δs/2 + r)²。3 mm 円板で ~50 |
| E (抽選) | O(侵入数 log 侵入数) | lexsort が支配 |

`CompositeArea` は 1 回の SDF 評価で**全 part を舐める**ので、part 数が線形に効く
(`hierarchical_modular_grid` は 36 part、`segment_inside` 1 回で 36×16 回の距離計算)。
これが §3(e) のビルド時間の差の主因。

---

## H. 論文との差分・既知の限界

1. **2 次元のみ。** `coords[:, :2]` しか見ない。3 次元培養は表現できない。
2. **樹状突起は全ニューロン共通の固定半径の円。** 原文も 150 µm 固定なので一致するが、
   ニューロンごとに半径を振る拡張は入っていない。
3. **壁は確率的に透過しない。** Sumi et al. の PDMS バンドは「下→上 5%、上→下 50% で
   透過し、失敗したら壁沿いに偏向」だが、実装の壁は `BaseArea` の境界 = 透過率 0%。
   モジュール構造は**幾何 (ブリッジ)** で表現しており、非対称な透過確率は再現できない。
   この非対称性が要る実験では `axon_growth` を拡張する必要がある。
4. **軸索はシナプスを作っても止まらないし、分岐もしない。** 1 本の折れ線が全長ぶん
   伸びきり、通過したすべての樹状突起円が抽選対象になる (原文どおり)。
5. **soma は点。** 軸索は soma の座標そのものから出る (initial segment / soma 半径なし)。
6. **E/I は結合の有無に一切関与しない。** `polarity` は重みの符号を決めるだけで、
   幾何は E も I も同じ。原文も同様。
7. **角の不動点。** 非平滑な点で `normal()` (中心差分) は 2 面の**二等分線**を返すため、
   その接線はどちらの面の接線でもなく、`deflect` が turn を見つけられない
   (CLAUDE.md 注記 16c)。Δs=100 で 1.3% の軸索が失われ、Δs=10 では 0%。
8. **`contains()` の許容差 `_CONTAINS_TOL = 1e-9` に依存している** (注記 16b)。
   0 にすると 45° の面で軸索が壁に貼り付いて止まる (26% の軸索が停止)。
9. **細い通路は Δs 次第で「存在しないのと同じ」になる** (§3(c))。
10. **損傷 (線で切る・軸索を切られたニューロンを死なせる) は実装していない。**
    `AxonGeometry` はそれを後から計算できるようにするための記録。
11. **旧実装との互換性はない。** 境界処理と抽選単位が変わった時点で、同じ seed でも
    別のネットワークになる (CLAUDE.md 注記 20: `axon_growth_modular` で 34,319 → 14,243
    シナプス)。互換フラグは用意していない。

---

## I. どのテストが何を pin しているか

`test/test_axon_growth.py`。仕様を変えるときは、まずここのどれが落ちるかを見る。

| クラス / テスト | pin している性質 |
|---|---|
| `TestAxonGrowth::test_axons_never_leave_the_area` | 全セグメントが領域内 |
| `…::test_axon_segments_never_cross_a_void` | 端点だけでなく**線分**が領域内 (§D.4 の前提) |
| `…::test_no_synapses_between_isolated_components` | 見通し判定が効いていること |
| `…::test_dense_and_sparse_agree` | `generate()` が `generate_sparse()` の散布であること |
| `…::test_reproducible_from_seed` | §F の乱数契約 |
| `TestFirstExit::test_prefix_is_always_inside` | `first_exit` が `segment_inside` 自身を二分探索していること (点サンプルで交点を求めると後から「外」と判定されうる) |
| `TestAdvanceAndSlide::test_convex_walls_never_truncate_an_axon` | 残り長さの引き継ぎ (§C.3-2) |
| `…::test_a_segment_never_exceeds_the_step_length` | 1 セグメント ≤ Δs |
| `…::test_turns_are_gentle_in_a_smooth_area` | 法線を**交点**で取ること (§C.3-1) |
| `…::test_stop_mode_ends_at_the_wall` | `stop` が壁**まで**伸ばすこと (§C.5) |
| `TestInvasionLottery::test_degree_matches_probability_times_invasions` | 1 侵入 1 ドロー |
| `…::test_degree_barely_depends_on_segment_length` | 侵入ルールの Δs 不変性 (§E.1) |
| `…::test_block_size_does_not_change_the_network` | `_SEGMENT_BLOCK` が性能ノブでしかないこと (§E.2) |
| `TestAxonGeometry::test_contact_is_the_earliest_touching_segment` | 重複畳み込みが「最初の侵入」を残すこと (§E.4) |
| `TestAreaRngContract::test_construction_consumes_no_randomness` | area が rng を引かないこと (CLAUDE.md 注記 12) |
