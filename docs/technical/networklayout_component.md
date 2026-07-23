# NetworkLayout 仕様(構造層 × 種別の2軸レイアウト)

`NetworkLayout`(`src/core/layout.py`)は、SNN のニューロン集団を **1本の正準グローバルID空間**の
上で管理し、そこに **2つの直交する分割(軸)** を貼るための中核データ構造です。
旧来の `group_info`(dict)を置き換えます。

- **構造層 (layer)** … グローバルID空間で**連続**するブロック。重み/接続行列の**ブロック(対角)構造**を規定し、解析軸①になる。
- **種別 (type = E/I = GeNN population)** … 各グローバルIDに割り当てられるラベル。**GeNN の population** を決め、解析軸②になる。

この2軸は独立なので、「層でブロック行列を作りつつ、E/I はそれと無関係に散らばらせる」「解析では層でも E/I でも自由に切り分ける」が同時に成立します。

---

## 1. 基本モデル

### 正準グローバルID空間
- グローバルID `0..N-1` を**層順**に並べる(層1 = `[0,n1)`、層2 = `[n1,n1+n2)`, …)。
- 空間座標(N次元ベクトル)、結合マスク・重み・遅延(N×N 行列)は**すべてこの空間で生成**する。
  層が連続しているので、行列は自然に**層でブロック対角**になる。
- 行列は生成後もこの正準空間のまま保持し、**破壊的なシャッフルはしない**。

### 種別(population)の割り当て
- 各グローバルID を population(= `config.neurons` の各エントリ = E/I など)に割り当てる。
- population のメンバー(グローバルID集合)は**昇順ソート**で保持する。
  population ローカル index `i` は「メンバー集合の i 番目に小さいグローバルID」に対応する
  (GeNN の発信ニューロン順格納と一致し、`pull_synapse` の順序前提を保つ)。
- 割り当て方式は2種類:
  - `sequential`(既定) … config 宣言順に連番(population = 連続ブロック)。
  - `random` … 全体比率でランダムに散布(population = 散在するソート済み集合)。seed 決定論的。

### ラベル表
- 各グローバルIDに対して `population` / `mode`(excitatory/inhibitory 等)/ `layer` のラベルを持つ
  (長さ N の配列)。これにより任意の軸で柔軟にグルーピング(再構築)できる。

---

## 2. Config 仕様

`layout:` は **任意** のトップレベルセクション。**未指定なら従来挙動(sequential・単一層)** で、
既存 config は一切影響を受けません。

```yaml
neurons:                         # 従来通り = type(=population)定義。num は「個数」
  Layer_Exc: { type: akita_escape_lif, mode: excitatory, num: 80 }
  Layer_Inh: { type: akita_escape_lif, mode: inhibitory, num: 20 }

layout:                          # 任意
  assignment: random             # sequential(既定) | random
  layers:                        # 任意。構造層(連続ブロック)。num の合計 = 総ニューロン数
    - { name: L1, num: 25 }
    - { name: L2, num: 25 }
    - { name: L3, num: 25 }
    - { name: L4, num: 25 }
```

| キー | 意味 | 既定 |
|------|------|------|
| `layout.assignment` | population の割当方式 `sequential` / `random` | 未指定(= 空間プラン→`sequential`) |
| `layout.layers` | 構造層の一覧 `[{name, num}, ...]`。合計は総数と一致必須 | 未指定(= 空間プラン→単一層 `all`) |

**フィールド単位の優先順位**(層・割当それぞれ独立に解決):
1. 明示的な `config.layout` の値
2. **空間コンポーネントが提供するプラン**(`BaseSpace.describe_layout`。§6.1)
3. 既定(`assignment=sequential`、単一層 `all`)

これにより、層構造が空間そのものに宿るモデル(データ由来の層、モジュール分割など)は、
`layout:` を書かずに層と割当を**自身で宣言**できる(例: C. elegans コネクトーム)。

- **`assignment`** は「どのグローバルIDがどの population(E/I)か」を決める。
  個数は `neurons.*.num` で固定され、`random` では**どのIDがどのtypeか**だけがランダムになる。
- **`layers`** は構造(ブロック)の境界。連続。行列生成器と解析の両方がこれを参照する。
- スキーマ: `config_manager.py` の `LayoutConfig` / `LayerSpecConfig`(`AppConfig.layout: Optional`)。

---

## 3. データの流れ

```
config ──► NetworkLayout.from_config ──► layout(層境界 + population集合 + ラベル)
                                              │
NetworkBuilder._generate_global_matrices ─────┤  生成器に layout を渡す
   space/connection/weight/delay を            │  (層を見てブロック生成。E/I は非依存)
   正準グローバルID空間で N×N 生成             ▼
NetworkBuilder._build_synapses ──► population集合で np.ix_ 抽出 ──► GeNN 配線
simulator (push/pull/spikes) ──► layout.split/merge/local_to_global で global↔local 変換
解析/可視化 ──► layout.ids_by_layer / ids_by_mode / ids_where で軸を選んで再構築
```

- **生成器は E/I を見ない**。層構造(`layout.layers()`)だけを見てブロックを作る。
  E/I による符号/シナプスモデル/可塑性は、従来通り `synapses`(source population ごと)で処理する。
- **GeNN 配線**: シナプス群 `src_pop → tgt_pop` は、正準行列を
  `M[np.ix_(global_indices(src), global_indices(tgt))]` で切り出す(散在集合でも正しい)。

---

## 4. 具体例(N=8、2層、E=6/I=2、assignment=random)

- 層: `L1 = ID{0,1,2,3}`、`L2 = ID{4,5,6,7}`(連続)。
- 種別割当(seed 固定の抽選例): **E = {0,1,3,4,6,7}**、**I = {2,5}**(散在)。

| グローバルID | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| layer | L1 | L1 | L1 | L1 | L2 | L2 | L2 | L2 |
| type  | E  | E  | I  | E  | E  | I  | E  | E  |

- **GeNN population**: Exc = `[0,1,3,4,6,7]`(local 0..5)、Inh = `[2,5]`(local 0..1)。
- **行列生成**: 8×8 マスクを層境界でブロック対角に作る(`M[0:4,0:4]` 密、`M[4:8,4:8]` 密、層間は疎)。
- **Exc→Inh の配線**: `sub = M[np.ix_([0,1,3,4,6,7],[2,5])]`(6×2)の非ゼロ位置がそのまま
  Exc-local → Inh-local の接続になる。
- **解析(同じ行列を2通りに切る)**:
  - 層で: `W[0:4,0:4]`(L1→L1)など連続スライス。
  - E/I で: `W[np.ix_(exc,exc)]`(E→E)など散在集合。
- **スパイク**: GeNN の Exc-local `2` が発火 → グローバルID = `exc[2] = 3` → 「L1 かつ E」。
  同じスパイクを層でも E/I でも集計できる。

---

## 5. NetworkLayout API リファレンス

構築:
| API | 説明 |
|-----|------|
| `NetworkLayout.from_config(config)` | config から決定論的に構築(GeNN ビルド不要、seed から再現可能) |

population(type)軸:
| API | 戻り値 | 説明 |
|-----|--------|------|
| `total_neurons` | int | 総ニューロン数 |
| `names()` | list[str] | population 名(config 順) |
| `items()` | (name, PopulationSpec) | population を反復 |
| `global_indices(name)` | ndarray | population のグローバルID集合(昇順) |
| `num(name)` / `mode_of(name)` | int / str | 個数 / mode |

構造層(layer)軸:
| API | 戻り値 | 説明 |
|-----|--------|------|
| `layers()` / `layer_names()` | list[LayerSpec] / list[str] | 層一覧 |
| `layer_slice(name)` | slice | 層の連続区間 |
| `layer_ids(name)` | ndarray | 層のグローバルID(連続) |
| `ids_by_layer()` | {層名: ndarray} | 層単位の解析用 |

ラベル / 柔軟グルーピング:
| API | 戻り値 | 説明 |
|-----|--------|------|
| `labels()` | {"population","mode","layer": ndarray} | 各長さ N のラベル |
| `ids_where(**filters)` | ndarray | 例 `ids_where(layer="L2", mode="excitatory")` = 交差 |
| `ids_by_mode()` | {"excitatory","inhibitory": ndarray} | mode 接頭辞で E/I 分類(昇順) |

global↔local 変換(simulator / data loader が委譲):
| API | 説明 |
|-----|------|
| `split_global_to_local(global_arr)` | (N,) → {pop: 配列} |
| `merge_local_to_global(local_dict, dtype)` | {pop: 配列} → (N,) |
| `local_to_global(name, local_ids)` | population ローカル → グローバルID |

補助データクラス:
- `PopulationSpec(name, params, global_indices)` … `.num` / `.mode`。
- `LayerSpec(name, start, num)` … `.stop` / `.slice` / `.global_indices`。

---

## 6. 層対応の生成器(コンポーネント)

### 6.1 空間コンポーネントが層/割当を宣言する(`describe_layout`)

構造層と割当方式は `config.layout` に**明示**するほか、**空間コンポーネント側から供給**できます。
空間構造そのものが層に一致するモデル(データ由来の層、モジュール分割など)向けの仕組みです。

`BaseSpace` は任意フック `describe_layout(config, total_neurons)` を持ち、既定は `None`
(= `config.layout` / 既定にフォールバック)。オーバーライドすると `LayoutPlan(layers, assignment)`
を返せ、`NetworkLayout.from_config` がフィールド単位で採用します(明示的 `config.layout` が優先)。

```python
from src.core.layout import LayoutPlan

@SPATIAL_MODELS.register("my_space")
class MySpace(BaseSpace):
    @classmethod
    def describe_layout(cls, config, total_neurons):
        # config(= space.yaml のこのプロファイル)やデータファイルから決定論的に層を導く
        n = total_neurons // config.num_modules
        layers = [(f"M{i}", n) for i in range(config.num_modules)]
        return LayoutPlan(layers=layers, assignment="random")

    def generate(self):
        ...  # generate() と describe_layout() は同じ順序(正準グローバルID順)で整合させる
```

要件:
- **決定論的**に導出すること(GeNN ビルド無しの再構築性を保つ)。データファイルや自身の
  `config`(`space.yaml`)を参照してよいが、乱数実現に依存しない。
- `generate()` が返す座標の**行順 = 正準グローバルID順**であり、`describe_layout()` の層境界と
  一致させること(同じ入力を同じ順序で読む単一の読み込み口を設けると安全)。
- 空間モデルが未登録(モジュール未 import)の場合、`from_config` はプラン無しとして
  フォールバックする(スタンドアロン再構築の後方互換)。

**実装例: C. elegans**([space.py](../src/models/network/space.py) `C_elegansSpace`) …
`ordered_coords.csv` の `Layer` 列(IN1..IN4)の連続ランを構造層として宣言し、
E/I は `assignment="random"`(層と無相関に散布)。座標は同じ CSV 行順で返す。
結合/重みも同じ NodeID 順のコネクトーム行列を読む:
[connectors.py](../src/models/network/connectors.py) `C_elegansConnection`(`synapse_mask.csv`=chem∪elec)、
[weights.py](../src/models/network/weights.py) `C_elegansWeight`(chem を基礎に elec 接続を elec 重みで上書き、
`config.min_weight`/`max_weight` でクリップ)。符号は発信 population のシナプスモデルが付ける。

### 6.2 層対応の生成器

構造層は `NetworkLayout` で一元管理され、ブロック構造を作る生成器がそれを読みます。

現状で層対応済み:
- `prob_based_block`([connectors.py](../src/models/network/connectors.py)) …
  `layout.layers()` が**複数層**なら各層をモジュール(ブロック)として採用。
  層が無い/単一なら従来通り `num_modules` で等分。
- `block_2d`([space.py](../src/models/network/space.py)) …
  同様に層を矩形モジュールへ割り当てる(層が無ければ `num_modules` 均等分割)。

### 新しい層対応コンポーネントの書き方

生成器は基底クラスで `self.layout` を受け取っています(空間/接続/重み/遅延の4種)。
層を使いたい場合の定型:

```python
@CONNECTION_MODELS.register("my_block")
class MyBlock(BaseConnection):
    def generate(self):
        N = self.num_neurons
        layers = self.layout.layers() if self.layout is not None else None
        if layers is not None and len(layers) > 1:
            ranges = [(l.start, l.stop) for l in layers]   # 層 = ブロック
        else:
            ranges = [(0, N)]                              # フォールバック(単一ブロック)
        mask = np.zeros((N, N), dtype=np.int8)
        for s, e in ranges:
            mask[s:e, s:e] = (self.rng.random((e - s, e - s)) < self.config.p_within)
        return mask
```

- 生成器は**種別(E/I)を見ない**設計(方針)。E/I 依存が必要なら `self.layout.ids_by_mode()` /
  `labels()["mode"]` を参照して符号や分布を変えることも可能(現状は未使用)。
- 追加パラメータは `configs/components/*.yaml` に書けば `self.config.<key>` で読める
  (Pydantic `extra='allow'`)。

---

## 7. 後方互換と再現性

- **既存 config は不変**: `layout:` 未指定 → `assignment=sequential`・単一層 `all` = 従来の
  連続割当と完全一致(v1 と同一ネットワーク)。
- **`assignment=random`** は seed → ネットワーク実現の依存を(意図的に)持ち込む。ただし各 run が
  保存する `config.yaml` に seed と `layout` が入るため、`NetworkLayout.from_config` だけで
  同一割当を決定論的に再現できる(replot / 再解析はビルド不要)。
- random 割当用の RNG は `config.simulation.seed + オフセット` の**別ストリーム**を使い、
  行列値生成用 RNG との相関を避けている。

---

## 8. 検証(実測)

- Layout 単体: 連続層・E/I個数固定・散在ソート・split↔merge round-trip・seed 再現・`ids_where` 交差 OK。
- 既存 config 不変: `layout` 未指定で v1 と同一の連続割当。
- ブロック対角: `prob_based_block`+4層で 層内密度 ≫ 層間密度(実測 0.095 vs 0.012)。
- **`pull_synapse` 順序**: `random` 散在割当でも初期重みが `global_weights` と厳密一致
  (max|diff| = 0)。
- 非回帰: `pytest test/test_akita_soc.py`、`test/core/STDPtest/test_multi_spike.py` 全 PASS。
