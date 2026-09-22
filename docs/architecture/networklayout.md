# NetworkLayout — 正準グローバルID空間と任意軸

`NetworkLayout` (`src/core/layout.py`) は、SNN のニューロンを **1 本の正準グローバルID空間
`0..N-1`** の上で管理し、そこに **任意個数の「軸」** を貼るデータ構造。

**特権的な軸は無い。** 軸とは各グローバルIDを値（カテゴリ名またはソート用の数値）へ写す
長さ N のラベル配列にすぎず、分類とソートはすべてその上で総称的に行われる。層もモジュールも
E/I も、この意味では対等な軸である。

| 知りたいこと | 見る場所 |
|---|---|
| 要約と設計方針 | ルート `CLAUDE.md` の不変条件 §2、`src/core/CLAUDE.md` §2（自動ロード） |
| **正の仕様** | `src/core/layout.py` のモジュール docstring と各メソッドの docstring |
| API の一覧・具体例・永続化・検証記録 | **この文書** |
| コンポーネントが軸を宣言する側の話 | `src/models/network/CLAUDE.md` §1 |
| ID 割り当てが変わった非互換 | `docs/technical/compat_breaks.md` |

> **この文書は `src/core/CLAUDE.md` §2 の詳細版。** 同じ話を繰り返さないよう、
> 「なぜ任意軸なのか」「assignment の既定がなぜ `random` か」はそちらにあり、ここは
> **一覧と具体例**を持つ。

---

## 1. 軸は 3 種類ある

| 種類 | 軸名 | 誰が貼るか | 永続化 |
|---|---|---|---|
| **変換表の実体** | `population` | `__init__`（必ず） | しない（config から再導出） |
| **config 由来のメタデータ** | `mode` / `polarity` | `from_config()` | しない（同上） |
| **外部軸** | `layer` / `module` / `axon_length` / … | コンポーネントの `describe_axes()` → `NetworkBuilder._inject_axes()` → `add_axis()` | **する**（`layout_axes.npz`） |

前 2 者はまとめて `_AUTO_AXES = ("population", "mode", "polarity")` と呼ばれ、
**「config だけから再導出できる軸」の定義そのもの**。だから `axes_to_dict()` が除外し、
復元時は `from_config()` が作り直す。config 由来の軸を足すならこのリストにも追記する。

### `mode` と `polarity` は別物

- **`polarity`** … `excitatory` / `inhibitory`。**E/I で切りたい側（結合・遅延・可視化・
  ブロック統計）は必ずこちらを見る。**
- **`mode`** … 力学のモード（`RSexci` / `LTS` / …）。E/I とは独立。

`from_config()` は `neurons.<name>.polarity` が無ければ**推測せず raise する**。推測すると
「E として結合を張りながら GeNN には I を渡す」という食い違いが無言で成立するため。

### 外部軸は「そのコンポーネントが走った後」にしか存在しない

`NetworkLayout.from_config()` だけで得られるのは自動軸 3 本だけ。`layer` や `module` が
欲しければ、ビルドを通すか、保存済みの `layout_axes.npz` を読む（§3 永続化 / §4）。

---

## 2. population の割り当て (`assignment`)

`config.layout.assignment` が、population の所属をグローバルID空間へどう写すかを決める。

| 値 | 意味 |
|---|---|
| `random`（**既定**） | 全体比率に従って散布。`simulation.seed` から決定論的 |
| `sequential` | config の宣言順に連番（population = 連続ブロック） |

個数は `neurons.*.num` で固定なので、変わるのは**どのIDがどの population か**だけ。
**コネクトームそのものは変わらない。**

既定値は `config_manager.DEFAULT_ASSIGNMENT` にだけ存在する。`from_config()` は未指定を
補完せず raise し、`load_resolved()` も補完しない（保存済み config は記録なので、欠落は
検証エラー）。詳しい理由は `src/core/CLAUDE.md` §2。

**`random` の RNG は別ストリーム。** `seed + _ASSIGN_SEED_OFFSET (= 104729)` を使い、
行列値を生成する `NetworkBuilder.rng` との相関を避ける（`docs/technical/reproducibility.md`）。
`random` かつ `seed` 無しは raise する —— 割当が再現できないと、保存済み config からの復元が
実行時と違う E/I を解析側へ返して結果が静かに壊れるため。

### 不変条件: `global_indices` は常に昇順

連番でも散在でも、population のグローバルID集合は昇順で保たれる（`ids_by` が
`np.nonzero` で返すので構造的に保証される）。population ローカル index `i` は
「その集合の i 番目に小さいグローバルID」。**GeNN が発信ニューロン順にシナプスを格納する**
ため、`local_to_global` と NetworkBuilder の疎/密経路の一致がこれに依存している。

---

## 3. API リファレンス

### 構築

| API | 説明 |
|---|---|
| `NetworkLayout(population_labels, order=None)` | 長さ N の population ラベル配列から構築。config は知らない |
| `NetworkLayout.from_config(config)` | **config に触れる唯一の場所。** `population` / `mode` / `polarity` を貼る。GeNN 不要、seed から再現可能 |

### 軸の管理

| API | 戻り値 | 説明 |
|---|---|---|
| `add_axis(name, values, *, overwrite=False)` | — | 長さ N の軸を追加。自動軸の上書きには `overwrite=True` が必須。`dtype=object` は具体 dtype へ正規化される（npz を `allow_pickle=False` で扱うため） |
| `drop_axis(name)` | — | 軸を削除。**自動軸は削除できない** |
| `axes()` | `list[str]` | 定義済みの軸名 |
| `has_axis(name)` | `bool` | |

### カテゴライズ

| API | 戻り値 | 説明 |
|---|---|---|
| `labels(axis)` | `ndarray` (N,) | その軸のラベル配列（コピー） |
| `ids_by(axis)` | `{値: ndarray}` | カテゴリ値 → グローバルID（昇順）。キー順はグローバルID空間での**初出順** |
| `ids_where(**filters)` | `ndarray` | 多軸の交差。例 `ids_where(layer="IN2", polarity="excitatory")` |
| `values(axis, order="appearance")` | `list` | 軸に現れる値。`"appearance"`（初出順）か `"sorted"` |

> `ids_by()` が返す dict も ID 配列も**呼び出し側専用のコピー**。`population` 軸の配列は
> ローカルID↔グローバルID変換表そのものなので、実体を渡すと `ids += offset` のような
> 何気ない加工が変換表と昇順不変条件を静かに壊す。グループ分け自体はキャッシュされる。

### ソート

| API | 戻り値 | 説明 |
|---|---|---|
| `order_by(*axes, ascending=True)` | `ndarray` (N,) | 表示順に並べた permutation。**先に書いた軸ほど優先**、同値はグローバルID昇順で安定。文字列軸と数値軸を混在できる |
| `rank_by(*axes, ascending=True)` | `ndarray` (N,) | `order_by` の逆写像。「そのIDが何番目に来るか」。ラスターの y 座標のように**グローバルIDを表示位置へ写す**のに使う（`rank[ids]` の fancy-index で一括変換） |
| `sort_ids(ids, *axes, ascending=True)` | `ndarray` | 部分集合をその軸で並べ替える |

```python
order = layout.order_by("module", "polarity")
W_sorted = W[np.ix_(order, order)]      # module → E/I の順にブロック化した行列
```

### global ↔ local 変換（simulator / DataLoader が委譲する）

| API | 説明 |
|---|---|
| `total_neurons` | 総ニューロン数（プロパティ） |
| `global_indices(name)` | population のグローバルID集合（**昇順**、コピー） |
| `split_global_to_local(global_arr)` | `(N,)` → `{pop: 配列}`。キーは**正準 population 順** |
| `merge_local_to_global(local_dict, dtype=np.float32)` | `{pop: 配列}` → `(N,)` |
| `local_to_global(name, local_ids)` | population ローカル → グローバルID |

> **正準 population 順**は `_pop_order`（`from_config()` 経由なら `config.neurons` の宣言順）
> が唯一の定義で、`split_global_to_local()` の dict キー順がそれ。
> **`ids_by("population")` のキー順は初出順であって正準順ではない。**
> population 名の一覧が欲しいだけなら config を直接読むこと —— layout が config の写しを
> 再配布すると入手経路が 2 本になる。

### 永続化

| API | 説明 |
|---|---|
| `axes_to_dict()` | **外部軸だけ**を返す（`_AUTO_AXES` を除外） |
| `save_axes(path)` | 外部軸を npz へ。**保存すべき軸が無ければ何もせず `None`** |
| `load_axes(dict)` / `load_axes_file(path)` | 復元（既存の同名軸は上書き） |

置き場所（どの run の、どのファイル名か）は**呼び出し側が決める**。規約は
`src/core/output_manager.py` の `AXES_NAME = "layout_axes.npz"`。

---

## 4. データの流れ

```
config ──► NetworkLayout.from_config ──► 自動軸 3 本 (population / mode / polarity)
                                              │
NetworkBuilder._generate_global_matrices ─────┤  area → space → connection → weight → delay
   各コンポーネントの generate() 直後に         │  の順に構築し、その都度 _inject_axes()
   _inject_axes(component) ───────────────────┤  → describe_axes() の戻り値を add_axis()
                                              │  （後段は先行コンポーネントの軸を読める）
   出口で GlobalCOO に正規化 ──────────────────┤
NetworkBuilder._build_synapses ──► global_indices で np.ix_ 抽出 ──► GeNN 配線
simulator (push/pull/spikes) ──► split/merge/local_to_global で global↔local
layout.save_axes(...) ──► layout_axes.npz
                                              ▼
解析・描画 ──► from_config + load_axes_file ──► ids_by / ids_where / order_by
```

**GeNN 配線**: シナプス群 `src_pop → tgt_pop` は正準 COO から
`global_indices(src)` × `global_indices(tgt)` で切り出す。散在集合でも正しい。

**解析側の復元**は `config.yaml` から `from_config()` し、`layout_axes.npz` があれば
`load_axes_file()` する（各実験の `store/` が実装。`docs/architecture/runview_contract.md`）。
**外部軸を持たない config では npz が書かれないので「ファイルが無い」は正常**であり、
旧レイアウトの吸収ではない。

---

## 5. 具体例 (N=8、E=6/I=2、`assignment: random`、space が `module` 軸を宣言)

```
グローバルID |  0   1   2   3   4   5   6   7
population   | Exc Exc Inh Exc Exc Inh Exc Exc
polarity     |  E   E   I   E   E   I   E   E     ← from_config が貼る
module       | M0  M0  M1  M0  M1  M1  M0  M1     ← space の describe_axes が貼る
```

- **どちらの軸も連続を要求されない。** `module` が飛び飛びでも構わない
  （空間配置がそう出たならそれが正しい）。
- **GeNN population**: `Exc = [0,1,3,4,6,7]`（local 0..5）、`Inh = [2,5]`（local 0..1）。
  どちらも昇順。
- **切り出しはすべて fancy-index**:

```python
exc = layout.ids_by("polarity")["excitatory"]     # [0,1,3,4,6,7]
W[np.ix_(exc, exc)]                               # E→E ブロック

m0 = layout.ids_by("module")["M0"]                # [0,1,3,6]
layout.ids_where(module="M0", polarity="excitatory")   # [0,1,3,6] の交差

order = layout.order_by("module", "polarity")     # → [0,1,3,6 | 4,7 | 2,5]
                                                  #   module 優先、その中で E→I
```

- **スパイク**: GeNN の `Exc` local `2` が発火 → グローバルID = `global_indices("Exc")[2] = 3`
  → 「M0 かつ E」。**同じスパイクをどの軸でも集計できる。**

---

## 6. コンポーネントが軸を宣言する

`BaseSpace` / `BaseConnection` / `BaseWeight` / `BaseDelay` は任意フック
`describe_axes() -> {軸名: 長さ N の配列}` を持つ（既定は空 dict）。`NetworkBuilder` が
`generate()` / `generate_sparse()` の**直後**に呼ぶので、自分が今計算した座標・マスク・
重みから軸を導出してよい。

**`BaseArea` にはこのフックが無い。** 座標が存在する前に構築されるので、ニューロンごとに
言うことが何も無い。領域由来の軸は space モデルが宣言する
（`AreaUniformSpace` が `CompositeArea.part_of()` を呼んで `module` 軸として公開する）。

実際に使われている例:

| 宣言するもの | 軸 | 中身 |
|---|---|---|
| `AreaUniformSpace` / `Block2DSpace` | `module` | どの部分領域／ブロックに属するか |
| `C_elegansSpace` | `layer` | コネクトーム CSV の `Layer` 列 |
| `AxonGrowthTopology` | `axon_length` | 実際に伸びた軸索の折れ線長 [µm]（**数値軸**。`order_by("axon_length")` で並べ替えられる） |

```python
class MySpace(BaseSpace):
    def generate(self):
        self.coords = ...
        self._module_labels = ...        # generate の中で作っておく
        return self.coords

    def describe_axes(self):
        if self._module_labels is None:  # generate 前なら何も宣言しない
            return {}
        return {"module": self._module_labels}
```

軸の値は**全部文字列か全部数値に揃える**こと。混在すると `add_axis` の dtype 正規化が
raise する（`dtype=object` は npz に `allow_pickle=False` で保存できない）。

追加パラメータは `configs/components/*.yaml` に書けば `self.config.<key>` で読める
（Pydantic `extra='allow'`）。手順の詳細は `docs/architecture/adding_components.md`。

---

## 7. 再現性

- **`assignment=random` は seed → ネットワーク実現の依存を持ち込む。** ただし各 run が
  保存する `config.yaml` に seed と `assignment` の実値が入るので、
  `NetworkLayout.from_config()` だけで同一割当を決定論的に再現できる（replot はビルド不要）。
- 割当用 RNG は `seed + _ASSIGN_SEED_OFFSET` の**別ストリーム**。行列値生成用 RNG との
  相関を避ける。新しく seed 由来の RNG を足すときは必ず専用のオフセットを与えること
  （`docs/technical/reproducibility.md`）。
- **run を特定するのは (seed, backend) の組。** ただし backend が効くのは GeNN の
  デバイス RNG（= スパイク列）であって、**レイアウトと構造には影響しない。**

---

## 8. 検証記録（NetworkLayout 導入時の実測）

> 当時の 2 軸版に対する検証。軸が一般化された後も、ここで確かめた不変条件
> （昇順・round-trip・`pull_synapse` 順序）はそのまま効いている。

- Layout 単体: E/I 個数固定・散在ソート・`split`↔`merge` round-trip・seed 再現・
  `ids_where` 交差 OK。
- ブロック対角: `prob_based_block` + 4 層で層内密度 ≫ 層間密度（実測 0.095 vs 0.012）。
- **`pull_synapse` 順序**: `random` 散在割当でも初期重みが `global_weights` と厳密一致
  （max|diff| = 0）。
- 非回帰: 当時の `test/test_akita_soc.py` と `test/core/STDPtest/test_multi_spike.py` が
  全 PASS（現在のパスは `test/experiments/akita_soc/test_akita_soc.py` と
  `test/models/plasticity/STDPtest/test_multi_spike.py`）。

**NetworkLayout 専用の自動テストは無い。** 現在 layout の API を通しているのは
`test/utils/test_plotting_ordering.py`（`assignment=random` での `add_axis` /
`order_by` / 並べ替え軸の選択）、`test/utils/test_connection_probability.py`
（軸ごとの群分け）、`test/test_akita_model.py`（E/I ブロック分解）、
`test/experiments/develop/test_develop.py`（`from_config` + 外部軸の往復）。
