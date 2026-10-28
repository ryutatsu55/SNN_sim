# src/models/network — 領域・空間・結合・重み・遅延

`src/models/network/` を触る前に読む。プロジェクトの全体像はルート `README.md`。

---

## 1. コンポーネントの並びと `describe_axes()`

構築順は **area → space → connection → weight → delay**。

`BaseSpace` / `BaseConnection` / `BaseWeight` / `BaseDelay` はいずれも省略可能な
`describe_axes() -> {軸名: 長さN配列}` フックを持つ。NetworkBuilder は `generate()` の
直後にこれを呼び、結果を layout へ注入する。だからコンポーネントは
自分が今計算したもの（座標・結合・重み）から軸を導出できる。連鎖の後ろのコンポーネントは
前のコンポーネントの軸を読める。

`BaseArea` には **意図的に `describe_axes()` が無い** — 座標が存在する前に構築されるので、
ニューロンごとに言うことが何も無いため。領域由来の軸は space モデルが宣言する:
`AreaUniformSpace` が `CompositeArea.part_of()` を呼び、結果を **`module` 軸**として公開する。
これが、専用の space クラス無しでモジュール構造が `layout.ids_by("module")` に届く経路。

---

## 2. BaseArea — SDF で表される領域

`area.py` の `BaseArea` は **space より前に**構築され、space モデル（体細胞の配置先）と
connection モデル（軸索の閉じ込め先）の両方に手渡される。唯一の抽象メソッドは
`sdf(points)` — 境界までの符号付き距離、内部が負。他はすべてそこから導かれる:

- `contains()` = `sdf <= 0`
- `normal()` = 正規化した勾配
- `segment_inside(a, b)` = 線分上の点をサンプル
- 汎用の `sample()` = `bounds` 内の棄却サンプリング

決定的に重要なのは **SDF が合成できる**こと — `CompositeArea` は子を
`union = min` / `intersection = max` / `difference = max(a, -b)` で入れ子に合成する。
だから任意に複雑な領域（モジュール、トラック、穴）に新しいクラスは不要で、YAML プロファイル
だけで済む。単純な形状は `sample()` を解析的にオーバーライドする。

**area は乱数を引かない。** `BaseArea.__init__` は `rng` を受け取らず、乱数は `sample(n, rng)` の引数としてのみ入る
(`CompositeArea.area_um2` もモンテカルロではなく固定グリッドで数える)。乱数を消費するのは space → connection →
weight → delay を逐次に通る 1 つの `NetworkBuilder.rng` で、area はそれら全部より前に構築される。
`DiskArea.sample()` の draw 順は `RandomCircle2DSpace` と同じなので、`area: disk` + `space: area_uniform` は
同一シードで `space: random_circle_2d` をビット単位で再現する。汎用の `sample()` は棄却サンプリングなので、
draw 回数は領域の形 (と `soma_area` の広さ) で変わり、下流の重み・遅延の実現値まで変わる。

### 名前付きの複雑領域を作る手順

`beggs_plenz` と同じ2段階:

1. `configs/components/areas.yaml` に `op` / `parts` プロファイルを書く
2. **同じ名前で空の `CompositeArea` サブクラスを登録する**（`Modular4Area` が実例）

両方必要なのは、`profile_name` が YAML のキーであると同時にレジストリのキーでもあるため。

### 出荷済みプロファイル

手書きのモジュール型が2つあり、違いは**パーツが重なるかどうか**の一点だけ:

- `modular_4` — ディスク + 40 µm 届かない十字 → **5個の孤立成分**
- `modular_4_grid` — 正方形 + 重なるブリッジ → **1個の連結成分**、対角ショートカットの無い4-cycle

`modular_grid` (`ModularGridArea`) はその**パラメトリック版**: `side`, `bridge_width`,
`spacing`（**中心間距離**であってギャップではない）, `num_modules` を与えると自分でパーツを組む。

- グリッドは `num_modules` の最も正方形に近い厳密分解（4→2×2, 6→2×3, 9→3×3, 素数→1行）
- ブリッジは直交する隣接のみを繋ぐ
- `bridge_overlap`（既定 `min(bridge_width, side/2)`、0 は拒否）が和集合の連結性を保証する
- 中身はただの `CompositeArea` なので、`part_of`・`module` 軸・`segment_inside` は変わらない
- 角は `chamfer`（既定 `side/4`）で45°に面取りされ、**正方形 ∩ ひし形**として構築される。
  `DiamondArea`（`diamond`、厳密な L1 ボールの SDF）が存在する理由がこれで、回転長方形の
  プリミティブが要らない理由でもある。
- 面取りは1モジュールあたり `2·chamfer²` を削るので、面積が減り、N 固定なら密度が上がる。

`hierarchical_modular_grid` (`HierarchicalModularGridArea`) は**もう一段深い**: `modular_grid` の
2×2 クラスタを4つ 2×2 に並べ、各クラスタの**内側モジュール1つ**（全体中心に最も近いもの）だけを
直交隣接と繋ぐ。16 モジュール・20 ブリッジ（クラスタ内16 + クラスタ間4）で、これは 4×4 グリッドの
直交辺24本から外周に沿う4本を引いた数。

要点は形ではなく*どのモジュールが繋がるか*である: 既定の `block_spacing`（`2 * spacing`）では
16 モジュールは完全に一様な 4×4 グリッド上に載るので、`modular_grid` の `num_modules: 16` が
同一のフットプリントを占める — ただしそちらは全隣接を等しく配線し、階層が生まれない。
`block_spacing` を上げるとクラスタが離れる。制約は `block_spacing > spacing + side`
（隣接クラスタの内側モジュール同士が `block_spacing - spacing` 離れるため）。

- **グリッドは 4×4 固定** — `num_modules` / `rows` / `cols` は無視ではなく raise する。
  「内側モジュール」は 2×2 の 2×2 でしか定義できないため。
- パーツ名は `C{c}-M{m}` / `BC{c}-{m1}-{m2}` / `BX{c1}-{c2}`。`C0-M0` という形は load-bearing で、
  **クラスタごとのブロックにソートされる** — だから `order_by("module")` と
  `group_connection_probability(..., "module")` が追加の軸なしで階層の両レベルを見せる。
  （`soma_in_bridge: true` だと `BC*`/`BX*` ラベルが全 `C*` より**前**にソートされ、粗視化マスクで
  ブリッジが先頭の1ブロックに集まる。ここで `soma_in_bridge: false` を選ぶもう一つの理由。）
- コストはパーツ数に比例する: `sdf()` は毎回36パーツ全部を評価し、`segment_inside()` は
  1線分あたりそれを16回やるので、この領域上の `axon_growth` は `modular_grid` より体感で遅い。

### 領域は2つある（area と soma_area）

領域そのものは*軸索*が行ける場所。`soma_area` は*体細胞*を置ける部分領域で、
`AreaUniformSpace` が見るのはこちらだけ。

`BaseArea.soma_area` は既定で `self` を返すので、`CompositeArea` のパーツに
`allow_soma: false` が付かない限り両者は一致する。付いた場合 `soma_area` は残りのパーツの
和集合になり、そのパーツは**軸索は横断できるが細胞体は座らない**回廊になる。
`modular_grid` はこれを `soma_in_bridge: false`（既定 `true`）として包む。

`connectors.py` も `NetworkBuilder` も `soma_area` を読まない。builder は領域そのものを
space と connection の両方へ渡し続ける。だからこそ軸索の閉じ込め・壁スライド・樹状突起の
視線判定はどちらでもビット単位で同一になる。

分割が依存している細部2つ:

1. `AreaUniformSpace` は `part_of()` / `part_names` を**領域全体ではなく soma 領域で**読む。
   ブリッジの `bridge_overlap` 帯の内側では、全体領域の `part_of()` がブリッジを「より深い」
   パーツとして拾い、体細胞に `B0-1` というラベルを付けうるため。モジュールだけの和集合
   （`part_names` は親から継承されるので依然 `M0`, `M1`, …）で軸を取る必要がある。
2. 表示される密度は *soma 領域*の面積で割る。N 個のニューロンが居られるのはそこだけなので。
   `plot_area()` は両者が違うとき両方の数値を出す。
