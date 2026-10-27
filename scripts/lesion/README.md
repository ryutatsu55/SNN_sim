# lesion 実験

`scripts/develop/` が育てたネットワークを引き継ぎ、**特定の結合を構造的に除去**して、
切断からの回復を追う。ベースライン (切断直前) は親 run の記録をそのまま使う。

**`scripts/develop/` と同じ層構造** (`report/` `store/` `analysis/` `figures/`) に、
実験の進め方を決める `protocol/` (どこを切るか・何を止めるか・親を正しく引き継げたか) を
1 つ足したもの。違うのは「何を記録し、何を測り、どう描くか」だけで、層の役割も
解析・描画関数の引数も同じ。

---

## 走らせる

```bash
python -m scripts.lesion --parent outputs/develop/<条件>
```

| 引数 | 意味 |
|---|---|
| `--parent` | 引き継ぐ親 run。**run ディレクトリそのものでも、`seedNN/` を並べた条件ディレクトリでもよい** |
| `--task` | `task.yaml` の損傷プロトコル名 (既定 `lesion`) |
| `--parallel` | 同時に走らせる run の数。**既定は親 run の `simulation.parallel`** (親の本数で頭打ち)。**結果に影響しない**ので引数で上書きしてよい |
| `--dry-run` | GeNN を触らず、切断対象を数えて終了。`<切断条件>.dry-run/` に作り、結果は `run.log` に出る |

### 出力は親 run の下

```
<親 run>/bridge-inter_cluster/          親が 1 本
<親条件>/seed01/bridge-inter_cluster/   親が sweep なら seed ごとに 1 つずつ
<親条件>/seed02/bridge-inter_cluster/
```

**`outputs/lesion/` は使わない。** 損傷 run は親 run に完全に従属していて、親を消したら
再現も再解析もできないので、置き場所も親に従属させる (ルート `CLAUDE.md` の
「出力は `outputs/<実験名>/<条件>/`」はこの実験には当てはまらない)。

ディレクトリ名 `<切断条件>` は**切断 spec (と `freeze_plasticity`) から機械的に決まる**
(`protocol/freeze.py` の `run_label()`)。だから条件名を受ける引数が無い ——
名前と中身がずれる余地を作らないため。同じ親を同じ spec で 2 度切ることはできない
(既にあれば止まる。撮り直すなら先に消す)。

### なぜ develop と違って `--config` を取らないのか

**ネットワーク設定が親 run から来る**ため。同じ seed で同じネットワークを再ビルドしないと
重みを復元できないので、config を別に指定する余地が無い。ランチャは親の `config.yaml` を
読み、`task` を損傷プロトコルへ差し替え、`task.parent_run` に親のパスを焼き込んでから
run ディレクトリへ置く。

**おかげで run の `config.yaml` だけで「どのネットワークを、どこで切って、どう観察したか」が
全部読める。** ディレクトリ名からは何も読み取らない。

失敗した run だけやり直す / 図と指標を作り直す:

```bash
python -m scripts.lesion.run_one <親>/seed03/bridge-inter_cluster
python -m scripts.lesion.replot  <親>/seed03/bridge-inter_cluster [--verify] [--no-structure]
```

構造図も切断直後の記録 (切断後の結合・遅延・重み・軸索幾何。一覧は `store/built.py`)
から描きます (ネットワークは組み直さない)。`--verify` を付けたときだけ、親と同じ
ネットワークを再ビルドして記録どおりに切り、記録と同じ結合と軸索幾何が出るかを確かめます。
切断後の軸索幾何を記録する前に走った run では `axon network` が skip になります。
`cut axon network` は切ったシナプスの経路を**親 run の軸索幾何**から引くので、親 run が要ります。

### sweep を丸ごと作り直す

`replot.py` が取るのは**run ディレクトリ 1 つ** (`config.yaml` と `data/` を持つ階層) です。
親が sweep なら seed ごとに 1 つあるので、シェルのループで回します。
**暫定の手順で、専用の入口はまだありません。**

```bash
for d in outputs/develop/<条件>/seed*/bridge-inter_cluster; do
    python -m scripts.lesion.replot "$d"
done
```

1 run あたりの時間は図を描く時間だけです。`--verify` を付けると親と同じネットワークを
組み直すので、親の `connection` プロファイルの重さがそのまま効きます。

---

## 損傷プロトコル (`task.yaml`)

記録プロトコルと損傷条件の**両方**がここに入る。develop の `task.yaml` が「いつ記録するか」
だけなのに対し、損傷実験は「どこを切るか」「いつ切るか」も結果を決めるため。

```yaml
lesion:
  from_hour: null              # 親のどの記録時刻を引き継ぐか。null なら最後
  cut:                         # 切断 spec。複数書ける。**run ディレクトリ名になる**
    - "bridge:kind=inter_cluster"
  cut_combine: or              # "or" | "and"
  freeze_plasticity: []        # 切断後に STDP を止めるシナプス群 (下の節)。**空でなければ名前に付く**
  settle_ms: 60000.0           # 切断直後に落ち着かせる時間。時間軸には含まれない
  record_hours: [0..12]        # 記録する時刻 [h]。**基準は切断の瞬間**
  record_window_ms: 600000.0   # 1 probe の記録窓
  record_buffer_ms: 10000.0
```

`record_hours` の書き方は develop と同じ (`[0, 6, 72]` / `[0..12]` / `[0..3:0.5]`、
**リストは範囲にならない**)。展開は `config_manager.expand_hours_spec()` が唯一の入口で、
run の `config.yaml` には展開後の時刻が並ぶ。

**総時間を書くフィールドは持たない。** run 長は `record_hours` の最大値 +
`record_window_ms` で決まるので、手計算した値と実際が食い違う余地を作らない。

**解析のパラメータはここに無い。** ハブ判定の z 閾値 (Guimera-Amaral の 2.5) も、
媒介中心性を厳密に解く上限 N も、それを使う `src/utils/analysis/graph.py` が既定値として
持っている。config に置くと同じ定義が 2 か所になるうえ、切断条件と紛らわしい ——
**何をどこで切るかを決めるのは `cut` の spec だけ**。

### 切断後に可塑性を止める (`freeze_plasticity`)

シナプス群の名前 (`config.synapses` のキー) を並べると、その群の STDP の学習率を 0 にした
ネットワークで切断後を走らせる。回復が i-STDP と e-STDP のどちらから来るかを切り分けるための操作。

```yaml
freeze_plasticity: [from_Inh]   # i-STDP を止める (I→E と I→I の重みが切断時の値で固定)
freeze_plasticity: [from_Exc]   # e-STDP を止める (E→E と E→I)
```

- **止めるのは学習率だけ。** 短期可塑性 (U, tau_rec) は動いたままなので、伝達の仕方は親と同じ
- **止めるのは切断の瞬間から。** 親 run (損傷前の発達) は止めていない
- run の `config.yaml` の `synapses` は**親と同じ値のまま**で、止めたことは `task.freeze_plasticity`
  と `lesion.json` の `frozen_plasticity` に残る。GeNN に渡す build だけが学習率 0 の複製を使う
- 学習率のパラメータ名はモデルごとに `protocol/freeze.py` の `LEARNING_RATES` が持つ
  (`custom_Akita` は `A_E` / `A_I`)。名前の分からないモデルや存在しない群はエラーで止まる
- `task.yaml` の `lesion_freeze_inh` / `lesion_freeze_exc` が、`random:fraction=0.1` を切って
  それぞれを止めるプロファイル。比べる相手は同じ切断で何も止めない run

---

## 切断 spec

`task.yaml` の `cut:` に並べる文字列。**「どのシナプスを切るか」を決めるのはここだけ**で、
ほかの設定は一切関与しない。

### 文法

```
NAME
NAME:key=value
NAME:key=value,key=value,...
```

- `NAME` は `hub` / `bridge` / `between` / `synapses` の 4 つ。**小文字のみ**
  (`Bridge` は解釈されずエラー)。
- 引数は `,` 区切りの `key=value`。空白は前後が除かれる。
- **キーを 1 つも書かなくてよい。** 省略したキーは既定値になる (`hub` だけで
  `hub:metric=participation,top=1,direction=out` と同じ)。
- **未知のキーは即エラー。** `top` を `topk` と書いて黙って全件切る、が最悪の失敗なので。
- 値の中で `+` は「複数並べる」の意味 (`parts=A+B` / `pairs=1-2+3-4`)。
- YAML 上はクオート無しでも文字列として読めるが (`- bridge:kind=inter_cluster`)、
  **クオートして書くのを既定にしている** —— コロンを含む値を足したときに黙って壊れないため。

```yaml
cut:
  - "bridge:parts=BX1-3"
  - "hub:metric=out_strength,top=5,direction=both"
cut_combine: or
```

**書いたら `--dry-run` で数える。** GeNN を触らずに下見だけ走るので、spec を確かめるのは
これが一番早い。切断本数・属性ごとの「切断群 vs 残存群」・幾何 vs 位相の突き合わせまで
`run.log` に出る (端末には出ない)。下見は本番とは別の `<親 run>/<切断条件>.dry-run/` に
作られ、`config.yaml` と `run.log` だけが残る (`data/` の記録は書かれない)。だから下見の後に
そのまま本番を走らせられる。同じ spec でもう一度下見すると「既にあります」で止まるので、
要らなくなった下見のディレクトリは手で消す。

```bash
python -m scripts.lesion --parent outputs/develop/<条件> --dry-run
```

---

### `bridge` —— 軸索が通ったブリッジで切る (幾何的)

**その part の領域を軸索の折れ線が実際に通過したシナプス**を切る。pre / post がどの
モジュールに属するかは見ない。

| キー | 既定 | 値 |
|---|---|---|
| `kind` | `any` | `any` / `inter_cluster` (`BX*`) / `intra_cluster` (`BC*`) |
| `parts` | (なし) | ブリッジ part 名を `+` で並べる。**書くと `kind` は無視される** |
| `samples` | 関数の既定 | 折れ線をブリッジ判定するときのサンプル点数。上げると取りこぼしが減る |

```yaml
- "bridge:kind=inter_cluster"        # BX で始まる 4 本すべて
- "bridge:kind=intra_cluster"        # BC で始まる 16 本すべて
- "bridge:parts=BX1-3"               # クラスタ 1-3 を繋ぐ 1 本だけ
- "bridge:parts=BX1-3+BX0-2"         # そのどちらかを通ったもの (OR)
- "bridge:parts=BC1-0-2,samples=200" # クラスタ 1 内の 1 本、判定を細かく
```

**part 名は area が決める。** `hierarchical_modular_grid` なら
`C{c}-M{m}` (モジュール 16) / `BC{c}-{m1}-{m2}` (クラスタ内ブリッジ 16) /
`BX{c1}-{c2}` (クラスタ間ブリッジ 4)。実際の並びは:

```
BC0-0-1 BC0-0-2 BC0-1-3 BC0-2-3  BC1-0-1 … BC3-2-3      (16 本)
BX0-1   BX0-2   BX1-3   BX2-3                            (4 本)
```

`kind` の `BX` / `BC` という接頭辞での分類は **`hierarchical_modular_grid` 固有の命名**に
依存しているので、`protocol/cut.py` (実験側) が判定を持つ。汎用側でのブリッジの定義は
「`allow_soma: false` の part」1 つだけ (`src/utils/analysis/connectivity.bridge_part_indices`)。

**切るのは「露出した通路を通った結合」。** ブリッジ part は union を連結にするため両端を
`bridge_overlap` だけモジュールへ食い込ませてあるが、その食い込みはモジュールの内側なので
判定から外す (`axons.synapse_crossed_bridges`)。ここを区別しないと、モジュール内で完結する
軸索が食い込みをかすめただけで切られる —— 既定値では切断本数が 2〜3 倍に膨らみ、その
半分近くが同一モジュール内の結合になる (`docs/technical/compat_breaks.md`)。

判定範囲は **soma から「その結合の接触点」まで**なので、接触より後に通ったブリッジでは
切られない。逆に、露出部へ出てから戻って接触した結合は切られる —— 軸索を 1 本の線として
焼くのだから、切断点より遠位の結合は全部落ちるのが正しい。

**要るもの**: 軸索の幾何 (`connection: axon_growth` 系) と、part を持つ area。
`constant_prob` や `no_space` の run では使えない (明示的にエラーで止まる)。

**止まる書き方**: 存在しない part 名 → 全 part 名を並べてエラー。モジュール part
(`C1-M3` など) を書いた → 「条件に合うブリッジ part がありません」。

---

### `between` —— 軸のラベルが違う結合を切る (位相的)

`layout` の軸を見て、**pre と post でラベルが違う**シナプスを切る。軸索がどこを通ったかは
見ない。

| キー | 既定 | 値 |
|---|---|---|
| `axis` | `module` | layout の軸名 |
| `pairs` | (全部) | `A-B+C-D`。指定したラベル対だけに絞る。**向きは問わない** (A→B も B→A も) |

```yaml
- "between:axis=module"                          # module が違う結合すべて
- "between:axis=polarity,pairs=excitatory-inhibitory"   # E→I と I→E だけ
```

**使える軸**は run によって違う。config だけから決まるのが `population` / `mode` /
`polarity` の 3 つで、それに space や connection が貼る軸が加わる。
`hierarchical_modular_grid` + `axon_growth` の run なら:

```
population    2 値   Layer_Exc / Layer_Inh
mode          2 値   excitatory / inhibitory
polarity      2 値   excitatory / inhibitory
module       16 値   C0-M0 … C3-M3
axon_length 255 値   (連続値。切断の軸には向かない)
```

**`pairs` はラベルにハイフンを含む軸では使えない。** 値は最初のハイフンで 2 つに割るので、
`pairs=C1-M3-C2-M0` は `("C1", "M3-C2-M0")` と解釈され、どのラベルにも一致しない
(結果 0 本になり、run は「切断対象が 0 本です」で止まる)。module 単位で絞りたいときは
`bridge:parts=` を使うか、`axis=module` を丸ごと切ること。

---

### `hub` —— 指標の上位ニューロンの結合を切る

ニューロン単位の指標を計算し、**上位 `top` 個**のニューロンに繋がるシナプスを切る。
閾値ではなく順位で選ぶ。

| キー | 既定 | 値 |
|---|---|---|
| `metric` | `participation` | 下表の 8 つ |
| `top` | `1` | 上位いくつ取るか |
| `direction` | `out` | `out` (そのニューロンが pre) / `in` (post) / `both` (和) |
| `axis` | `module` | `participation` と `within_module_z` を測るときのモジュール軸 |

| `metric` | 何の上位か |
|---|---|
| `out_degree` / `in_degree` / `degree` | 出次数 / 入次数 / 合計 |
| `out_strength` / `in_strength` | 重み付き次数 (重みは**親から復元した実値**) |
| `betweenness` | 媒介中心性 (有向、正規化済み) |
| `participation` | Guimera-Amaral の参加係数 (モジュールをまたぐ度合い) |
| `within_module_z` | モジュール内 z スコア (モジュール内でのハブ度) |

```yaml
- "hub:metric=participation,top=3,direction=out"   # コネクタ上位 3 個の出力
- "hub:metric=out_strength,top=5,direction=both"   # 強い出力を持つ 5 個の入出力
- "hub"                                            # = participation, top=1, out
```

**同点の扱いは決めていない。** 次数系の指標は小さい網だと同点が多く、`top=3` が同点の中から
どの 3 個を取るかは規定されていない (`np.argsort` の順)。再現はするが、**「上位 3 個」に
意味があるかは指標の分布を見てから決めること。**

**`metric=betweenness` は N > 2000 で推定値に落ちる** (`graph.py` の
`DEFAULT_MAX_EXACT_BETWEENNESS`)。ピボットのシードは固定なので run をまたいで再現するが、
厳密解ではない。大きな網でこれを切断の基準にするなら、その前提を承知の上で使う。

**指標はすべて切断前のネットワークで測る。** 切った後にハブが入れ替わっても選択は変わらない。

---

### `random` —— 極性で絞った結合から無作為に切る

pre / post の極性で候補を絞り、その**候補のうち** `fraction` の割合を無作為に切る。
E/I の釣り合いを意図的に崩す損傷 (E の出力だけを失わせる等) に使う。

| キー | 既定 | 値 |
|---|---|---|
| `fraction` | — (必須) | (0, 1]。**候補の本数に対する**割合 |
| `pre` | `any` | `exc` / `inh` / `any` |
| `post` | `any` | `exc` / `inh` / `any` |
| `seed` | `0` | 抽選の乱数。**シミュレーションの乱数とは独立** |

```yaml
- "random:pre=exc,fraction=0.1"            # E の出力 (E→E と E→I) の 10%
- "random:pre=exc,post=exc,fraction=0.1"   # E→E だけの 10%。I が受ける E 入力は残る
```

**E→I も切るかどうかで意味が変わる。** `pre=exc` だけだと I ニューロンも E 入力を失うので、
E が受ける I 駆動も一緒に落ちて E/I 比が崩れにくい。E が受ける E/I 比を崩したいなら
`post=exc` まで絞る。

同じ親 run・同じ spec なら同じシナプスが選ばれる (候補は COO の row-major 順、
抽選は `seed` だけで決まる)。別の抽選を見たいときは `seed` を変える —— ディレクトリ名に
入るので別の run になる。

---

### `synapses` —— 名指しで切る

| キー | 既定 | 値 |
|---|---|---|
| `pairs` | — | `pre-post` を `+` で並べる。**グローバル ID** |
| `file` | — | `pre` / `post` 列を持つ CSV へのパス。実行時のカレントからの相対 (ランチャ経由ならプロジェクトルート) |

```yaml
- "synapses:pairs=3-7+12-40"
- "synapses:file=data/cut_list.csv"
```

**存在しないペアはエラー。** 黙って無視すると「何も切らなかった」が静かに通るため。
ID はグローバル (`0..N-1`) で、population ローカルではない。

---

### 複数の spec を組み合わせる

`cut` はリストなので何本でも書ける。合成は `cut_combine`:

| `cut_combine` | 意味 |
|---|---|
| `or` (既定) | どれかに該当するシナプス |
| `and` | 全部に該当するシナプス |

```yaml
cut:
  - "bridge:kind=inter_cluster"
  - "hub:metric=participation,top=3,direction=out"
cut_combine: and        # ハブの出力のうち、クラスタ間ブリッジを通ったものだけ
```

`and` は交わりが空になりやすく、0 本なら run は止まる。`lesion.json` の
`selector_detail.parts` に **spec ごとの本数**が残るので、どちらが効いたかは後から読める。

### spec は run ディレクトリ名になる

| spec | ディレクトリ名 |
|---|---|
| `bridge:kind=inter_cluster` | `bridge-inter_cluster` |
| `bridge:parts=BX1-3` | `bridge-BX1-3` |
| `between:axis=module` | `between-module-all` |
| `hub:metric=participation,top=3,direction=out` | `hub-participation-top3-out` |
| `random:pre=exc,post=exc,fraction=0.1` | `random-exc-exc-f0.1-seed0` |
| `synapses:pairs=3-7+12-40` | `synapses-2` |
| 上記 2 つを `and` | `<label1>_and_<label2>` |

ネットワークを組む前に spec だけから決まる (`protocol/cut.py` の `cut_label()`)。**同じ親を同じ spec で
2 度切ることはできない** (既にあれば止まる)。

`freeze_plasticity` が空でなければ、`+freeze-<群>` が後ろに付く
(`random-any-any-f0.1-seed0+freeze-from_Inh`。決めるのは `freeze.run_label()`)。
止めない run の名前は変わらない。

### `bridge` と `between` は一致しない

「ブリッジを横切った軸索が作った結合」と「module ラベルが違う結合」は別の集合で、本数も違う。
モジュール内で完結しつつブリッジに出入りする軸索があるため。実測 (N=256 の run):

```
bridge_crossing_geometric : 1106 本   (どれかのブリッジを通った)
between_module_topological:  542 本   (module ラベルが違う)
bridge:kind=inter_cluster :  262 本   (BX* を通った = 実際に切った本数)
```

`lesion.json` の `cross_check` に前 2 つが入るので、**解釈するときに取り違えないこと。**

---

## 切断は再ビルド / ベースラインは親 run から

GeNN のシナプス集団は `set_sparse_connections()` で確定し、実行中に行を削れない。
だから構造的除去は「生成直後の COO から行を落としてから登録する」という形になる
(`NetworkBuilder.build(transform_coo=...)`)。

```
下見 (build するが compile しない)
  → 親の重みを (pre, post) で復元し、切断対象を決める
本番 (compile する 1 回きりの build)
  → 復元した重みを載せ、切断マスクで行を削除
  → 記録を書く (config / 切断前後の結合・遅延・重み・軸索幾何 / lesion.json / 切断プロファイル)
  → 読み直して構造図 (series.built())
  → setup → GeNN 上の重みが復元値と一致するか確かめる
  → ベースライン (番号 BASELINE_INDEX) = 親 run の記録時刻の窓をそのまま置く
  → settle → record_hours の各時刻で probe (番号は record_hours の添字 0, 1, …)
```

**ベースラインは測り直さない。** 切断前のネットワークをもう一度走らせても、settle と
記録窓のあいだに起きるのは親 run の続きでしかないので、親の `weights_{h}h.npz` と
`spikes_{h}h.npz` をそのままベースラインに置く (引き継ぎ時刻は `task.from_hour`)。
おかげで GeNN のコンパイルは run につき 1 回で済む。

**ベースライン窓の幅は親のもの**で、`task.record_window_ms` とは違いうる。窓幅は窓ごとに
npz が持つので発火率は正しく出るが、**全部を同じ幅で割ってはいけない**。

復元経路は自己検証する。`setup()` の後に GeNN から重みを pull し、親から引き当てた値の
残存ぶんと一致することを assert する。ここが狂うと以降の指標は全部別物になる。

### 重みの引き当ては位置ではなく (pre, post)

**並べ替えのためではなく、検証のためにこの形にしてある。** COO は row-major ソート済み
(ルート `CLAUDE.md` の不変条件 4) なので、親と同じネットワークが出ているなら並びも同じで、
位置で対応づけても結果は変わらない。違うのは**同じでなかったときの振る舞い**:

- 位置: 黙って別のシナプスへ値が乗る。どこも例外を出さない
- join: 本数が違う / 保存側に無いシナプスがある / 保存側に重複がある、のどれかで止まる
  (`src/utils/analysis/weights.align_saved_to_coo`)

3 つを全部通ることは「結合集合として同一」と同値なので、**この join が
「親 run と同じネットワークを組み直せた」ことの証明そのもの**になっている。
位置で対応づけると、この証明ごと無くなる。コストは数十万本でも sort 1 回ぶん。

### 外部軸は切断対象を決める前に読み戻す

軸そのものは下見の build で各コンポーネントが貼り直す (`describe_axes()`)。親の
`layout_axes.npz` を読み戻すのは、**貼り直した軸を記録された値に釘で留める**ため ——
コンポーネントの実装が変われば、同じ config・同じ seed でも別のラベルになりうる。

`between:axis=module` はその軸を見て切るので、読み戻しが選択より後だと
**「親 run とは違うラベルで切った」ことに気づけないまま通る**。親が記録した軸を
今回の build が貼らない場合も、読み戻しが後なら `labels()` の例外が出ずに
別の選択になる。`run_one.py` は下見の build の直後・`select_cut()` の前に
`load_axes_file()` する。

---

## 出力

run ディレクトリは `<親 run>/<切断条件>/`。

```
<run>/
├── config.yaml           **run の記録**。親 run・切断 spec・タイムラインが全部入る
├── run.log
├── data/                 **記録** (測った値と、プログラムが読む計算結果)
│   ├── connectivity_pre.npz  切断**前**の結合 (全シナプス)
│   ├── connectivity.npz      切断**後**の結合 (生き残り)
│   ├── layout_axes.npz
│   ├── coords.npz            soma の座標 (`no_space` の run には無い)。切断で動かない
│   ├── lesion.json           **切断で何が起きたかの記録**
│   ├── lesion_cut.npz        切断されたシナプス 1 本ごとの素性
│   ├── structure.csv         配線のトポロジー指標 (次数・成分・ハブ・媒介中心性…)。切断前と切断後の 2 行
│   ├── metrics.csv           probe ごとに変わる指標 (**1 行ずつ追記**)
│   ├── rte.csv               probe ごとの RTE (実効結合の損傷前比。run 終了後に一括)
│   ├── weights_p{NNN}.npz    その probe の重み (NNN = record_hours の添字)
│   ├── spikes_p{NNN}.npz     スパイク + 窓の原点 / 窓幅 / phase
│   ├── voltage_p{NNN}.npz    窓の先頭の膜電位。**採った probe だけ**
│   ├── current_p{NNN}.npz    窓の先頭の入力電流。**採った probe だけ**
│   └── {kind}_baseline.npz   親から引き継いだベースライン (番号を持たない)
└── report/               **報告** (人が読む図と表)。消しても replot で作り直せる
    ├── structure/        切断**後**のネットワークの形
    │                     connection_probability.csv / bridge_hops.csv  切断**後**の群間結合確率
    │                     cut_profile.csv  属性ごとの「切断群 vs 残存群」(lesion_cut.npz などの記録から作る)
    ├── panels/           **probe ごと**の図。probe の数だけ増えるものはここへ
    │   ├── raster/
    │   ├── avalanche/
    │   ├── weight/               その probe の重み行列
    │   ├── weight_distribution/  その probe の重み分布
    │   ├── weight_network/       その probe の重みで描いたネットワーク図
    │   ├── correlation/          その probe のニューロン間相関行列
    │   ├── correlation_network/  相関の強いペアだけを空間に描いた無向グラフ
    │   └── trace/                task.trace_neuron を指定した run のみ
    └── overview/         weight_trajectories / weight_distribution_shift /
                          firing_rate_scatter / mean_correlation_track / drive_track /
                          chi_track / participation_heatmap / rte_track /
                          {weight,correlation}_change_{matrix,network}[_across_cut] /
                          weight_change_axon_network[_across_cut] /
                          flip_timing.csv (下の「反転の規則」) /
                          input_budget.csv / neuron_budget.csv (下の「駆動の収支」)
```

`*_change_*` は 2 つの窓の差 (後 − 先) を描いた図 (`figures/change.py`)。

- 末尾なし: **切断直後 (0 h の probe) → 20 h** の差 (回復)。目標時刻は `TARGET_HOUR`。
  20 h の窓を持たない run では skip される。`weight_change_axon_network` は軸索の幾何を
  持つ run だけ。
- `_across_cut`: **切断前 (baseline の窓) → 切断直後** の差 (切断そのものの効果)。
  切断直後の窓は settle の後なので、settle 中の可塑性による変化も含む。重みの差は切断前の
  結合の上で取り、切ったシナプスは切断後を 0 として数える (切断前の重みぶんの減少として出る)。
  `weight_change_axon_network_across_cut` は親 run の (切断前の) 軸索幾何の上に描くので、
  どの軸索のどこで切ったかが読める。

色の範囲は図ごとにデータから決めるので、run や図をまたいで色で比べないこと。

### `metrics.csv` の駆動の列 (`drive_*`)

各ニューロンが E / I から実際に受け取った**時間平均コンダクタンス [nS]** を、後ニューロンの
極性ごとに平均したもの (`analysis/drive.py`)。スパイク列から STP を再現して積むので、
発火率や重みだけでは見えない「実際に届いた量」になる。

| 列 | 意味 |
|---|---|
| `drive_exc_onto_e_ns` / `drive_inh_onto_e_ns` | E ニューロンが E / I から受けた駆動 |
| `drive_ei_ratio_onto_e` | その比 (E / I) |
| `drive_*_onto_i_*` | I ニューロンについて同じもの |

**可塑性 `custom_Akita` × シナプス `ExpCond` の組でしか計算できない** (それ以外の run では
`NotImplementedError` で止まる)。**電流ではなくコンダクタンス**なので、駆動力
(V − E_rev) は掛かっていない。

### 群ごとの同期 (`chi_*` の列、`chi_track` / `participation_heatmap`)

群は `module` ラベルの `-` より前 (`C0-M1` → `C0`)。全体 (`all`) と群ごとに測る
(`figures/style.py` の `synchrony_groups`)。

| 出力 | 中身 |
|---|---|
| `chi_<群>` (metrics.csv) | Golomb の同期指標 χ。50 ms ビンのスパイク数で、χ² = 群平均の分散 ÷ 各ニューロンの分散の平均。全員揃って 1、独立で約 1/√n |
| `chi_track.png` | χ の推移。群の人数で決まる独立時の底 (1/√n) を同じ色の点線で引く |
| `participation_heatmap.png` | 100 ms ビンごとの参加率 (群の何割が撃ったか) の分布。横 = probe、縦 = 参加率、色 = 1 秒あたりの出現回数。群の誰かが撃ったビンを全部数え、イベント閾値 0.2 は線で重ねるだけ |

### 実効結合 (`rte.csv`、`rte_track`)

Watanabe et al. 2026 の GTE → RTE (`src/utils/analysis/transfer_entropy.py`)。50 ms
フレームに 1 発以上で 1、マルコフ次数 2、大域状態での条件づけなし、全ニューロン。
**基準は baseline の窓**で、そこで z > 1 だった結合の集合 C を固定し、各 probe の TE を
baseline の TE で割った比を C の上で平均する。基準の窓が決まらないと値が決まらないので
`metrics.csv` には入れず、run 終了後に `rte.csv` として一括で書く (`analysis/effective.py`)。

| 列 | 意味 |
|---|---|
| `rte` | C の上の TE / TE₀ の平均。baseline の行は 1 |
| `rte_connections` | \|C\| |
| `mean_te_bits` | その窓の全ペアの TE の平均 [bit]。C に限らない水準 |

**RTE は C の外を見ない**ので、損傷後に生まれた経路は `rte` に出ない。`mean_te_bits` が
`rte` より戻っていれば、元の経路以外で情報が流れている。TE はフレーム数で偏るので、
baseline と post の窓幅は揃えておくこと (今の task.yaml はどちらも 10 分)。

### `structure/` と `panels/` の出自は develop の複製

`figures/` と `report/{structure,panels}.py` (コードのパッケージ) は **`scripts/develop/` から複製したもの**。
いまのところ出る図の種類・ファイル名・描き方はほぼ同じで、窓のタイトルだけ `hour` では
なく `label` (`baseline (-0.17 h)` / `post (+3.00 h)`) を使う —— 損傷実験では時刻だけでは
切断の前後が読めないため。

これは**出自の説明であって、揃え続ける決まりではない**。複製であって共有ではないのは、
図が実験ごとに分岐するのが目的だから (ルート `CLAUDE.md`)。片方の図を直したときに
もう片方へ持っていくかどうかは、**そのつど決める**。

### ファイル名が develop と違う

`{kind}_p{index:03d}.npz`。理由は 3 つ:

1. **時刻をファイル名に埋めると精度を失う。** probe 間隔が細かいと `1.66667e-05h.npz` になる
2. **切断前の probe は負の時刻を持つ。** 記録軸は「切断からの経過」であって絶対時刻ではない
3. **develop の解析 CLI に誤読されない。** あちらの glob は `spikes_*h.npz`

**時刻の正は npz が持つ** (`record_start_ms` / `record_window_ms` / `phase`)。index は
並べ替えと人間の目印だけに使う。これは develop と同じ約束で、違うのは名前の付け方だけ。

**index は `task.record_hours` の添字。** `record_hours: [0..20]` なら `p000`〜`p020` で、
3 番目の記録時刻が `p002`。ベースラインはこの並びに属さない (親 run 由来で窓幅も別) ので
番号を持たず、`{kind}_baseline.npz` という名前で区別する。

**図のファイル名も同じ token を使う** (`raster_p003.png` / `raster_baseline.png`)。
npz と図を突き合わせるのに読み替えが要らない。

### `hour` の基準は切断時刻

契約の `Window.hour` は「基準時刻からの経過 [h]」で、基準は実験が決める。develop は
run の開始、**lesion は切断の瞬間**。baseline は窓の幅ぶん手前 (負) に置く —— 0.0 にすると
post の最初の probe と重なり、切断の瞬間が図でも CSV でも判別できなくなる。

**`settle_ms` はこの時間軸に含まれない。** `record_hours` の 0 h は「切断して落ち着かせた
直後」で、切断からの実時間は settle のぶんだけ後ろにある。

### 窓幅は probe ごとに読む

baseline は親 run の窓、post はこの run の窓なので**幅が違いうる**。全部を同じ幅で
割ってはいけない —— 窓の違いがそのまま発火率の段差として現れる。窓幅は各 probe の
npz が持つ。

---

## run をまたいで比べる (`compare`)

1 つの親条件の損傷 run を全部束ね、1 run = 1 点の表と図を作る (用量反応)。GeNN は使わない。

```bash
python -m scripts.lesion.compare outputs/develop/<条件> --control outputs/develop/<対照の条件>
```

| 引数 | 意味 |
|---|---|
| 位置引数 | 親 run を並べた条件ディレクトリ。その下の損傷 run (`data/lesion.json` を持つもの) を全部拾う |
| `--control` | **対照**: 切らずに走り続けた develop 形式の run を並べた条件ディレクトリ。切断時刻 + 20 h 以上の記録が要る (例: 20 h で切った run には 72 h sweep) |

出力は**条件ディレクトリの直下** (run の中ではない):

```
<条件>/lesion_compare/
├── runs.csv       損傷 run 1 本 = 1 行 (切断数 % / 切った重み % / 低下 / χ・相関の低下 / 回復 …)
├── control.csv    対照 run 1 本 = 1 行 (同じ式で測った回復)
└── report/        cut_weight / activity_drop / recovery_vs_drop / recovery_vs_corr_drop /
                   recovery_vs_cut / detour_*.csv (下の「迂回経路」) /
                   flip_rule*.csv (下の「反転の規則」) /
                   freeze_rate_track / freeze_recovery_split / freeze_split.csv (下の「可塑性を止めた run」)
```

- **定義は `analysis/summary.py` のモジュール docstring が正。** 回復 = (18–20 h の平均 − 0 h) ÷ 損傷前。
  対照は切断時刻を 0 h に読み替えて同じ式で測る
- **種類は切断 spec から決める** (`summary.KINDS`: bridge / random / random_exc)。表に無い spec の
  run は「対象外」と出して表に入れない。新しい種類を比べるときはここに 1 行足す
- **切断後に可塑性を止めた run (`freeze_plasticity`) も対象外。** 用量反応は何も止めずに切った run
  どうしの比較なので、同じ spec でも混ぜない (止めた run の比較は下の「可塑性を止めた run」と迂回経路の表)
- `control.csv` の `matches_lesion_baseline` は、対照の切断時刻の発火率が同じ seed の損傷 run の
  baseline と一致するか (= 対照が切断時刻まで親と同じ run か)。False なら対照として使えない
- 図の細線は、同じ seed で割合だけ変えた run (入れ子の抽選) を切断数の順に結んだもの

この段階は run 1 本の 3 段階 (structure / panels / overview) の外側にある 4 つ目で、形は
overview と同じ (表を書いてから、その表を図が読む)。リーダーは `store/collection.py` の `Collection`。

### 可塑性を止めた run の比較 (`freeze_*`)

同じ切断で、切断後に e-STDP / i-STDP の片方を止めた run と止めない run を seed ごとに対にして、
活動の回復のうちどれだけが各則の重みの変化から来たかを出す。止めた run が無い束では skip になる。

| 出力 | 中身 |
|---|---|
| `freeze_rate_track.png` | 切断からの時刻ごとのネットワーク全体の発火率 / 切断前。3 条件と対照の seed 平均 ± SEM |
| `freeze_recovery_split.png` | 寄与 Δe / Δi と Δe+Δi、止めない run の回復 (seed ごとの点、同じ seed を線で結ぶ) |
| `freeze_split.csv` | (spec, seed) ごとの 3 条件の回復と Δe / Δi / Δsum |

回復は `runs.csv` と同じ式。Δe = 回復(i-STDP を止めた) (e-STDP だけが働いた回復)、Δi も同様。両方止めた run は
無いので、Δe + Δi が止めない run の回復に一致するか (加法性) は確かめられない。
**定義は `analysis/freeze_split.py` のモジュール docstring が正。**

### 迂回経路の表 (`detour_*.csv`)

切断した E→E シナプス a→b について、残った E→E だけでたどる長さ 2〜3 の迂回経路
(順方向 a⇝b と逆方向 b⇝a) を列挙し、切断後にそれが強まったかを測る。「失った直接の結合の
代わりに迂回経路が強まる」という回復の仮説を確かめるためのもの。`compare` が一緒に出す
(1 run 0.3 秒ほど)。

4 つの表はどれも**条件ごとに seed をプールした**値。条件は `lesion.json` から作る
`spec` (切断 spec を `;` で連結。`runs.csv` の `spec` と同じ) と `frozen` (止めた可塑性の群を
`-` で連結。止めていなければ空) の 2 列。E→E を切った run が無い束では skip になる。

| 表 | 中身 |
|---|---|
| `detour_paths.csv` | 対象 (切った / 切らなかった × 強い / 0) と向きごとの、迂回経路の本数・ボトルネック重みの分布・切断後に強化された割合・強化された辺の位置 (始点側 / 中間 / 終点側) |
| `detour_reverse.csv` | 強い a→b の逆向き b→a (切断前 0) が最後に強くなったか。順方向の強い迂回の短さで分ける |
| `detour_last_hop.csv` | 順方向の迂回の最後の段 k→b の post 先行率 (切断前 / 0 h / 最後) と、弱かった k→b の強化 |
| `detour_turnover.csv` | 切らなかった E→E の強化 / 弱化の割合。対照 (`control (no cut)`) の行を含む |

**定義 (迂回経路・ボトルネック・強化・post 先行率) は `analysis/detours.py` のモジュール
docstring が正。** 切断前は親 run の記録、最後は損傷 run の最後の probe。

### 反転の規則の表 (`flip_rule*.csv`)

切られずに残った E 発の結合 (E→E / E→I) について、切断前と切断直後の probe (0 / 1 / 2 h) の
窓で e-STDP のドリフト D (全スパイク対の核の和) と post 先行率を測り、最後の probe で重みが
0.5 をまたいだか (弱 → 強 / 強 → 弱) を D の符号で当てられるかを出す。「切断で順序がずれ、
D の符号が変わった結合だけが反転する」という規則を確かめるためのもの。`compare` が一緒に出す
(1 run 0.5 秒ほど)。群は `EE` / `EI` / `EX` (両方) の 3 つを並べる。条件の列は迂回経路の表と同じ。

| 表 | 中身 |
|---|---|
| `flip_rule.csv` | 群 × 切断前の状態 (弱 / 強) × 時点ごとの、変化した割合・D の符号で予測した割合・予測の当たり具合 (予測したもの / しなかったもののうち変化した割合、感度、特異度)・D と post 先行率の AUC |
| `flip_rule_classes.csv` | 弱い結合を、切った強い結合との位置関係 (逆向き `rev` / 長さ n ≤ 3 の迂回経路の i 段目 `len{n}_hop{i}` / `other`) で分けた同じ表。複数の位置に当てはまる結合は全部の位置に数えるので、位置ごとの本数の和は結合の本数より多い |

**定義 (群・D・予測・位置関係) は `analysis/flip_rule.py` のモジュール docstring が正。**

run 1 本の表として、overview が `report/overview/flip_timing.csv` も書く。強化された E 発の結合
1 本 = 1 行で、切断後の probe (1 時間ごと) をたどり、D が初めて正になった時刻 (`drift_hour`) と
重みが 0.5 を超えた時刻 (`cross_hour`) のどちらが先か (`first`) を出す。0 h で D ≤ 0 なのに
強化された結合が、規則の範囲 (D が遅れて正になった) か規則の外 (重みが先) かを分けるためのもの。
列の定義は `flip_rule.flip_timing()` の docstring。

### 駆動の収支 (`input_budget.csv` / `neuron_budget.csv`)

run 1 本の表として、overview が書く。ニューロン b の発火率の低下 (切断前 → 0 h) と回復
(0 h → 20 h) を、b に入る結合 k→b の駆動 (時間平均コンダクタンス) の変化で説明するためのもの。

| ファイル | 1 行 | 中身 |
|---|---|---|
| `input_budget.csv` | 切断前の結合 k→b 1 本 (切った結合も含む。post → pre の順) | 重みと駆動の推移、強化 / 弱化、k 自身の低下・回復、切った強い E→E の迂回経路 (3 段まで) の何段目か (残った E→E のみ。当てはまるものをすべて `;` でつなぐ)、駆動の変化 (切られて失った / 切られずに失った / 0 h から得た / k の重み 1 あたりの駆動の変化) |
| `neuron_budget.csv` | ニューロン b 1 個 | 発火率・低下・回復、E 発 / I 発の入力の本数 (切断前と残った分)、駆動の変化の列を E 発の入力 / I 発の入力に限って足したもの |

b に入る駆動の差は、切断時 (切断前 → 0 h) は「切られて失った分 + 切られずに失った分」、回復時
(0 h → 20 h) は「0 h から得た分」の和に一致する。
**定義は `analysis/input_budget.py` のモジュール docstring が正。** 20 h の probe を持たない run では書かない。

---

## 結果を読むときの注意

**run 1 本の中には sham (切らない対照) が無い。** ベースラインは親 run の記録 1 点だけなので、
run 1 本では「回復」と「損傷が無くても進んだ発達の続き」を分離できない。対照との比較は
`compare` の `--control` (切らずに走り続けた run) で行う。`lesion.json` の
`parent_drift` に親 run の終盤の動き (`store/parent_run.py` の `DRIFT_METRIC_COLUMNS` の列) が
入っているので、**回復幅がこのドリフト幅と同オーダーなら結論を出さないこと。**

---

## ファイル構成

```
scripts/lesion/
├── __main__.py     入口: ランチャ (親)。親 run を探して run ディレクトリを作り子を並列起動
├── run_one.py      入口: 1 run (下見 → 記録 → 切断して build → probe)
├── replot.py       入口: 再解析。記録から指標と図を全部作り直す (--verify で再ビルド照合)
├── compare.py      入口: 1 つの親条件の損傷 run を束ねて比べる (用量反応の表と図)
├── task.yaml       損傷プロトコル
│
├── report/         いつ何を出すか
│   ├── structure.py   切断直後: 切断後のネットワークの形
│   ├── panels.py      probe ごと: ラスター / アバランチ / metrics 1 行
│   ├── overview.py    run 終了後: 重み軌跡 / 重み分布の変化 / 発火レート / 同期 / RTE / 差の表 / 反転の時刻 / 駆動の収支
│   └── compare.py     run をまたいで: runs.csv / control.csv、用量反応の図、迂回経路と反転の規則の表
├── store/          どこに何があり、どう読み書きするか
│   ├── paths.py       run 内部のディレクトリ規約
│   ├── formats.py     記録ファイル 1 つの読み書き
│   ├── built.py       切断直後に何を書くかの一覧 (write_built)
│   ├── views.py       Built (切断直後) / Window (probe 1 つ) / Series (run 全体)
│   ├── collection.py  Collection (1 つの親条件の損傷 run の束 + 対照)
│   └── parent_run.py  **親 run (develop 形式) を読む唯一の場所**
├── analysis/       何を測るか (表)
│   ├── metrics.py     probe 1 つの指標 + 切断したものの素性
│   ├── structure.py   配線のトポロジー指標 (切断前と切断後の 1 回ずつ) → structure.csv
│   ├── effective.py   probe ごとの RTE (baseline を基準に一括) → rte.csv
│   ├── summary.py     損傷 run 1 本 = 1 行の要約 (低下・回復・対照の定義) → runs.csv / control.csv
│   ├── detours.py     切断した E→E の迂回経路と、その切断後の強化 → 報告の表
│   ├── flip_rule.py   切断直後のドリフトの符号で、残った E 発の結合の反転を当てる / 反転の時刻 → 報告の表
│   ├── freeze_split.py 片方の STDP を止めた run と止めない run の対で、回復への各則の寄与 → 報告の表と図
│   ├── input_budget.py ニューロンごとの駆動の収支 (切断で失った分と回復で得た分の分解) → 報告の表
│   └── connectivity.py 群間結合確率 → 報告の表
├── protocol/       実験の進め方 (入口だけが import する)
│   ├── cut.py         どこを切るか (spec のパースと選択)
│   ├── freeze.py      切断後に止める可塑性と、run ディレクトリ名 (run_label)
│   └── restore.py     再ビルドが親と同じネットワークかの検証 (軸索幾何)
└── figures/        どう描くか (図)
```

### `store/parent_run.py` —— 実験どうしの横 import を作らないための場所

lesion は develop の run を読むが、**`scripts/develop/` を import しない**。代わりに
「develop 形式の記録を読む実装」をこちらが持つ。読むのは 7 つだけ
(`config.yaml` / `connectivity.npz` / `weights_{h}h.npz` / `spikes_{h}h.npz` /
`layout_axes.npz` / `axon_geometry.npz` / `metrics.csv`) で、これは develop の内部事情ではなく
**develop と lesion の契約**。ファイル名も `store/parent_run.py` がローカルに宣言する。
`metrics.csv` から読む列も同じく `CONTROL_METRIC_COLUMNS` / `DRIFT_METRIC_COLUMNS` として宣言してある。

規約が変わったら直す場所は 1 つで、そのことがモジュールの docstring に書いてある。
**この一覧が、develop 側で記録の規約を変える人にとっての索引になる。**

---

## 旧 `scripts/lesion.py` からの変更

| 旧 | 新 |
|---|---|
| CLI 引数 19 個 | `--parent` / `--task` / `--parallel` / `--dry-run` |
| 実験条件が `run_lesion.sh` の設定ブロックにあり `config.yaml` に残らない | 全部 `config.yaml` の `task.*` に焼き込む |
| 出力先が `outputs/lesion/<日時>_seedN_<label>/` = **ディレクトリ名から条件を読む形** | `<親 run>/<切断条件>/`。名前は切断 spec から機械的に決まり、読み取りには使わない |
| 走り終えてから `organize_output()` で移す | 最初から `data/` へ書く |
| 再解析の入口が無い | `replot.py` (本番と同じ関数を通り、図も `metrics.csv` もバイト一致) |
| 時刻は `probes.csv` を index で引く | 窓の npz が原点・窓幅・phase を持つ |
| 図が `src/utils/experiments/lesion/figures.py` | `figures/` (1 ファイル 1 種類) |

テストは `test/experiments/lesion/`。
