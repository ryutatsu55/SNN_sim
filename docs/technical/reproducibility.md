# 再現性: seed / backend / RNG ストリーム

run を一意に決めるものは何か、どこで乱数が消費されるか、どの変更が「同じシードでも別の
ネットワーク」を生むかをまとめる。ルート `CLAUDE.md` からはここへのポインタのみを置く。

---

## 1. run を特定するのは (seed, backend) の組

**`seed` だけでは run は決まらない。`backend` も必要。** GeNN のデバイス RNG（escape noise 等）は
バックエンドごとに別のストリームなので、同じシードでも CPU と GPU では別のスパイク列になる。

実測: N=100, 2000 ms, seed=777 → CPU 73 spikes / GPU 81 spikes。

1つのバックエンド内では bit-exact。CUDA で N=2000 / 126k spikes / `num_threads_per_spike=8`
まで検証済み。**CPU と GPU の結果を同じ実験として比較してはならない。**

だから `ConfigManager.resolve()` は `simulation.backend` を材料化し、`auto` が検証済みの
`AppConfig` に残ることはない。記録は常に具体的なバックエンド名を持つ。

- `cuda` | `cpu` | `auto`（`AUTO_BACKEND_NEURON_THRESHOLD` = 400 ニューロンで解決）
- 既定は `cuda` (`DEFAULT_BACKEND`)。置き換えた旧 `NetworkBuilder.USE_GPU = True` の挙動を保つため。

---

## 2. `seed: null` は「再現不能」ではなく「今ここで引く」

`resolve()` が OS エントロピーからシードを引き、print し、config に書き込む。だから
`config.yaml` は常に実際に使われたシードを記録する。

- config の中の `null` は問題ない
- `null` が `NetworkBuilder` や `NetworkLayout` に届くのは問題である

手書きで config を組むなら `ConfigManager` を通すか、自分でシードを設定すること。

`save_config(config, save_dir)` は2ファイルを書く:

| ファイル | 内容 |
|---|---|
| `config.yaml` | 解決済み、実シード。解析 CLI が読むのはこちら |
| `source_config.yaml` | 入力 YAML の逐語コピー。`seed: null` とコメントを保持 |

`load_resolved()` はシードを発明しない。保存済み config は記録なので、そこの `null` は警告のみ。

---

## 3. シード由来のストリームは互いに意図的にオフセットされている

`simulation.seed` から派生する消費者は3つ:

| 消費者 | オフセット | 用途 |
|---|---|---|
| `NetworkBuilder.rng` | なし（生のまま） | ネットワーク構造 |
| `NetworkLayout` | `_ASSIGN_SEED_OFFSET` | E/I 割り当て |
| `BaseDataLoader` | `_LOADER_SEED_OFFSET` | 刺激 |

**同じシードから作った2つの `RandomState` は同一の系列を吐く。** したがって新しく
シード由来の RNG を足すときは必ず専用のオフセットを与えること。さもないとその draw が
ネットワークの生ビットを黙って共有する。

---

## 4. NetworkBuilder.rng は逐次に流れる1本のストリーム

`NetworkBuilder.rng` は space → connection → weight → delay を順に通る1つの `RandomState`。
**上流で draw 回数が1回でも変われば、下流の全部（重み・遅延）がずれる。**

これが以下すべての根拠になっている。

### `BaseArea` は RNG を保持も消費もしてはならない

area は上記4つ全部より前に構築される。構築中の1回の draw が下流の全ストリームをずらし、
同じシードの既存ネットワークを黙って全部変える。だから:

- `BaseArea.__init__` は `rng` を受け取らない（参照を持たないオブジェクトは消費できない）
- 乱数は `sample(n, rng)` の引数としてのみ入る
- `CompositeArea.area_um2` はモンテカルロではなく固定グリッドを数える

### `DiskArea.sample()` の draw 順は load-bearing

`RandomCircle2DSpace` と厳密に一致する（`uniform(0,1,n)` → `uniform(0,2π,n)`）ので、
`area: disk` + `space: area_uniform` は同一シードで `space: random_circle_2d` を
ビット単位で再現し、既存 config が無料で移行できる。

`test/test_axon_growth.py::test_disk_sample_matches_random_circle_2d` が固定している。
**draw の順序を入れ替えないこと。**

### 汎用 `Area.sample()` は棄却サンプリング → draw 回数が形状依存

領域の幾何だけが違う2つの config は、座標だけでなく*重みと遅延まで*ずれる。
（新しいプロファイルしかこの経路に来ないので互換性の破壊ではないが）
**比較実験では area を固定すること。**

`allow_soma: false` / `soma_in_bridge: false` も同じ理由で実現値を変える。既存 run が無傷なのは
既定が「どこでも体細胞可」で、`soma_area` が area オブジェクト*自身*を返すから draw が1つも
動かないというだけ。

---

## 5. `layout.assignment` を変えるとネットワークが変わる

`sequential` ↔ `random` の切り替えはどのニューロンが E/I かを組み替えるので、同じシードでも
シナプス群の所属と動態が変わる。**既存 run はこの設定をまたいで比較可能ではない。**

`assignment` も `seed` / `backend` と同様に `ConfigManager` が材料化する。既定値は
`config_manager.DEFAULT_ASSIGNMENT` にのみ存在し、`NetworkLayout.from_config()` は推測せず
raise し、`load_resolved()` は補完しない。
