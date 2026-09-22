# 設定ファイルの構造

ルート `CLAUDE.md` には最小の雛形だけを置く。ここはその詳細版。

---

## メイン config はどこにあるか

**1 実験 = 1 ディレクトリ**なので、実験のメイン config はその実験が持つ
(`scripts/<実験>/<名前>.yaml`)。`configs/` に残っているのは**どの実験にも属さないもの**
——道具が使う既定 config と、構造だけを見たい/性能を測りたいときのプロファイル——だけ。

| 置き場所 | 中身 |
|---|---|
| `scripts/<実験>/<名前>.yaml` | その実験のメイン config (`scripts/develop/axon_growth_hierarchy.yaml`, `scripts/akita_soc/akita_soc.yaml`)。lesion はネットワーク設定が親 run から来るので持たない |
| `scripts/<実験>/task.yaml` | その実験の記録プロトコル |
| `configs/*.yaml` | 道具が使うメイン config と、構造確認・ベンチ用のプロファイル (下表) |
| `configs/components/*.yaml` | コンポーネントのプロファイル。**全実験・全道具で共有** |

### `configs/` の現況

| ファイル | N | 用途 |
|---|---|---|
| `test.yaml` | 150 | `scripts/tools/pipeline_check.py` と `scripts/tools/spike_animation.py` の既定 |
| `temp.yaml` | 100 | 使い捨ての作業用。**結果を残さない前提** |
| `axon_growth.yaml` | 2827 | 軸索伸長。Sumi et al. 2025 の培養系ジオメトリ (直径 3 mm の円板) |
| `axon_growth_grid.yaml` | 256 | 軸索伸長 × `modular_grid` (2×2 をブリッジで連結。全体が 1 連結成分) |
| `axon_growth_grid2.yaml` | 67 | 同じ形の小型版 |
| `axon_growth_modular.yaml` | 891 | 軸索伸長 × 円 4 + 十字。隙間があるので **5 つの独立成分**になる |
| `criticality_test.yaml` | 3600 | 臨界テスト。`modular_4_3mm` + `axon_growth` |
| `criticality_test2.yaml` | 3600 | 臨界テスト。`no_space` + `beggs_plenz` (確率ベース) |
| `celegans.yaml` / `celegans_distance.yaml` | 81 | C. elegans コネクトーム由来の構成 |
| `bench_dc_arrival.yaml` / `_400.yaml` | 100 / 400 | per-synapse 遅延 (到着イベント駆動) の性能計測 (`docs/technical/gpu_vs_cpu.md`) |
| `profile_axonal.yaml` / `profile_dc.yaml` | 100 | 均一遅延 (`delay_by_target` あり) と per-synapse 遅延の比較 |

これらは主に `python -m scripts.tools.visualize_network_structure <config>` で構造だけを
見るために使う。実験に食わせたいときは `--config configs/criticality_test.yaml` のように
**パスを含む形**で渡す (名前だけだとその実験のディレクトリを見にいく)。

> `celegans.yaml` と `celegans_distance.yaml` は**現在バイト単位で同一**。名前が示す
> 違い (距離依存かどうか) は中身に無い。

---

## メイン config (`configs/test.yaml`)

各コンポーネントについて「どのプロファイルを使うか」を指定する。

```yaml
simulation:
  dt: 0.001
  seed: 42
  backend: cuda            # "cpu" (既定) | "cuda" | "auto"。具体値として記録される。

layout:                    # 省略可。省略すると既定が適用され、かつ記録される。
  assignment: sequential   # "random" (既定) | "sequential"

neurons:
  Layer_Exc:
    type: PQN_int          # @NEURON_MODELS.register("PQN_int") を指す
    mode: RSexci           # components/neurons.yaml のモードキー（力学のみ）
    polarity: excitatory   # E/I。mode とは独立。E/I 依存の処理は全部こちらを読む
    num: 48

network:
  area: disk                 # 必須。プロファイルは components/areas.yaml。
                             # 領域が制約を課さない場合は "no_space" を使う。
  connection: constant_prob  # components/connections.yaml のプロファイルを指す
  weight: normal_broad
  space: ...
  delay: ...
```

`layout` セクションが制御するのは population → グローバルID の割り当てだけ。
カテゴリ軸（`layer`, `module`, …）は YAML で宣言しない — コンポーネントが
`describe_axes()` で供給する。

### `network.area` が必須である理由

`area` は**他の4つの network コンポーネントと全く同じく必須**で、既定値も材料化のステップも
無い。`seed` / `backend` / `layout.assignment` が `resolve()` に既定を埋められるのとは違い、
`area` を省いた config は検証エラーになる。

これは意図的である: 領域は体細胞がどこに座れるかと軸索がどこまで届くかを決めるので、
暗黙に決まってよいことにすると、保存された `config.yaml` を見てもその run がどの領域を
使ったか分からなくなる。

結合が純粋に確率的な config（`constant_prob`, `beggs_plenz`, …）は `area: no_space` と書く。
非有界なので `contains()` は常に真で、軸索の境界処理は一度も発火しない。

---

## コンポーネント config (`configs/components/*.yaml`)

モデル固有のパラメータとモードのバリアントを持つ。

```yaml
# connections.yaml
constant_prob:
  num_modules: 4
  e_rate: 0.75
  p_out: 0.05

# neurons.yaml
PQN_int:
  RSexci:
    in_var: "Iext"
    out_var: "V"
```

コンポーネントクラスからは `self.config.<param_name>` でアクセスする
（Pydantic のドット記法。辞書キーではない）。

---

## 解決済み config の保存

`ConfigManager.save_config(config, save_dir)` が2ファイルを書く。詳細と seed / backend の
材料化については `docs/technical/reproducibility.md`。

| ファイル | 内容 |
|---|---|
| `config.yaml` | 解決済み、実シード。解析 CLI が読むのはこちら |
| `source_config.yaml` | 入力 YAML の逐語コピー。`seed: null` とコメントを保持 |
