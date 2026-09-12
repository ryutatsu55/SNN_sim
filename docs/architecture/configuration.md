# 設定ファイルの構造

ルート `CLAUDE.md` には最小の雛形だけを置く。ここはその詳細版。

---

## メイン config (`configs/test.yaml`)

各コンポーネントについて「どのプロファイルを使うか」を指定する。

```yaml
simulation:
  dt: 0.001
  seed: 42
  backend: cuda            # "cuda" (既定) | "cpu" | "auto"。具体値として記録される。

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

`ConfigManager.save_resolved(config, save_dir)` が2ファイルを書く。詳細と seed / backend の
材料化については `docs/technical/reproducibility.md`。

| ファイル | 内容 |
|---|---|
| `config.yaml` | 解決済み、実シード。解析 CLI が読むのはこちら |
| `source_config.yaml` | 入力 YAML の逐語コピー。`seed: null` とコメントを保持 |
