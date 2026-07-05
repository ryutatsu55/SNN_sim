# NetworkLayout を使ったコンポーネントのカスタマイズ

`src/models/network/` の各コンポーネント(space / connectors / weights / delays)は、
コンストラクタで `NetworkLayout` を受け取り `self.layout` として保持している。
これを使うと、**ニューロン種ごとに生成データの分布を変える(意図的バイアス)** ことも、
**ニューロン種と生成構造の相関を打ち消す(シャッフル/無相関化)** ことも実装できる。

本ドキュメントは、その具体的な実装方法をまとめたもの。

---

## 前提: なぜ layout が必要か

グローバルインデックスは **config のニューロン宣言順に連番** で割り当てられる
(`NetworkLayout.from_config`)。例えば以下の config なら:

```yaml
neurons:
  Layer_Exc: { type: ..., mode: excitatory, num: 80 }
  Layer_Inh: { type: ..., mode: inhibitory, num: 20 }
```

- `Layer_Exc` → グローバルインデックス `0..79`
- `Layer_Inh` → グローバルインデックス `80..99`

各コンポーネントは全ニューロン `0..N-1` に対して座標ベクトルや `N×N` 行列を生成するので、
「どのインデックスがどのニューロン種か」を知りたい場合に `self.layout` を参照する。

---

## 使える NetworkLayout API(コンポーネント側で有用なもの)

| API | 返り値 | 用途 |
|-----|--------|------|
| `self.layout.total_neurons` | `int` | 総ニューロン数(= `self.num_neurons`) |
| `self.layout.items()` | `(name, GroupSpec)` の iterator | グループごとに区間を処理 |
| `self.layout.names()` | `list[str]` | グループ名一覧(config順) |
| `self.layout.slice_of(name)` | `slice` | グループの連続区間 |
| `self.layout.global_indices(name)` | `np.ndarray` | グループのグローバルID配列 |
| `self.layout.mode_of(name)` | `str` | グループの mode(例 `"excitatory"`) |
| `self.layout.ids_by_mode()` | `{"excitatory": ids, "inhibitory": ids}` | 興奮性/抑制性を prefix 分類したID |

`GroupSpec`(`self.layout.items()` が返す値)には
`.name` / `.start` / `.num` / `.slice` / `.global_indices` / `.mode` / `.params` がある。
`.params` は `config.neurons[name]`(Pydantic)なので、`params.type` 等の細かい情報も引ける。

config 値の受け取りは `configs/components/*.yaml` に任意キーを書けば
`self.config.<key>`(Pydantic `extra='allow'`)で読める。全コンポーネント共通のスキーマ変更は不要。

---

## パターン A: ニューロン種ごとの意図的バイアス

### 例 A-1: 重みの符号を種で変える(Dale則 = 興奮性は正・抑制性は負)

`src/models/network/weights.py` に新しい `@WEIGHT_MODELS.register(...)` クラスを追加する。

```python
@WEIGHT_MODELS.register("dale_normal")
class DaleNormalWeight(BaseWeight):
    def generate(self):
        weights = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float32)
        idx = (self.mask != 0)

        # まず全結合に基準分布を敷く
        base = np.abs(self.rng.normal(self.config.mean, self.config.std,
                                      size=(self.num_neurons, self.num_neurons)))

        # 抑制性ニューロンを「発信源(行)」とするシナプスだけ符号反転する
        inh_ids = self.layout.ids_by_mode()["inhibitory"]
        base[inh_ids, :] *= -1.0   # 行 = presynaptic

        weights[idx] = base[idx]
        return weights
```

対応する YAML(`configs/components/weights.yaml`):

```yaml
dale_normal:
  mean: 0.9
  std: 0.2
```

### 例 A-2: 空間モジュールをニューロン種に対応させる

`src/models/network/space.py` で、グループごとに別々の矩形領域へ配置する。

```python
@SPATIAL_MODELS.register("per_group_block")
class PerGroupBlockSpace(BaseSpace):
    def generate(self):
        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)
        groups = list(self.layout.items())
        for k, (name, grp) in enumerate(groups):
            # k 番目のグループを x 方向に区切った k 番目の帯に配置
            x0 = k / len(groups) * 100.0
            x1 = (k + 1) / len(groups) * 100.0
            n = grp.num
            coords[grp.slice, 0] = self.rng.uniform(x0, x1, size=n)
            coords[grp.slice, 1] = self.rng.uniform(0.0, 100.0, size=n)
        return coords
```

こうすると `grp.mode`(`"excitatory"`/`"inhibitory"`)や `grp.params.type` に応じて
配置ルールを分岐させることもできる。

---

## パターン B: シャッフル(ニューロン種と構造の相関を打ち消す)

連番割り当てが既定なので、`block_2d` / `prob_based_block` のような
**連番インデックスでブロック構造を作る生成器** をそのまま使うと、
ニューロン種がモジュールと相関する(例: 興奮性が全員モジュール0に入る)。
これを避けたい場合は、生成した行列/ベクトルの**要素をシャッフル**する。

### 重要: 整合性のための「共有パーミュテーション」

シャッフルを複数コンポーネントに適用する場合、
coords / mask / weight / delay が**同一の並べ替え**を使わないと、
「あるインデックスの空間座標」と「そのインデックスの結合構造」がズレて破綻する。

そこで、**シードから決定論的に生成した単一のパーミュテーション**を全コンポーネントで共有する。
以下のヘルパを1箇所(例: `src/models/network/_shuffle.py`)に置いて各生成器から呼ぶとよい。

```python
# src/models/network/_shuffle.py
import numpy as np

def structure_permutation(layout, seed_base) -> np.ndarray:
    """layout.total_neurons 個の決定論的パーミュテーション。
    全コンポーネントが同じ (layout, seed_base) を渡せば同一の並べ替えになる。"""
    rng = np.random.RandomState(seed_base)          # 生成値用 RNG とは別ストリーム
    return rng.permutation(layout.total_neurons)

def apply_vector(vec, perm):    # (N, D) 座標など
    return vec[perm]

def apply_matrix(mat, perm):    # (N, N) mask/weight/delay
    return mat[np.ix_(perm, perm)]
```

### 例 B-1: 座標をシャッフルする(下流は自動的に整合)

`space` でシャッフルすれば、そこから距離ベースで作られる
`distance_based` の connection / weight / delay は**シャッフル済み座標の上で生成される**ため、
追加のシャッフル無しで自動的に整合する。

```python
from src.models.network._shuffle import structure_permutation, apply_vector

@SPATIAL_MODELS.register("random_2d_shuffled")
class Random2DShuffledSpace(BaseSpace):
    def generate(self):
        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)
        coords[:, 0] = self.rng.uniform(0.0, 100.0, self.num_neurons)
        coords[:, 1] = self.rng.uniform(0.0, 100.0, self.num_neurons)

        if getattr(self.config, "shuffle", False):        # YAML の shuffle: true で有効化
            perm = structure_permutation(self.layout, self.config.shuffle_seed)
            coords = apply_vector(coords, perm)
        return coords
```

```yaml
# configs/components/space.yaml
random_2d_shuffled:
  shuffle: true
  shuffle_seed: 12345
```

### 例 B-2: インデックス直参照の生成器をシャッフルする

`prob_based_block` のように**座標を使わずインデックス範囲でブロックを組む**生成器は、
座標のシャッフルを自動では引き継がない。座標と揃えたい場合は、
**同じ `shuffle_seed`** を渡して同一パーミュテーションを適用する。

```python
from src.models.network._shuffle import structure_permutation, apply_matrix

@CONNECTION_MODELS.register("prob_based_block_shuffled")
class BlockShuffled(BlockRandomTopology):   # 既存 prob_based_block を継承
    def generate(self):
        mask = super().generate()           # 連番前提でブロックを構築
        if getattr(self.config, "shuffle", False):
            perm = structure_permutation(self.layout, self.config.shuffle_seed)
            mask = apply_matrix(mask, perm)
        return mask
```

`space` と `connection` に**同じ `shuffle_seed`** を書けば、両者は同じ並べ替えになり整合する。
逆に座標ベースの結合(`distance_based`)を使うなら、`space` 側だけシャッフルすれば十分。

---

## まとめ

- **意図的バイアス** → `generate()` の中で `self.layout.items()` / `ids_by_mode()` /
  `slice_of()` を参照し、グループごとに生成ルールを分岐する(パターンA)。
- **無相関化(シャッフル)** → 生成した配列を並べ替える。複数コンポーネントに適用するときは
  **共有パーミュテーション**(同一 `shuffle_seed`)で整合性を保つ(パターンB)。
- 追加パラメータは `configs/components/*.yaml` に書けば `self.config.<key>` で読める
  (Pydantic `extra='allow'`。共通スキーマ変更は不要)。
- グローバルID → ニューロン種の対応(`NetworkLayout`)は連番のまま**変わらない**。
  シャッフルはあくまで「インデックス ↔ 構造」の対応を変えるだけで、
  `layout` / GeNN 集団 / 消費側の索引には影響しない。
