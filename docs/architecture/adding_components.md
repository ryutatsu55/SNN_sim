# コンポーネントと軸の追加手順

ルート `CLAUDE.md` の「Architecture」§1 の実務版。実際に拡張点を追加するときに読む。

---

## 1. 新しいコンポーネントを追加する

### 1-1. 基底クラスを継承したクラスを作る

```python
# src/models/neurons/my_neuron.py
from .BASE_neuron import BaseNeuronModel
from src.core.registry import NEURON_MODELS

@NEURON_MODELS.register("my_neuron_name")
class MyNeuron(BaseNeuronModel):
    @property
    def model_class(self):
        return my_genn_neuron_class

    @property
    def params(self):
        return {"param1": self.config.value1, ...}

    @property
    def initial_vars(self):
        return {"V": 0.0, ...}
```

各拡張点の基底クラスと実装すべきインタフェース:

| 基底クラス | 実装するもの |
|---|---|
| `BaseNeuronModel` | `model_class`, `params`（定数）, `initial_vars` |
| `BaseSynapse`, `BasePlasticity` | 同様。GeNN 固有のパラメータ期待値を持つ |
| `BaseConnection`, `BaseWeight`, `BaseDelay` | ネットワークトポロジのヘルパ |
| `BaseDataLoader` | trial データ（入力 + メタデータ）を yield する |
| `BaseSpace` | ニューロン座標を生成する |
| `BaseArea` | ニューロンが住み、軸索が閉じ込められる 2-D 領域。`sdf(points)` のみ抽象 |

### 1-2. `configs/components/` に YAML プロファイルを作る

```yaml
# configs/components/neurons.yaml
my_neuron_name:
  my_mode:
    in_var: "Iext"
    out_var: "V"
    value1: 1.5  # self.config.value1 として参照できる
```

### 1-3. メイン config から参照する

```yaml
# configs/test.yaml
neurons:
  MyLayer:
    type: my_neuron_name
    mode: my_mode
    polarity: excitatory
    num: 100
```

### 1-4. メインスクリプトで登録をトリガする

```python
# scripts/test.py
import src.models.neurons.my_neuron  # @register デコレータを走らせる
```

**YAML のプロファイル名が解決できないときは、まずこの import を疑うこと。**

---

## 2. カテゴリ軸 / ソート軸を追加する

任意のネットワークコンポーネントで `describe_axes()` をオーバーライドする。長さ N の配列を
返す。カテゴリなら文字列、ソートキーなら数値。フックは `generate()` の直後に走るので、
必要なものは `generate()` 中に保存しておく。

```python
@SPATIAL_MODELS.register("my_space")
class MySpace(BaseSpace):
    def generate(self):
        coords = ...
        self._coords = coords
        return coords

    def describe_axes(self):
        return {
            "module": np.array([f"M{i}" for i in assign_modules(self._coords)]),
            "depth": self._coords[:, 2],   # 数値軸 → order_by("depth") でソート可
        }
```

消費側:

```python
layout.ids_by("module")
layout.ids_where(module="M0", polarity="excitatory")
layout.order_by("depth")
```

### 解析から後で読めるようにする

build 後に永続化する。**配置場所は呼び出し側の決定**なので、run ディレクトリの規約は自分で
渡すこと:

```python
layout.save_axes(out_dir / AXES_NAME)   # AXES_NAME は src/core/output_manager.py
```

`scripts/akita_soc_fig2.py` が実際の呼び出し例。`NetworkLayout` はシリアライズを所有するが、
`outputs/` のレイアウトは所有しない。

### `module` 軸は大抵コードを書かなくてよい

`space: area_uniform` が `CompositeArea.part_of()` / `part_names` を読んで無料で宣言するので、
どの composite area プロファイル（`modular_4`, `modular_grid`, …）でも、全ニューロンが
着地したパーツの名前を得る。

1つだけ注意: パーツにはブリッジが含まれるので、`soma_in_bridge: true`（既定）だと一部の
ニューロンが `M0..M{n-1}` ではなく `B0-1` というラベルになる。軸に「このニューロンがどの
モジュールに属するか」だけを意味させたいなら `soma_in_bridge: false` にすること。

---

## 3. 名前付きの複雑な領域(area)を追加する

`beggs_plenz` と同じ2段階。

1. `configs/components/areas.yaml` に `op` / `parts` プロファイルを書く
2. **同じ名前で空の `CompositeArea` サブクラスを登録する**（`Modular4Area` が実例）

両方必要なのは、`profile_name` が YAML のキーであると同時にレジストリのキーでもあるため。

詳細と出荷済みプロファイルの一覧 → `src/models/network/CLAUDE.md`
