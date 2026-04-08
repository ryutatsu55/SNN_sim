# SNN_sim

PyGeNNをバックエンドに採用した、スパイキングニューラルネットワーク(SNN)シミュレーションフレームワーク
カスタムモデルであるPQN (Piecewise Quadratic Neuron) の実装（浮動小数点/固定小数点）を含み、YAMLとPydanticによる設定管理

## Features

- **PyGeNN Backend**: GeNN (GPU Enhanced Neuronal Networks) を利用した高速なSNNシミュレーション。
- **Flexible Configuration**: YAMLベースの設定ファイルをモジュール化(`components/`)し、`Pydantic`による型チェックとバリデーションを実装
- **Registry Pattern**: ニューロンモデル、シナプス、トポロジー、データローダーなどをレジストリパターンで動的に登録・読み込み可能
- **PQN Models**: Piecewise Quadratic Neuron モデルの浮動小数点演算版(`PQN_float`)および、GeNN上で安全に動作する固定小数点演算版(`PQN_int`)を実装

## 初期設定
github、pythonの環境構築、GeNNのインストール、remoteSSHの設定が必要になる。引継ぎ用に**`github_setup.md`, `python_env_setup`**を用意したのでそれを参照して初期設定してちょうだい
GeNNのインストールについては最新版の5.4を使用している。公式ドキュメントを参考にインストールしてください。`(https://genn-team.github.io/genn/documentation/5/index.html)` ※editable installってやつです。homeディレクトリにcloneまでしてあります。各自の仮想環境にインストールしてください。
remoteSSHについては、黒木または先輩に直接聞いてください。セキュリティーの観点でgithubには共有しないこと。


## Directory Structure

```text
SNN_sim/
├── configs/            # YAML設定ファイル群
│   ├── components/     # コンポーネント別の設定 (neurons, connections, tasks など)
│   └── test.yaml       # メイン設定ファイル
├── scripts/            # 実験・シミュレーション実行スクリプト
│   └── test.py         # メインシミュレーション実行パイプライン
├── src/                # ソースコード
│   ├── core/           # ビルダー、シミュレータ、設定マネージャ、レジストリ
│   ├── data/           # データローダー (空間認識、テストデータ等)
│   ├── models/         # モジュール定義
│   │   ├── network/    # トポロジー、重み、遅延空間モデル
│   │   ├── neurons/    # PQNモデル、LIFモデルなど
│   │   ├── readouts/   # リードアウト層 (Ridge回帰など)
│   │   └── synapses/   # シナプス力学 (Tsodyks-Markram等)
│   └── utils/          # 評価、可視化ツール
└── test/               # 単体テスト、アルゴリズム検証スクリプト

```
<!-- ## 基本的な使い方

**testシミュレーションの実行**
`scripts/test.py` を使用して、設定読み込みからネットワーク構築、シミュレーション実行、可視化までのパイプラインを実行します。
```bash
python scripts/test.py
``` -->

---

## アーキテクチャと設定管理 (YAML + Pydantic + Registry)

本プロジェクトでは、「設定ファイル (YAML)」「型バリデーション (Pydantic)」「動的クラス生成 (Registry)」の3者が協調して動作します。既存のコアコード（`NetworkBuilder`等）を書き換えることなく、新しいモデルや実験を追加できる設計です。

1. **`configs/` (YAML)**: 実験パラメータや、使用するクラスの「名前（文字列）」を階層的に定義します。実行ファイル`scripts/*.py`から指定されるメイン設定(例:`test.yaml`)とモジュール別設定(`components/*.yaml`)に分割して記述します。
2. **`src/core/config_manager.py`(`pydantic`)**: 読み込まれたYAMLデータを結合し、型チェックと値のバリデーションを行ってオブジェクト化します。
3. **`src/core/registry.py`**: YAMLで指定された「文字列」を、実際の「Pythonクラス」に紐付けます。

### Coreモジュールの役割 (`src/core/`)
コアモジュールはシミュレータの心臓部。
**モデル追加時にここのコードを修正しなくていい**
* **`config_manager.py`**: YAMLの読み込み、コンポーネントの結合、Pydanticによる構造化。
    ちなみにこれをメインファイルとして実行すると最終的に完成されるconfigオブジェクトを確認できる
* **`registry.py`**: コンポーネントをグローバルに登録・取得する仕組み。
* **`NetworkBuilder.py`**: 設定とクラス部品からネットワークトポロジーを構築するファクトリ。
* **`simulator.py`**: PyGeNNへのロード、GPU転送、ステップ実行を管理。

---

## 設定ファイル (YAML) の記述ルール

パラメータは「Network系（空間・結合・重み・遅延）」と「Neuron/Synapse系」で書き方が異なります。
基本的にモデルを問わず必要なパラメータ

### 1. Network系 (トポロジー、重み、遅延など)
メインYAMLでは「使用するプロファイル名」のみを指定し、具体値はコンポーネントYAMLに記述します。

**メイン設定 (例: `configs/test.yaml`)**
```yaml
network:
  connection: constant_prob  # components/connections.yaml内のプロファイル名
  weight: normal_broad       # components/weights.yaml内のプロファイル名
```

**コンポーネント設定 (例: `configs/components/connections.yaml`)**
ここに書かれた設定値がconnection Class ("constant_prob")の中で config.e_rateのような形式でアクセスできる。
```yaml
constant_prob:
  num_modules: 4
  e_rate: 0.75
  p_out: 0.05
```

### 2. Neuron / Synapse系
メインYAMLで「モデルクラス (type)」と「動作モード (mode)」を指定します。共通のデフォルトパラメータはコンポーネント側に記述します。(Neuron系については、今のところnumのみ共通)

**メイン設定 (例: `configs/test.yaml`)**
```yaml
neurons:
  Layer_Exc:
    type: PQN_int
    mode: RSexci
    num: 48
```

コンポーネント側にはin_varとout_varで入力対象の変数と出力対象の変数を指定する。
その他そのモデルに必要な設定値もここで定義する。

**コンポーネント設定 (例: `configs/components/neurons.yaml`)**
```yaml
PQN_int:
  RSexci:
    in_var: "Iext"
    out_var: "V"
```

---

## 新しいモデルクラスの開発・追加ルール

`models/` や `data/` 配下に新しいモデルを追加する場合、以下のルールに従います。

### 1. 対応する抽象基底クラス (ABC) の継承
カテゴリに応じて必ず基底クラスを継承し、必須メソッドを実装します。

* **ニューロン (`BaseNeuronModel`)**: `model_class`, `params` (定数), `initial_vars` (初期値) を実装する。
* **データローダー (`BaseDataLoader`)**: `generate()` を実装し、1トライアルごとのデータを `yield` する。
* **空間配置 (`BaseSpace`)**: `generate()` を実装し、座標配列を返す。
* **結合マスク (`BaseConnection`)**: `generate()` を実装し、結合有無の配列を返す。
* **重み生成 (`BaseWeight`)**: `generate()` を実装し、重みの配列を返す。
* **遅延生成 (`BaseDelay`)**: `generate()` を実装し、遅延の配列を返す。

### 2. レジストリへの登録
作成したクラス定義の直上に、必ず `@カテゴリ名.register("YAMLで使用する名前")` デコレータを付与します。

```python
import numpy as np
from src.core.registry import WEIGHT_MODELS
from .weights import BaseWeight

@WEIGHT_MODELS.register("custom_weight")
class CustomWeightModel(BaseWeight):
    def generate(self):
        # 実装ロジック
        pass
```

### 3. クラス内でのパラメータの読み出し方
初期化時に渡される `self.config` は辞書ではなく **Pydanticモデルのインスタンス** です。YAMLで定義したパラメータは属性としてアクセスします。

**基本的なアクセス (ドット記法):**
```python
val = self.config.base_value
```
