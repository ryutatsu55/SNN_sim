# 01. GeNN 概要

[← README](README.md) ｜ [次: 02 インストール →](02_installation.md)

---

## 1.1 GeNN とは

**GeNN (GPU-enhanced Neuronal Networks)** は、**スパイキングニューラルネットワーク (SNN)** の
シミュレーションを高速化するためのソフトウェアです。ネットワークは Python API（**pygenn**）で記述し、
GeNN がそれを **C++/CUDA のコードに生成（code generation）** して、NVIDIA GPU などのハードウェア上で
実行します。

GeNN が他の多くの SNN シミュレータと決定的に違うのは、**モデルの挙動を文字列の C 風コード
（GeNNCode）で完全にカスタマイズできる**点です（→[06_custom_models.md](06_custom_models.md)）。
組み込みモデルを使うこともできますが、それらも内部的には同じ仕組みで定義されています。

```python
from pygenn import GeNNModel

model = GeNNModel("float", "YourModelName")
```

- 第1引数 `precision`: `scalar` 型の精度（`"float"` または `"double"`）。
- 第2引数 `model_name`: モデル名（生成コードのディレクトリ名等に使われる）。

## 1.2 何ができるか

- **任意の SNN の構築と実行**：ニューロン群・シナプス群・電流源を組み合わせてネットワークを定義。
- **イベント駆動／時間駆動のシナプス**：スパイク到達時のみ走るコードと、毎ステップ走るコードを両方記述可能。
- **可塑性（学習則）**：STDP などをはじめ、任意の重み更新則を GeNNCode で記述。
- **疎結合・密結合・手続き的結合・畳み込み(Toeplitz)** など多様な結合表現。
- **シナプス遅延**（軸索遅延・逆伝播遅延・樹状突起遅延）。
- **スパイク記録**（高効率な専用記録機構）。
- **バッチ実行**（同一モデルの多数コピーを同時実行：勾配学習やパラメータスイープ向け）。
- **カスタム更新（custom update）**：毎ステップではなく任意のタイミングで GPU 上の変数を一括更新
  （状態リセット、重み更新、転置計算、バッチ間リダクションなど）。
- **カスタム結合更新（custom connectivity update）**：実行中に疎結合トポロジ自体を書き換える。
- **性能プロファイリング**（各カーネルの所要時間計測）。

## 1.3 コード生成（lazy build）の考え方

`model.build()` を呼ぶと GeNN のコードジェネレータが起動します。コード生成は **「遅延 (lazy)」**
で、**モデルが前回から変わっていなければ再生成はほぼ瞬時**に終わります（ハッシュで差分判定）。
生成された C++/CUDA はコンパイルされ、`model.load()` でメモリにロードされます。

```python
model.build()   # 必要時のみコード生成 + コンパイル
model.load()    # GPU/CPU にメモリ確保してロード
```

このため GeNN は「Python で書いた抽象的なモデル」と「実際に走る最適化済みネイティブコード」を
両立しています。

## 1.4 バックエンド

`GeNNModel` 生成時、利用可能なら **ハードウェアアクセラレーション対応バックエンドが自動選択**されます。
`backend` キーワードで明示指定も可能です。

| バックエンド | 指定文字列 | 用途 |
|--------------|-----------|------|
| CUDA (NVIDIA GPU) | `"cuda"` | 既定（GPU があれば自動選択）。`manual_device_id` でデバイス選択可 |
| HIP (AMD/NVIDIA GPU) | `"hip"` | AMD GPU 等。`HIP_PLATFORM` 環境変数と併用 |
| シングルスレッド CPU | `"single_threaded_cpu"` | GPU が無い環境・デバッグ・小規模モデル |

```python
# CPU バックエンドを明示
model = GeNNModel("float", "YourModelName", backend="single_threaded_cpu")

# CUDA で使用デバイスを指定
model = GeNNModel("float", "YourModelName", backend="cuda", manual_device_id=0)
```

> **SNN_sim ではどうなっているか**: `NetworkBuilder` は `pygenn.GeNNModel("double", model_name,
> time_precision="double")` を使い、バックエンドは未指定（=自動選択）です。GPU が無い環境で動かす場合は
> CPU バックエンドの指定が必要になります（→[10_snn_sim_integration.md](10_snn_sim_integration.md)）。

## 1.5 バッチング（batch_size）

GPU 上で小さいモデルを動かすとデバイスを使い切れないことがあります。勾配学習やパラメータスイープでは、
**同一モデルの複数コピーを同時に走らせる（バッチ）**ことでこれを解消できます。

```python
model.batch_size = 512
```

- **パラメータと疎結合はバッチ間で共有**されます。
- **状態変数を複製するか共有するか**は、各変数の `VarAccess` / `CustomUpdateVarAccess`
  （→[06_custom_models.md](06_custom_models.md#変数アクセスvaraccess)）で決まります。
  共有変数は読み取り専用でなければなりません。
- スパイク記録などバッチ依存のデータは `pop.spike_recording_data[b]` のようにバッチ `b` を指定して取得します。

## 1.6 非同期実行

CUDA などの GPU プラットフォームでは、`step_time()` のループは**各タイムステップのカーネルを
非同期に発行するだけ**で、CPU と同期しません。`t` や変数を読み出す（pull する）タイミングで
必要な同期が行われます。これにより高いスループットが得られます。

---

[← README](README.md) ｜ [次: 02 インストール →](02_installation.md)
