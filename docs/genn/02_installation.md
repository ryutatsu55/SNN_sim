# 02. インストールとセットアップ

[← 01 概要](01_overview.md) ｜ [次: 03 クイックスタート →](03_quickstart.md)

---

GeNN はソースからビルドしてインストールします（コード生成のため C++ コンパイラが必須）。

## 2.1 事前準備（Pre-installation）

1. **C++ コンパイラ**
   - Windows: Visual Studio 2019 以上（Community 版可）。インストール時に「C++ によるデスクトップ開発」を選択。
   - Linux: GCC **7.5 以上**。Ubuntu なら `sudo apt-get install g++`。

2. **GPU を使う場合（任意）**
   - NVIDIA: CUDA Toolkit をインストール。`CUDA_PATH` 環境変数で使用バージョンを指定。
     - Windows: CUDA インストール時に自動設定（`echo %CUDA_PATH%` で確認）。
     - Linux: 手動設定が必要。例: `export CUDA_PATH=/usr/local/cuda`
   - AMD: HIP をインストールし、`HIP_PATH` と `HIP_PLATFORM`（`'amd'` か `'nvidia'`）を設定。
   - これらは永続化のため `.profile` / `.bashrc` に追記する。

3. **Linux: libffi の開発版**
   - 例（Ubuntu）: `sudo apt-get install libffi-dev`

> GPU が無い環境でも、**シングルスレッド CPU バックエンド**で動作します（CUDA は不要）。
> その場合 `backend="single_threaded_cpu"` を `GeNNModel` に渡します。

## 2.2 pip でインストール

```bash
# pip を最新化
pip install -U pip

# 最新開発版（master）をインストール
pip install https://github.com/genn-team/genn/archive/refs/heads/master.zip

# 特定リリース（例: 5.3.0）を指定する場合
pip install https://github.com/genn-team/genn/archive/refs/tags/5.3.0.zip
```

## 2.3 editable インストール（開発・userproject 実行向け）

GeNN 自体を改変したり、リポジトリ同梱の userproject を動かす場合は editable install が便利です。

```bash
git clone https://github.com/genn-team/genn.git
cd genn
pip install -e .

# userproject の追加依存も入れる場合
pip install -e .[userproject]
```

> 本リポジトリの `/home/tanii/genn` は、まさにこの形（クローン済みソース）です。
> `pygenn/` 配下にビルド済み拡張モジュール（`_genn.*.so` 等）が存在することが、ロード済みの目印です。

## 2.4 レガシー: setup.py によるビルド

特殊な開発版が必要な場合のみ（非推奨）:

```bash
pip install pybind11 psutil pkgconfig "setuptools>=61"
git clone https://github.com/genn-team/genn.git
cd genn
python setup.py develop
# デバッグビルド:
python setup.py build_ext --debug develop
```

## 2.5 インストール確認

```bash
python -c "import pygenn; print(pygenn.GeNNModel)"
```

エラーなくクラスが表示されれば OK です。CPU バックエンドで最小モデルが通るかは
[03_quickstart.md](03_quickstart.md) の例で確認できます。

## 2.6 SNN_sim の依存関係との関係

SNN_sim は `kuroki/SNN_sim/requirements.txt` に依存を記載しています。GeNN/pygenn は上記の方法で
別途インストールされている前提で、SNN_sim 側は pygenn を `import pygenn` して利用します
（`src/core/NetworkBuilder.py`, `src/core/simulator.py`）。SNN_sim 固有の依存（numpy, scipy,
pydantic, PyYAML, 可視化系など）は requirements.txt 側で管理されます。
詳細は [10_snn_sim_integration.md](10_snn_sim_integration.md) を参照。

---

[← 01 概要](01_overview.md) ｜ [次: 03 クイックスタート →](03_quickstart.md)
