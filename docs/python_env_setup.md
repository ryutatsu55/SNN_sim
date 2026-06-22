# Python仮想環境の構築と環境同期の手順

このドキュメントは、プロジェクトで使用するPythonのバージョンやパッケージ（`lyon`、`numpy`、`librosa`など）を全員で統一し、安全に開発を進めるための手順書です。

## 💡 なぜ仮想環境（venv）が必要なのか？
PC全体のPython環境に直接パッケージをインストールすると、他のプロジェクトとバージョンが衝突して動かなくなる（環境が汚染される）リスクがあります。
プロジェクトごとに専用の「仮想環境（venv）」を作成することで、他の環境に影響を与えずに、全く同じバージョンのライブラリを全員の環境で再現できます。

---

## フェーズ1：現在の環境を書き出す(venvがすでにある人が行う。初めて環境作る人はフェーズ2から)
すでに正しく動いている先人の環境を「設計図（`requirements.txt`）」として書き出します。

```bash
# 既存の.venvかアクティブであることを想定しています
# 現在インストールされているパッケージ一覧をテキストに書き出す
pip freeze > requirements.txt
# 書き出された内容をgit push する
git add requirements.txt
git commit -m "何かコメントあればここに書いて"
git push origin master
```

## フェーズ2：新しい環境を構築する
### 1.仮想環境の作成
```bash
# 自分の作業フォルダに移動
cd 自分の作業フォルダ

# フェーズ1で事前に書き出されたrequirements.txtを含む最新のプロジェクトをpullする
git pull

# "venv" という名前の仮想環境を作成（少し時間がかかります）
python3 -m venv .venv
```
<!-- ### 2.仮想環境をGitの管理から除外する
.gitignore ファイルに "venv/" を追記する
```bash
echo ".venv/" >> .gitignore
``` -->

### 3. 仮想環境の有効化（アクティベート）
```bash
source .venv/bin/activate
```

### 4. パッケージの一括インストール

PyGeNN は `pip install` 実行時に `CUDA_PATH` が設定されていると、**自動で CUDA バックエンドもビルド**します。
GPU を使うため、`CUDA_PATH` を設定した状態でインストールしてください。

> **前提条件**
> - NVIDIA GPU ドライバがインストール済みであること（`nvidia-smi` で確認）
> - CUDA Toolkit がインストール済みであること（`nvcc --version` で確認）
>   - 未インストールの場合は https://developer.nvidia.com/cuda-downloads からインストール
>   - 通常は `/usr/local/cuda` にインストールされる

```bash
# CUDA のインストールパスを確認
ls /usr/local/cuda/bin/nvcc   # 存在すれば OK

# CUDA_PATH を設定してインストール（これだけで CUDA バックエンドも自動ビルドされる）
CUDA_PATH=/usr/local/cuda pip install -r requirements.txt
```

インストール後、CUDA バックエンドが有効になっているか確認できます。

```bash
python3 -c "import pygenn; m = pygenn.GeNNModel(); print(m.backend_name)"
# -> cuda  と表示されれば OK
# -> single_threaded_cpu  の場合は CUDA_PATH が設定されていなかった可能性がある
#    その場合は下記「CUDA_PATH なしでインストールしてしまった場合」を参照
```

> **備考**: `src/core/NetworkBuilder.py` が起動時に `CUDA_PATH` を自動補完するため、
> `.bashrc` 等への `export CUDA_PATH=...` の追記は不要です。

#### CUDA_PATH なしでインストールしてしまった場合

`pip install -r requirements.txt` を CUDA_PATH なしで実行した場合は、以下で後から CUDA バックエンドのみ追加できます。

```bash
# PyGeNN ソースのディレクトリへ移動
cd .venv/src/pygenn

# CUDA バックエンドのライブラリをビルド
CUDA_PATH=/usr/local/cuda make cuda_backend DYNAMIC=1 -j$(nproc)

# Python バインディングを再インストール
cd ../../..     # プロジェクトディレクトリ
CUDA_PATH=/usr/local/cuda pip install -e .venv/src/pygenn/ --no-build-isolation
```

### 5. 動作確認
構築した環境でプロジェクトが動作するか確認
```bash
python scripts/test.py
```
