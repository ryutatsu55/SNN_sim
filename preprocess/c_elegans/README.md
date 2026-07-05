# preprocess/c_elegans

このディレクトリは、線虫（*Caenorhabditis elegans* / C. elegans）の神経回路網データの前処理スクリプトとソースデータを格納しています。

## ディレクトリ構造

```
preprocess/c_elegans/
├── data_source/
│   └── SI5.xlsx                    # 論文付録データ (Cook et al.)
├── get_coords.py                   # ステップ1: 座標データの抽出
├── weight_reconstruction.py        # ステップ2: 重み行列の抽出
├── get_mask.py                     # ステップ3: シナプスマスクの生成
└── README.md
```

処理済みデータの出力先:
```
src/models/network/data/c_elegans/
├── ordered_coords.csv              # ニューロンの空間座標データ
├── weight_matrix_chem.csv          # 化学シナプスの重み行列
├── weight_matrix_elec.csv          # 電気シナプスの重み行列
└── synapse_mask.csv                # シナプスタイプの識別マスク
```

## データ処理パイプライン

以下の順序で3つのスクリプトを実行してください（プロジェクトルートから実行）：

### ステップ 1: `get_coords.py` - ニューロンの空間座標データ抽出

```bash
cd /home/tanii/kuroki/SNN_sim
python preprocess/c_elegans/get_coords.py
```

**処理内容:**
1. `data_source/SI5.xlsx` から介在ニューロン（INTERNEURONS）のリストと階層情報を動的に抽出
2. GitHubの標準C. elegans 3Dアトラスから座標データをダウンロード
3. 座標情報をマージし、階層番号（IN1, IN2, ...）とZ座標でソート
4. ニューロンごとに一意のNodeIDを割り当て

**出力ファイル:** `src/models/network/data/c_elegans/ordered_coords.csv`

### ステップ 2: `weight_reconstruction.py` - シナプス重み行列の抽出

```bash
python preprocess/c_elegans/weight_reconstruction.py
```

**処理内容:**
1. ステップ1で生成された `ordered_coords.csv` からニューロン順序を読み込む
2. SI5.xlsx の「hermaphrodite chemical」シートから化学シナプスの重み行列を抽出
3. SI5.xlsx の「herm gap jn symmetric」シートから電気シナプス（ギャップジャンクション）の重み行列を抽出
4. マスターニューロン順序で行列を再配置し、N×N行列として出力

**出力ファイル:**
- `src/models/network/data/c_elegans/weight_matrix_chem.csv` - 化学シナプスの重み行列（非対称）
- `src/models/network/data/c_elegans/weight_matrix_elec.csv` - 電気シナプスの重み行列（対称）

### ステップ 3: `get_mask.py` - シナプスタイプの識別マスク生成

```bash
python preprocess/c_elegans/get_mask.py
```

**処理内容:**
1. ステップ2で生成された化学シナプス行列と電気シナプス行列を読み込む
2. シナプスの有無を二値化（重み > 0 で存在と判定）
3. 各結合をシナプスタイプで分類：
   - `1`: 化学シナプスのみ
   - `-1`: 電気シナプスのみ
   - `2`: 両方のシナプスが存在（共発現）
   - `0`: シナプスなし
4. マスク行列として保存

**出力ファイル:** `src/models/network/data/c_elegans/synapse_mask.csv`

## 出力ファイル詳細

### `ordered_coords.csv`

ニューロンの空間座標情報を格納しています。

**カラム:**
- `NodeID`: ニューロンのノード識別子（0から始まるインデックス）
- `Neuron`: ニューロンの命名（例：AVAL, RIPL, AVEL など）
- `X, Y, Z`: 3次元空間座標 **[単位: μm（マイクロメートル）]**
- `Layer`: ニューロンが属する階層（IN1, IN2, IN3, IN4）

### `weight_matrix_chem.csv` / `weight_matrix_elec.csv`

シナプス結合の重み行列（行=接続元ニューロン、列=接続先ニューロン）。

- **化学シナプス（chem）**: 非対称（有向グラフ）
- **電気シナプス（elec）**: 対称（無向グラフ）

### `synapse_mask.csv`

ニューロン間の結合タイプを示す識別マスク行列。

- `0`: 結合なし
- `1`: 化学シナプスのみ
- `-1`: 電気シナプスのみ
- `2`: 化学シナプス＆電気シナプスが共発現

## データソース

- **SI5.xlsx**: C. elegans 神経回路網の詳細アトラス (Cook et al., Supplementary Information 5)
- **3D座標**: GitHub上のC. elegans標準3Dアトラス（`get_coords.py` が自動ダウンロード）
  - https://raw.githubusercontent.com/kipolovnikov/celegans_nonbacktracking/master/Data/3D_coordinates_celegans.csv

## 注意事項

- 座標の単位は **マイクロメートル (μm)** です
- SI5.xlsxのシート名（「hermaphrodite chemical」「herm gap jn symmetric」）は固定です
- スクリプトはすべてプロジェクトルート（`SNN_sim/`）から実行してください
