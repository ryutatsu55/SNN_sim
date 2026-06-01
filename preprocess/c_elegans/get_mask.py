import numpy as np
import pandas as pd

# ==========================================
# 設定 (ファイルパスなどをここで一括定義)
# ==========================================
CHEM_CSV_PATH = "src/models/network/data/c_elegans/weight_matrix_chem.csv"
ELEC_CSV_PATH = "src/models/network/data/c_elegans/weight_matrix_elec.csv"
OUTPUT_MASK_PATH = "src/models/network/data/c_elegans/synapse_mask.csv"

def generate_synapse_mask(chem_path, elec_path, output_path):
    print("シナプスマスク行列の生成を開始します...")

    # 1. データの読み込み
    # 前のスクリプトで header=False, index=False で出力しているため、header=Noneで読み込む
    try:
        df_chem = pd.read_csv(chem_path, header=None).fillna(0)
        df_elec = pd.read_csv(elec_path, header=None).fillna(0)
    except FileNotFoundError as e:
        print(f"ファイル読み込みエラー: {e}")
        return None

    # 行列の形状が完全に一致しているかアサーションで検証
    assert df_chem.shape == df_elec.shape, f"Error: 行列のサイズが一致しません (Chem: {df_chem.shape}, Elec: {df_elec.shape})"

    # NumPy配列に変換
    chem_matrix = df_chem.values
    elec_matrix = df_elec.values

    # 2. 結合の有無を二値化 (重みが0より大きければ True)
    chem_exists = chem_matrix > 0
    elec_exists = elec_matrix > 0

    # 3. マスク行列の初期化 (すべて0)
    mask_matrix = np.zeros_like(chem_matrix, dtype=int)

    # 4. マスク値の割り当て
    mask_matrix[chem_exists & ~elec_exists] = 1   # 化学シナプスのみ = 1
    mask_matrix[~chem_exists & elec_exists] = -1  # 電気シナプスのみ = -1
    mask_matrix[chem_exists & elec_exists] = 2    # 両方が存在(共発現) = 2

    # 5. 結果の保存 (C++側でパースしやすいようにヘッダー・インデックスなしのCSV出力)
    df_mask = pd.DataFrame(mask_matrix)
    df_mask.to_csv(output_path, header=False, index=False)
    
    # 統計情報の出力
    total_connections = np.sum(mask_matrix != 0)
    chem_only = np.sum(mask_matrix == 1)
    elec_only = np.sum(mask_matrix == -1)
    both_present = np.sum(mask_matrix == 2)

    print("\n" + "="*50)
    print("【マスク行列 生成完了】")
    print("="*50)
    print(f"Total connections   : {total_connections}")
    print(f"  Chemical only (1) : {chem_only}")
    print(f"  Electrical only (-1): {elec_only}")
    print(f"  Both present (2)  : {both_present}")
    print(f"-> {output_path} に保存しました。")
    print("==================================================")

    return df_mask

if __name__ == "__main__":
    # 上部で定義した定数を渡して実行
    generate_synapse_mask(CHEM_CSV_PATH, ELEC_CSV_PATH, OUTPUT_MASK_PATH)