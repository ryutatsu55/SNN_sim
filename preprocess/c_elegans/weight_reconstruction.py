
import pandas as pd
import numpy as np

def extract_weight_matrix(csv_master, si5_path, target_sheet, output_filename):
    print(f"--- 【{target_sheet}】の処理を開始 ---")
    
    try:
        df_master = pd.read_csv(csv_master)
        ordered_neurons = df_master['Neuron'].tolist()
    except FileNotFoundError:
        print(f"エラー: {csv_master} が見つかりません。")
        return None

    if not ordered_neurons:
        print("エラー: マスターデータにニューロンが存在しません。")
        return None
        
    # 【動的アンカー】
    anchor_neuron = ordered_neurons[0]

    try:
        df_raw = pd.read_excel(si5_path, sheet_name=target_sheet, header=None)
    except Exception as e:
        print(f"読み込みエラー: {e}")
        return None

    df_raw = df_raw.fillna("").astype(str)
    for col in df_raw.columns:
        df_raw[col] = df_raw[col].str.strip()

    header_row_idx = -1
    index_col_idx = -1
    
    for idx, row in df_raw.iterrows():
        if anchor_neuron in row.values:
            header_row_idx = idx
            break
            
    for col in df_raw.columns:
        if anchor_neuron in df_raw[col].values:
            index_col_idx = col
            break
            
    if header_row_idx == -1 or index_col_idx == -1:
        print(f"エラー: 行列内にニューロン '{anchor_neuron}' が見つかりませんでした。")
        return None

    # データ本体の抽出
    pre_labels = df_raw.iloc[(header_row_idx+1):, index_col_idx].values
    post_labels = df_raw.iloc[header_row_idx, (index_col_idx+1):].values
    data_body = df_raw.iloc[(header_row_idx+1):, (index_col_idx+1):].values
    
    df_clean = pd.DataFrame(data_body, index=pre_labels, columns=post_labels)
    df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated(keep='first')]

    # reindexの前に、安全に浮動小数点数(float)にキャストする
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    
    print("マスターデータの順序で N x N 行列を再配置中...")
    try:
        weight_matrix_df = df_clean.reindex(index=ordered_neurons, columns=ordered_neurons, fill_value=0.0)
    except Exception as e:
        print(f"再配置エラー: {e}")
        return

    # print("\n" + "="*50)
    print("【SNN用 結合重み行列 (Weight Matrix) 抽出成功】")
    # print("="*50)
    print(f"抽出サイズ: {weight_matrix_df.shape[0]} × {weight_matrix_df.shape[1]} (マスターのNodeID数と完全一致)")
    print("\n--- 行列のプレビュー (左上 5x5) ---")
    print(weight_matrix_df.iloc[0:5, 0:5])
    # print("==================================================")

    weight_matrix_df.to_csv(output_filename, header=False, index=False)
    print(f"[完了] 純粋な数値行列を {output_filename} として保存しました。\n")
    return weight_matrix_df

def build_all_snn_matrices():
    csv_master = "src/models/network/data/c_elegans/ordered_coords.csv"
    si5_path = "preprocess/c_elegans/data_source/SI5.xlsx"

    # 1. 化学シナプス行列の抽出 (非対称・有向グラフ)
    extract_weight_matrix(
        csv_master=csv_master,
        si5_path=si5_path,
        target_sheet='hermaphrodite chemical',
        output_filename="src/models/network/data/c_elegans/weight_matrix_chem.csv"
    )

    # 2. 電気シナプス行列の抽出 (対称・無向グラフ)
    extract_weight_matrix(
        csv_master=csv_master,
        si5_path=si5_path,
        target_sheet='herm gap jn symmetric',
        output_filename="src/models/network/data/c_elegans/weight_matrix_elec.csv"
    )

if __name__ == "__main__":
    print("SNN用 結合重み行列 (Weight Matrices) の抽出を開始します...\n" + "="*50)
    build_all_snn_matrices()
    print("="*50 + "\n全ての抽出プロセスが完了しました。")