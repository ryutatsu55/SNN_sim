import pandas as pd
import requests
import io

def prepare_spatial_data():
    # ※ 環境に合わせてSI5のパスを調整してください
    si5_path = "preprocess/c_elegans/data_source/SI5.xlsx"
    target_sheet = 'hermaphrodite chemical'

    print("1. SI5から介在ニューロン(INTERNEURONS)のリストと階層を動的に抽出中...")
    try:
        df_raw = pd.read_excel(si5_path, sheet_name=target_sheet, header=None)
    except Exception as e:
        print(f"SI5読み込みエラー: {e}")
        return

    # 全セルを文字列化して空白除去
    df_raw = df_raw.fillna("").astype(str)
    for col in df_raw.columns:
        df_raw[col] = df_raw[col].str.strip()

    # --- Excelのメタデータ構造をハッキングして範囲を特定 ---
    cat_row_idx, int_start_col = -1, -1
    for r_idx, row in df_raw.iterrows():
        for c_idx, val in enumerate(row.values):
            if val.upper() == "INTERNEURONS":
                cat_row_idx = r_idx
                int_start_col = c_idx
                break
        if cat_row_idx != -1: break

    if cat_row_idx == -1:
        print("エラー: 'INTERNEURONS' の列が見つかりません。")
        return

    # INTERNEURONSセクションの終わり（次の大カテゴリが来る列）を探す
    int_end_col = len(df_raw.columns)
    for c_idx in range(int_start_col + 1, len(df_raw.columns)):
        val = df_raw.iloc[cat_row_idx, c_idx].upper()
        if val != "" and val != "NAN":
            int_end_col = c_idx
            break

    # 階層番号とニューロン名の行インデックス
    layer_row_idx = cat_row_idx + 1
    name_row_idx = cat_row_idx + 2

    hierarchy_map = {}
    current_layer = None
    
    # セル結合(空欄)をまたぎながら層番号を記憶して抽出
    for c_idx in range(int_start_col, int_end_col):
        l_val = df_raw.iloc[layer_row_idx, c_idx]
        if l_val and l_val.upper() != "NAN":
            try:
                current_layer = int(float(l_val))
            except ValueError:
                pass
                
        n_val = df_raw.iloc[name_row_idx, c_idx]
        if n_val and n_val.upper() != "NAN":
            hierarchy_map[n_val] = f"IN{current_layer}" if current_layer else "IN_UNKNOWN"

    print(f"  -> {len(hierarchy_map)} 個の介在ニューロンと階層情報を抽出しました。")

    print("2. 標準3Dアトラスから空間座標データを取得中...")
    url = "https://raw.githubusercontent.com/kipolovnikov/celegans_nonbacktracking/master/Data/3D_coordinates_celegans.csv"
    response = requests.get(url)
    if response.status_code != 200:
        print("ダウンロードエラー")
        return

    df_coords = pd.read_csv(io.StringIO(response.text))
    df_coords.columns = df_coords.columns.str.strip()
    df_coords = df_coords[['Name', 'x', 'y', 'z']]
    df_coords.columns = ['Neuron', 'X', 'Y', 'Z']

    print("3. 座標とマージし、複合ソートを実行中...")
    # 動的抽出したリストでフィルタリング
    df_in = df_coords[df_coords['Neuron'].isin(hierarchy_map.keys())].copy()
    df_in['Layer'] = df_in['Neuron'].map(hierarchy_map)

    # アトラス側に座標が存在しないニューロンの警告
    missing = set(hierarchy_map.keys()) - set(df_in['Neuron'])
    if missing:
        print(f"\n[警告] SI5には存在するが、3D座標データに存在しないニューロン ({len(missing)}個):")
        print(sorted(list(missing)))
        print("※ これらは物理的距離が計算できないため、今回のマスター配列からは除外されます。\n")

    # 複合ソート: Layerの番号(1〜4)順、同階層ならZ座標順
    df_in['LayerNum'] = df_in['Layer'].str.extract(r'(\d+)').fillna(999).astype(int)
    df_ordered = df_in.sort_values(by=['LayerNum', 'Z']).reset_index(drop=True)
    
    # 整理してNodeIDを付与
    df_ordered.insert(0, 'NodeID', df_ordered.index)
    df_ordered = df_ordered.drop(columns=['LayerNum'])

    output_file = "src/models/network/data/c_elegans/ordered_coords.csv"
    df_ordered.to_csv(output_file, index=False)
    print(f"[完了] マスター座標データを {output_file} として生成しました。")

if __name__ == "__main__":
    prepare_spatial_data()