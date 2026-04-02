import argparse
import sys
from pathlib import Path

# プロジェクトルートにパスを通す
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.core.config_manager import ConfigManager
from src.core.registry import DATA_LOADERS

# デコレータを実行させるため、各種ローダーをインポートしてレジストリに登録させる
import src.data.spatial_loader 
# import src.data.audio_loader  # 将来追加した場合はここに追記するだけ

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_src", type=str, default="default.yaml")
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    # 1. Config統合 (ConfigManagerのハードコードバグは修正済みとする)
    config_mgr = ConfigManager(config_source=args.cfg_src, active_task=args.task)
    resolved_cfg = config_mgr.resolve()
    
    # 2. タスク名の取得とレジストリからの動的ディスパッチ
    task_name = resolved_cfg["task"]["name"]
    
    print(f"=== データの準備 (Task: {task_name}) ===")
    DataLoaderClass = DATA_LOADERS.get(task_name)
    
    # 3. インスタンス生成 (この瞬間、内部のPydanticスキーマによって型検証が走る)
    data_loader = DataLoaderClass(resolved_cfg["task"])
    
    # 4. データ取得
    X_train, Y_train, X_test, Y_test = data_loader.load_data()
    print(f"Loaded Train Data: X={X_train.shape}, Y={Y_train.shape}")

if __name__ == "__main__":
    main()