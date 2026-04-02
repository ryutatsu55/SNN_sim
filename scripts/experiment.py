import argparse
import sys
from pathlib import Path
import pprint

# プロジェクトルートにパスを通す
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from src.core.config_manager import ConfigManager
# from src.topology.NetworkBuilder import NetworkBuilder  # フェーズ2で実装
# from src.core.simulator import Simulator                # フェーズ3で実装

def main():
    parser = argparse.ArgumentParser(description="SNN Experiment using PyGeNN")
    parser.add_argument(
        "--cfg_src", 
        type=str, 
        default="default.yaml", 
        help="Name of the main config file (e.g. exp1.yaml)"
        )
    parser.add_argument(
        "--task", 
        type=str, 
        required=True, 
        help="Task name to run (e.g. spatial_recognition)"
        )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="Override random seed"
        )
    args = parser.parse_args()

    print(f"=== Phase 1: Resolving Configuration ({args.cfg_src}) ===")
    config_mgr = ConfigManager(config_source=args.cfg_src)
    resolved_cfg = config_mgr.resolve()

    # コマンドライン引数でのオーバーライド処理
    if args.seed is not None:
        resolved_cfg["base"]["seed"] = args.seed

    # 実験記録の保存 (これで後から「どの設定だったか」迷わなくなります)
    saved_path = config_mgr.save_resolved(resolved_cfg)
    print(f"Configuration resolved and saved to: {saved_path}\n")
    
    # 結合された設定の一部を表示して確認
    print("[Resolved Network Component]")
    pprint.pprint(resolved_cfg["network"])
    print("\n[Resolved Task Component]")
    pprint.pprint(resolved_cfg["task"])

    # --- 今後のフェーズのプレースホルダー ---
    # print("\n=== Phase 2: Building Network ===")
    # builder = NetworkBuilder(resolved_cfg)
    # genn_model = builder.build()

    # print("\n=== Phase 3: Running Simulation ===")
    # sim = Simulator(genn_model, resolved_cfg)
    # results = sim.run(task_data=...)

if __name__ == "__main__":
    main()