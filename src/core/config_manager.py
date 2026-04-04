import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_source: str, active_task: str = None, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        # サブファイルをまとめるディレクトリ
        self.components_dir = self.config_dir / "components" 
        self.main_config_path = self.config_dir / config_source
        self.active_task = active_task

    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """指定されたPathのYAMLファイルを読み込む（ファイルがない場合は空辞書を返す）"""
        if not filepath.exists():
            print(f"Warning: Missing config file: {filepath}. Returning empty dict.")
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def resolve(self) -> Dict[str, Any]:
        """
        全YAMLファイルを統合し、1つの resolved_config を生成する。
        """
        main_cfg = self._load_yaml(self.main_config_path)
        
        # test.yaml の構造に合わせてベースとなる設定箱を用意
        # neurons や synapse_groups はメイン設定ファイルから直接取得する
        resolved = {
            "simulation": main_cfg.get("simulation", {}),
            "neurons": main_cfg.get("neurons", {}),
            "synapse_groups": main_cfg.get("synapse_groups", {}),
            "network": {},
            "data": main_cfg.get("data"),
            "task": {},
            "meta": {
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }
        }

        # メイン設定ファイルの network ブロックを取得
        network = main_cfg.get("network", {})

        # 読み込むべきコンポーネントYAMLと、test.yaml 内のキー名のマッピング
        network_map = {
            "space.yaml": ("space", network.get("space")),
            "connections.yaml": ("connection", network.get("connection")),
            "weights.yaml": ("weight", network.get("weight")),
            "delays.yaml": ("delay", network.get("delay")),
        }

        for yaml_file, (key_name, profile_name) in network_map.items():
            if not profile_name:
                resolved["network"][key_name] = {"type": "none"}
                continue
            
            # コンポーネント用フォルダから読み込み
            data = self._load_yaml(self.components_dir / yaml_file)
            
            if profile_name not in data:
                # print(f"Warning: Profile '{profile_name}' not found in {yaml_file}. Please check test.yaml.")
                # プロファイルが見つからない（またはファイルがない）場合は、
                # プロファイル名そのものをクラス名（type）として仮設定するフォールバック
                resolved["network"][key_name] = {"type": profile_name}
                continue
                
            # プロファイルが存在する場合は、その中身をコピーして割り当てる
            profile_data = data[profile_name].copy()
            # どのプロファイルを展開したかデバッグ用に記録しておく
            profile_data["_profile_name"] = profile_name
            resolved["network"][key_name] = profile_data

        # taskプロファイルの読み込み (active_taskが指定されている場合)
        if self.active_task:
            tasks_data = self._load_yaml(self.components_dir / "tasks.yaml")
            if self.active_task in tasks_data:
                resolved["task"] = tasks_data[self.active_task].copy()
            else:
                print(f"Warning: Task '{self.active_task}' not found in tasks.yaml.")

        return resolved

    def save_resolved(self, resolved_config: Dict[str, Any], save_dir: str = "results"):
        """実験の証拠として、結合済みのコンフィグを保存する"""
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # タイムスタンプ付きで保存
        timestamp = resolved_config["meta"]["timestamp"]
        out_path = out_dir / f"config_{timestamp}.yaml"
        
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(resolved_config, f, default_flow_style=False, sort_keys=False)
        
        return out_path