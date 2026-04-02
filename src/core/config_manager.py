# src/core/config_manager.py
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
# from src.models.neurons.PQN_origin import PQNengine

class ConfigManager:
    def __init__(self, config_source: str, active_task: str, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        # サブファイルをまとめるディレクトリ
        self.components_dir = self.config_dir / "components" 
        self.main_config_path = self.config_dir / config_source
        self.active_task = active_task

    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """指定されたPathのYAMLファイルを読み込む"""
        if not filepath.exists():
            raise FileNotFoundError(f"Missing config file: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # def _inject_pqn_constants(self, neuron_config: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     PQNモデルの場合は PQN_origin.py からパラメータを抽出・注入。
    #     それ以外のモデル（LIF等）はYAMLに書かれた params をそのまま保持する。
    #     """
    #     if neuron_config.get("type") != "PQN_Float32":
    #         return neuron_config

    #     mode = neuron_config.get("mode", "RSexci")
        
    #     try:
    #         # PQNengineを初期化して、浮動小数点演算用のPARAM辞書を丸ごと取得
    #         pqn_instance = PQNengine(mode=mode)
    #         # PQN.pdfの微分方程式に必要な a_fn, b_fn などのパラメータがこれで一括注入される
    #         neuron_config["params"] = pqn_instance.PARAM
    #     except ValueError as e:
    #         raise ValueError(f"Failed to load PQN parameters for mode '{mode}': {e}")

    #     return neuron_config

    def resolve(self) -> Dict[str, Any]:
        """
        全YAMLファイルを統合し、1つの resolved_config を生成する。
        """
        main_cfg = self._load_yaml(self.config_dir / "default.yaml")
        
        # ベースとなる設定箱を用意
        resolved = {
            "base": main_cfg["base"],
            "network": {
                "total_n": main_cfg["network"]["total_n"],
                "module_count": main_cfg["network"]["module_count"]
            },
            "task": {},
            "meta": {
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            }
        }

        # 1. タスクの解決 (components/tasks.yamlから)
        tasks_cfg = self._load_yaml(self.components_dir / "tasks.yaml")
        if self.active_task not in tasks_cfg:
            raise KeyError(f"Task '{self.active_task}' not found in tasks.yaml")
        resolved["task"] = tasks_cfg[self.active_task]

        # 2. ネットワークプロファイルの解決
        profiles = main_cfg["network"]["profiles"]
        
        # 各要素のマッピング (YAMLファイル名 : プロファイルキー)
        profile_map = {
            "neurons.yaml": ("neuron", profiles.get("neuron_model")),
            "topologies.yaml": ("topology", profiles.get("topology")),
            "weights.yaml": ("weight", profiles.get("weight")),
            "delays.yaml": ("delay", profiles.get("delay")),
            "synapses.yaml": ("synapse", profiles.get("synapse")),
            "plasticity.yaml": ("plasticity", profiles.get("plasticity")),
        }

        for yaml_file, (key_name, profile_name) in profile_map.items():
            if not profile_name:
                continue
            
            # コンポーネント用フォルダから読み込み
            data = self._load_yaml(self.components_dir / yaml_file)
            if profile_name not in data:
                print(f"Warning: Profile '{profile_name}' not found in {yaml_file}. Please check main.yaml.")
                resolved["network"][key_name] = {"type": "unknown", "raw_profile_name": profile_name}
                continue
                
            profile_data = data[profile_name].copy()
                
            resolved["network"][key_name] = profile_data

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