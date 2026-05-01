import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

# ==========================================
# 1. Pydanticによるスキーマ定義 (データコントラクト)
# ==========================================

class SimulationConfig(BaseModel):
    """シミュレーションの基本設定"""
    dt: float = Field(..., description="シミュレーションのタイムステップ(ms)")
    N: int = Field(..., description="総ニューロン数")
    seed: Optional[int] = Field(default=np.random.randint(0, 2**32), description="乱数シード (Noneはランダム)")
    # duration: Optional[float] = Field(default=None, description="合計シミュレーション時間(ms)")
    # backend: str = Field(default="CUDA", description="GeNNのバックエンド")
    # model_config = ConfigDict(extra='allow')

class NeuronConfig(BaseModel):
    """ニューロングループの設定"""
    type: str = Field(..., description="ニューロンモデルの種類 (例: PQN_int)")
    mode: Optional[str] = Field(default=None, description="モデルのモード (例: RSexci)")
    num: int = Field(..., description="このグループのニューロン数")
    model_config = ConfigDict(extra='allow') # components/neurons.yamlから読み出される追加パラメータ(tau, v_rest等)を許可

class SynapseParamsConfig(BaseModel):
    """シナプスダイナミクス(伝達物質の放出など)の設定"""
    type: str = Field(..., description="シナプスモデルの種類 (例: tsodyks_markram)")
    mode: Optional[str] = Field(default=None, description="シナプスのモード (例: facilitate)")
    model_config = ConfigDict(extra='allow')

class PlasticityConfig(BaseModel):
    """STDPなどの学習・可塑性の設定"""
    type: str = Field(..., description="可塑性モデルの種類 (例: static)")
    mode: Optional[str] = Field(default=None, description="可塑性のモード (例: null)")
    model_config = ConfigDict(extra='allow')

class SynapseGroupConfig(BaseModel):
    """シナプス結合グループの設定"""
    source: str = Field(..., description="シナプス前ニューロングループ名")
    # target: str = Field(..., description="シナプス後ニューロングループ名")
    # weight_scale: float = Field(default=1.0, description="初期重みのスケーリング係数")
    synapse: SynapseParamsConfig
    plasticity: PlasticityConfig
    # model_config = ConfigDict(extra='allow')

class ComponentConfig(BaseModel):
    """ネットワーク生成(空間、結合確率、重み、遅延)のベーススキーマ"""
    profile_name: str = Field(..., description="componentsディレクトリのYAMLで定義されたプロファイル名")
    model_config = ConfigDict(extra='allow') 

class NetworkConfig(BaseModel):
    """ネットワークトポロジー生成の設定"""
    space: ComponentConfig
    connection: ComponentConfig
    weight: ComponentConfig
    delay: ComponentConfig

class InputSourceConfig(BaseModel):
    """各入力ソースの汎用設定"""
    enable: bool = Field(..., description="この入力ソースを有効にするかどうか")
    model_config = ConfigDict(extra='allow')

class InputsConfig(BaseModel):
    GaussianNoise: Optional[InputSourceConfig] = None

class MetaConfig(BaseModel):
    timestamp: str

class AppConfig(BaseModel):
    """アプリケーション全体の設定を統括するルートスキーマ"""
    simulation: SimulationConfig
    inputs: InputsConfig
    neurons: Dict[str, NeuronConfig] = Field(default_factory=dict)
    synapses: Dict[str, SynapseGroupConfig] = Field(default_factory=dict)
    network: NetworkConfig
    task: ComponentConfig
    meta: MetaConfig

# ==========================================
# 2. ConfigManager 実装
# ==========================================

class ConfigManager:
    def __init__(self, config_source: str, active_task: str ):
        self.config_dir = Path("configs")
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

    def resolve(self) -> AppConfig:
        """
        全YAMLファイルを統合し、Pydanticで型検証された AppConfig を生成する。
        """
        main_cfg = self._load_yaml(self.main_config_path)
        
        # 統合用の辞書を構築
        resolved = {
            "simulation": main_cfg["simulation"],
            "inputs": main_cfg["inputs"],
            "neurons": {},
            "synapses": {},
            "network": {},
            "task": {},
            "meta": {"timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}
        }

        # コンポーネントYAMLの事前読み込み
        neurons_data = self._load_yaml(self.components_dir / "neurons.yaml")
        synapses_data = self._load_yaml(self.components_dir / "synapses.yaml")
        plasticity_data = self._load_yaml(self.components_dir / "plasticity.yaml")

        # --- 1. ニューロン設定の解決 ---
        for group_name, n_cfg in main_cfg["neurons"].items():
            resolved["neurons"][group_name] = n_cfg.copy()
            n_type, n_mode = n_cfg["type"], n_cfg["mode"]
            # components/neurons.yaml から特定の type > mode のパラメータをマージ
            if n_type in neurons_data and n_mode in neurons_data[n_type]:
                resolved["neurons"][group_name].update(neurons_data[n_type][n_mode])

        # --- 2. シナプス＆可塑性設定の解決 ---
        for group_name, s_cfg in main_cfg["synapses"].items():
            resolved["synapses"][group_name] = s_cfg.copy()

            # plasticity (STDP等)
            plas_info = s_cfg["plasticity"]
            p_type, p_mode = plas_info["type"], plas_info["mode"]
            if p_type in plasticity_data and p_mode in plasticity_data[p_type]:
                resolved["synapses"][group_name]["plasticity"].update(plasticity_data[p_type][p_mode])
                
            # synapse (コンダクタンスや放出ダイナミクス)
            syn_info = s_cfg["synapse"]
            s_type, s_mode = syn_info["type"], syn_info["mode"]
            if s_type in synapses_data and s_mode in synapses_data[s_type]:
                resolved["synapses"][group_name]["synapse"].update(synapses_data[s_type][s_mode])

        # メイン設定ファイルの network ブロックを取得
        network = main_cfg["network"]

        # 読み込むべきコンポーネントYAMLと、test.yaml 内のキー名のマッピング
        network_map = {
            "space.yaml": ("space", network["space"]),
            "connections.yaml": ("connection", network["connection"]),
            "weights.yaml": ("weight", network["weight"]),
            "delays.yaml": ("delay", network["delay"]),
        }

        for yaml_file, (key_name, profile_name) in network_map.items():
            data = self._load_yaml(self.components_dir / yaml_file)
            profile_data = data[profile_name].copy()
            profile_data["profile_name"] = profile_name
            resolved["network"][key_name] = profile_data

        # タスク設定の読み込み
        tasks_data = self._load_yaml(self.components_dir / "tasks.yaml")
        profile_data = tasks_data[self.active_task].copy()
        profile_data["profile_name"] = self.active_task
        resolved["task"] = profile_data
        # ★ 最後にPydanticモデルに流し込んで検証（Validation）を行う
        try:
            validated_config = AppConfig(**resolved)
            return validated_config
        except Exception as e:
            raise ValueError(f"Config validation failed: {e}")

    def save_resolved(self, resolved_config: AppConfig, save_dir: str = "results") -> Path:
        """実験の証拠として、結合済みのコンフィグを保存する"""
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = resolved_config.meta.timestamp
        out_path = out_dir / f"config_{timestamp}.yaml"
        
        # Pydanticモデルを辞書に変換して保存
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(resolved_config.model_dump(), f, default_flow_style=False, sort_keys=False)
        
        return out_path
    


if __name__ == "__main__":
    import json

    # 1. ConfigManagerの初期化
    # test.yamlを読み込み、active_taskとして 'pqn_test' を指定
    manager = ConfigManager(config_source="test.yaml", active_task="lif_test")

    # 2. 設定の統合と検証を実行
    print("--- Resolving Config ---")
    config = manager.resolve()

    # 3. 読み込み結果の確認
    print(f"Successfully loaded timestamp: {config.meta.timestamp}")
    print(f"Simulation DT: {config.simulation.dt} ms")
    # print(f"Backend: {config.backend if hasattr(config, 'backend') else config.simulation.backend}")

    # 4. ネストされたデータのアクセス確認
    # NetworkConfig 内の各コンポーネントが正しく展開されているか
    print(f"Weight Type: {config.network.weight.profile_name}")
    
    # 5. タスク設定が正しく読み込まれているか
    if config.task:
        print(f"Active Task Duration: {config.task.duration} ms")

    # 6. 保存機能のテスト
    save_path = manager.save_resolved(config)
    print(f"--- Resolved config saved to: {save_path} ---")

    # デバッグ用：全データの構造を表示（辞書形式）
    # print(json.dumps(config.model_dump(), indent=2, ensure_ascii=False))
