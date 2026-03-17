import yaml
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config_data = yaml.safe_load(f)

    def get_data(self) -> dict:
        """設定データを辞書として返す"""
        return self.config_data