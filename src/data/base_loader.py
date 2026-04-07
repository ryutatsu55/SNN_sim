from abc import ABC, abstractmethod
from typing import Dict, Iterator, Tuple, Any, List
import numpy as np

from src.core.config_manager import AppConfig

class BaseDataLoader(ABC):
    def __init__(self, config: 'AppConfig', io_map: dict):
        self.config = config
        self.io_map = io_map
        self.input_map = io_map.get("inputs", {})
        
        # ネットワーク全体のニューロン数を取得
        self.total_neurons = io_map.get("meta", {}).get("total_neurons", 0)
        
        self.dt = self.config.simulation.dt
        self.duration = config.task.duration
        self.total_steps = int(self.duration / self.dt)
        
        seed = getattr(self.config.simulation, 'seed', 42)
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def generate(self) -> Iterator[Tuple[List[Tuple[Dict[str, np.ndarray], int]], Dict[str, Any]]]:
        """
        【候補1のインターフェース】
        データを、意味のある単位（1トライアル等）で逐次生成する。
        """
        pass

    def load_all(self) -> Tuple[List[List[Tuple[Dict[str, np.ndarray], int]]], List[Dict[str, Any]]]:
        """
        【候補2のインターフェース】（基底クラスで共通実装）
        generate() を最後まで回し、すべての結果をリスト化して一括で返す。
        """
        all_inputs = []
        all_metadata = []
        
        for inputs, metadata in self.generate():
            all_inputs.append(inputs)
            all_metadata.append(metadata)
            
        return all_inputs, all_metadata

    def format_global_to_group(self, global_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """
        [共通インターフェース]
        グローバルインデックスベースのデータ(形状: [total_neurons])を受け取り、
        Simulatorが受け付けるグループベースの辞書 {"pop_name": tensor} に変換する。
        """
        inputs = {}
        for pop_name, info in self.input_map.items():
            # そのグループが担当するグローバルインデックスのリストを取得
            indices = info["global_indices"]
            # グローバル空間から抽出して割り当て
            inputs[pop_name] = global_tensor[indices]
            
        return inputs
