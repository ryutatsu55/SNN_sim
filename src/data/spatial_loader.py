import os
import numpy as np
from typing import Dict, Iterator, Tuple, Any, List

# レジストリと基底クラスのインポート
from src.core.registry import DATA_LOADERS
from src.data.base_loader import BaseDataLoader
from src.core.config_manager import AppConfig

@DATA_LOADERS.register("spatial_recognition")
class SpatialRecognitionLoader(BaseDataLoader):
    def __init__(self, config: 'AppConfig', io_map: dict):
        super().__init__(config, io_map)
        
        self.num_classes = 3
        self.output_dir = "RNN_analyze/reservoir_outputs"   # TODO 要修正
        
        # 固有のステップ数計算
        self.stim_duration = config.task.input.duration_stim
        self.teaching_duration = config.task.output.teaching_duration
        self.teaching_steps = int(self.teaching_duration / self.dt)
        
        self.n_train = config.task.experiment.n_train
        self.n_test = config.task.experiment.n_test
        self.input_strength = config.task.input.strength

        # 空間マッピングとキャッシュ作成
        self._prepare_spatial_cache()

    def _prepare_spatial_cache(self):
        """固有の重い初期化処理はプライベートメソッドに分離"""
        neurons_per_area = self.total_neurons // 6
        self.area_names = ["top", "middle", "bottom"]
        self.area_indices = {
            name: np.arange(i * neurons_per_area, (i + 1) * neurons_per_area)
            for i, name in enumerate(self.area_names)
        }
        
        self.cached_inputs = {"zero": self.format_global_to_group(np.zeros(self.total_neurons, dtype=np.float32))}
        for name, indices in self.area_indices.items():
            stim_global = np.zeros(self.total_neurons, dtype=np.float32)
            stim_global[indices] = self.input_strength
            self.cached_inputs[name] = self.format_global_to_group(stim_global)
        
        print(f"  [DataLoader] SpatialRecognitionLoader Ready.")
        print(f"               (Train samples: {self.n_train}, Test samples: {self.n_test}, Total neurons: {self.total_neurons})")

    def create_target_signal(self, label: int, total_steps: int, delay_steps: int = 0) -> np.ndarray:
        """
        形状 (total_steps, num_classes) の教師信号（One-hot）を作成
        
        Args:
            label (int): 正解クラスのインデックス
            total_steps (int): トライアル全体のステップ数
            delay_steps (int, optional): 教師信号の開始を遅らせるステップ数. Defaults to 0.
        """
        target = np.zeros((total_steps, self.num_classes), dtype=np.float32)
        
        # 遅延を考慮した開始・終了インデックスの計算
        start_idx = delay_steps
        end_idx = min(delay_steps + self.teaching_steps, total_steps)
        
        if start_idx < total_steps:
            target[start_idx:end_idx, label] = 1.0
            
        return target

    def generate(self) -> Iterator[Tuple[List[Tuple[Dict[str, np.ndarray], int]], Dict[str, Any]]]:
        """
        指定されたデータを、1トライアルごとに逐次生成して yield する。
        """
        trials = {"train": self.n_train, "test": self.n_test}
        
        for key, num_samples in trials.items():
            for trial_idx in range(num_samples):
                total_steps = self.total_steps if key == "train" else self.teaching_steps
                
                # 1. ラベル（刺激エリア）の決定
                target_label = self.rng.randint(0, self.num_classes)
                target_area_name = self.area_names[target_label]
                
                # 2. 事前計算されたキャッシュから入力を構築
                inputs = [
                    (self.cached_inputs[target_area_name], self.stim_duration),
                    (self.cached_inputs["zero"], total_steps - self.stim_duration)
                ]
                
                # 3. 保存先パス等のメタデータ生成
                filename = f"{trial_idx}.npy"
                save_dir = os.path.join(self.output_dir, key, target_area_name)
                save_path = os.path.join(save_dir, filename)
                
                metadata = {
                    "phase": key,
                    "trial_idx": trial_idx,
                    "label": target_label,
                    "class": target_area_name,
                    "total_steps": total_steps,
                    "save_dir": save_dir,
                    "save_path": save_path
                }
                
                yield inputs, metadata
                