import numpy as np
from typing import Dict, Iterator, Tuple, Any, List

from src.core.registry import DATA_LOADERS
from src.data.base_loader import BaseDataLoader
from src.core.config_manager import AppConfig

@DATA_LOADERS.register("pqn_test")
class pqn_test_loader(BaseDataLoader):
    def __init__(self, config: 'AppConfig', group_info: dict):
        super().__init__(config, group_info)
        
        # ターゲットとなるポピュレーション名とニューロン数を取得
        self.num_neurons = sum(info["num"] for _, info in self.group_info.items())
        
        # 入力電流の強度
        self.input_current = self.config.task.input
        
        # --- キャッシュ化: 2つの状態（ゼロ電流、刺激電流）を事前作成 ---
        self.zero_array = np.zeros(self.num_neurons, dtype=np.float32)
        self.stim_array = np.full(self.num_neurons, self.input_current, dtype=np.float32)
        
        # ステップ数の計算 (1/4 から 3/4 までの期間に電流を注入)
        self.steps_off_1 = self.total_steps // 4
        self.steps_on = self.total_steps // 2
        self.steps_off_2 = self.total_steps - self.steps_off_1 - self.steps_on
        
        print(f"  [DataLoader] pqn_test Ready.")
        print(f"               (Neurons: {self.num_neurons}, Total steps: {self.total_steps})")

    def generate(self) -> Iterator[Tuple[List[Tuple[Dict[str, np.ndarray], int]], Dict[str, Any]]]:
        """
        テスト用の1トライアル分のデータを生成する。
        """
        # 事前作成した配列を使って、状態と継続ステップ数のタプルを構築
        inputs = [
            (self.zero_array, self.steps_off_1),
            (self.stim_array, self.steps_on),
            (self.zero_array, self.steps_off_2)
        ]
        
        # テスト用なのでメタデータはシンプルに
        metadata = {
            "phase": "test",
            "trial_idx": 0,
            "total_steps": self.total_steps
        }
        
        yield inputs, metadata

    def reconstruct(self, inputs):
        """
        シミュレータに渡されるチャンク形式の inputs を (total_steps, num_neurons) の 2D配列に復元する。

        Args:
            inputs: [(input, DurationSteps), ...] の形式のリスト
            target_pop: 抽出対象のニューロンポピュレーション名 (例: "input_pop")

        Returns:
            (total_steps, num_neurons) の形状を持つ 2D NumPy 配列
        """
        if not inputs:
            return np.array([])

        # 1. トータルステップ数の計算
        total_steps = sum(duration for _, duration in inputs)

        num_neurons = self.num_neurons

        # 3. 結果を格納する配列を初期化 (データ型は元の入力に合わせるのがベストですが、通常はfloat32)
        result_array = np.zeros((total_steps, num_neurons))

        # 4. データを時間軸に沿って埋めていく
        current_step = 0
        for input, duration in inputs:
            # Numpyのブロードキャストを利用して、duration行分に一括代入
            result_array[current_step : current_step + duration, :] = input
            
            current_step += duration

        return result_array