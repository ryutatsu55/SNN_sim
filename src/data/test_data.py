import numpy as np
from typing import Dict, Iterator, Tuple, Any, List

from src.core.registry import DATA_LOADERS
from src.data.base_loader import BaseDataLoader
from src.core.config_manager import AppConfig

@DATA_LOADERS.register("pqn_test")
class pqn_test_loader(BaseDataLoader):
    def __init__(self, config: 'AppConfig', group_info: dict):
        super().__init__(config, group_info)
        
        # 入力電流の強度
        self.input_current = self.config.task.input
        
        # --- キャッシュ化: 2つの状態（ゼロ電流、刺激電流）を事前作成 ---
        self.zero_array = np.zeros(self.total_neurons, dtype=np.float32)
        self.stim_array = np.full(self.total_neurons, self.input_current, dtype=np.float32)
        
        # ステップ数の計算 (1/4 から 3/4 までの期間に電流を注入)
        self.steps_off_1 = self.total_steps // 4
        self.steps_on = self.total_steps // 2
        self.steps_off_2 = self.total_steps - self.steps_off_1 - self.steps_on
        
        print(f"  [DataLoader] pqn_test Ready.")
        print(f"               (Neurons: {self.total_neurons}, Total steps: {self.total_steps})")

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
        total_steps = self.total_steps

        num_neurons = self.total_neurons

        # 3. 結果を格納する配列を初期化 (データ型は元の入力に合わせるのがベストですが、通常はfloat32)
        result_array = np.zeros((total_steps, num_neurons))

        # 4. データを時間軸に沿って埋めていく
        current_step = 0
        for input, duration in inputs:
            # Numpyのブロードキャストを利用して、duration行分に一括代入
            result_array[current_step : current_step + duration, :] = input
            
            current_step += duration

        return result_array
    
@DATA_LOADERS.register("lif_test")
class lif_test(BaseDataLoader):
    def __init__(self, config: 'AppConfig', group_info: dict):
        super().__init__(config, group_info)
        
        self.zero_array = np.full(self.total_neurons, self.config.neurons["Layer_Exc"].Vrest, dtype=np.float32)
        self.stim_array = np.copy(self.zero_array)
        self.stim_array[self.config.task.tgt_ID] = self.config.task.input
        
        self.devided_steps = self.total_steps // self.config.task.devide
        
        print(f"  [DataLoader] pqn_test Ready.")
        print(f"               (Neurons: {self.total_neurons}, Total steps: {self.total_steps})")

    def generate(self) -> Iterator[Tuple[List[Tuple[Dict[str, np.ndarray], int]], Dict[str, Any]]]:
        """
        テスト用の1トライアル分のデータを生成する。
        """

        inputs = [(self.stim_array, self.devided_steps) for _ in range(self.config.task.devide)]
        # inputs[0] = (self.zero_array, self.devided_steps)
        # テスト用なのでメタデータはシンプルに
        metadata = {
            "phase": "test",
            "trial_idx": 0,
            "total_steps": self.total_steps
        }
        
        yield inputs, metadata

@DATA_LOADERS.register("stdp_test")
class stdp_test(BaseDataLoader):
    def __init__(self, config: 'AppConfig', group_info: dict):
        super().__init__(config, group_info)
        
        # --- ハードコーディングされたパラメータ (適宜修正してください) ---
        self.pre_id = config.network.connection.src_ID   # 刺激を与えるPreニューロンのグローバルID
        self.post_id = config.network.connection.tgt_ID   # 刺激を与えるPostニューロンのグローバルID
        self.stim_intensity = self.config.task.input  # 発火させるための瞬間的な入力 (mVなど)
        dt_max = config.task.dt_max
        dt_min = config.task.dt_min
        ddt = config.task.ddt
        
        # Δt の範囲設定 (ms単位)
        # -50msから50msまで 2ms刻みで測定する例
        self.dt_values_ms = np.arange(dt_min, dt_max + ddt, ddt) 
        self.dt_steps = (self.dt_values_ms / self.config.simulation.dt).astype(int)
        
        # 1回発火させた後の余韻（重み変化が完了するのを待つ時間）のステップ数
        # post_stim_steps = 500 
        # -----------------------------------------------------------
        
        # 基本配列の作成
        layer = (
            self.config.neurons.get("Layer_Exc")
            or self.config.neurons.get("Layer_Inh")
        )

        self.base_array = np.full(
            self.total_neurons,
            layer.Vrest,
            dtype=np.float32
        )
        
        # 刺激用配列（Preのみ、Postのみ）
        self.stim_pre = np.copy(self.base_array)
        self.stim_pre[self.pre_id] = self.stim_intensity
        
        self.stim_post = np.copy(self.base_array)
        self.stim_post[self.post_id] = self.stim_intensity

        self.both_stim = np.copy(self.base_array)
        self.both_stim[self.pre_id] = self.stim_intensity
        self.both_stim[self.post_id] = self.stim_intensity

        print(f"  [DataLoader] stdp_test Ready.")
        print(f"               (Testing {len(self.dt_steps)} points of Delta-t)")

    def generate(self) -> Iterator[Tuple[List[Tuple[Dict[str, np.ndarray], int]], Dict[str, Any]]]:
        """
        各Δtごとに1トライアルとしてデータを生成する。
        """
        for i, dt_step in enumerate(self.dt_steps):
            inputs = []
            
            if dt_step > 0:
                # Pre -> Post の順 (LTP側)
                # 1. Preを1ステップだけ刺激
                inputs.append((self.stim_pre, dt_step))
                # 3. Postを1ステップだけ刺激
                inputs.append((self.stim_post, self.total_steps-dt_step))
                
            elif dt_step < 0:
                # Post -> Pre の順 (LTD側)
                abs_dt = abs(dt_step)
                # 1. Postを1ステップだけ刺激
                inputs.append((self.stim_post, abs_dt))
                # 3. Preを1ステップだけ刺激
                inputs.append((self.stim_pre, self.total_steps-abs_dt))
                
            else:
                # Δt = 0 (同時)
                inputs.append((self.both_stim, self.total_steps))

            # 刺激後の待機時間（重み更新処理が確実に走るようにする）
            # inputs.append((self.base_array, post_stim_steps))

            # メタデータに現在のΔtを乗せる
            metadata = {
                "phase": "stdp_validation",
                "trial_idx": i,
                "delta_t_ms": self.dt_values_ms[i],
                "delta_t_steps": dt_step
            }
            
            yield inputs, metadata
