import numpy as np
from pydantic import BaseModel, Field, ValidationError
from src.core.registry import DATA_LOADERS

# ---------------------------------------------------------
# 1. このタスク専用の型定義（スキーマ）をファイル内に配置
# ---------------------------------------------------------
class SpatialTaskSchema(BaseModel):
    """空間認識タスクに要求されるパラメータの型と制約"""
    input_nodes: int = Field(..., gt=0, description="入力ノード数")
    duration_stim: float = Field(0.1, gt=0.0, description="刺激の持続時間")
    input_dt: float = Field(0.01, gt=0.0, description="入力データのタイムステップ")
    teaching_duration: float = Field(2.5, gt=0.0)
    duration_interval: float = Field(10.0, gt=0.0)

# ---------------------------------------------------------
# 2. データローダークラス本体
# ---------------------------------------------------------
@DATA_LOADERS.register("spatial_recognition")
class SpatialDataLoader:
    def __init__(self, raw_config: dict):
        # YAMLから渡された生の辞書を統合して自身のスキーマで検証
        try:
            # taskブロック以下の input, output, experiment をマージして検証
            merged_config = {
                **raw_config.get("input", {}),
                **raw_config.get("output", {}),
                **raw_config.get("experiment", {})
            }
            self.cfg = SpatialTaskSchema(**merged_config)
        except ValidationError as e:
            # 必須パラメータの欠如や型の不一致があればここでクラッシュさせる
            raise ValueError(f"[SpatialDataLoader] Configuration error:\n{e}")

    def load_data(self):
        """
        以前の make_spatial_input.py や recognition_test.py 内の
        load_and_process_data() のロジックをここに移植します。
        """
        print(f"[DataLoader] Setup Spatial Data:")
        print(f"  - Nodes: {self.cfg.input_nodes}")
        print(f"  - Input dt: {self.cfg.input_dt}")
        print(f"  - Teaching duration: {self.cfg.teaching_duration}s")
        
        # ※本来はここでデータを生成・ロードする
        X_train = np.random.rand(10, int(self.cfg.duration_interval / self.cfg.input_dt), self.cfg.input_nodes)
        Y_train = np.random.randint(0, 3, 10)
        X_test = np.random.rand(5, int(self.cfg.duration_interval / self.cfg.input_dt), self.cfg.input_nodes)
        Y_test = np.random.randint(0, 3, 5)
        
        return X_train, Y_train, X_test, Y_test