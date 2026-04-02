# src/models/topology/delays.py
import numpy as np
from pydantic import BaseModel, Field, ValidationError
from src.core.registry import DELAY_MODELS

# ---------------------------------------------------------
# 1. スキーマ定義 (ファイル内自己完結)
# ---------------------------------------------------------
class EmpiricalDelaySchema(BaseModel):
    mean: float = Field(..., gt=0.0, description="遅延の平均値(ms)")
    std: float = Field(..., gt=0.0, description="遅延の標準偏差(ms)")
    min_delay: float = Field(..., ge=0.0, description="遅延の下限(ms)")
    max_delay: float = Field(..., gt=0.0, description="遅延の上限(ms)")

# ---------------------------------------------------------
# 2. ビルダークラス (Registryへの登録)
# ---------------------------------------------------------
@DELAY_MODELS.register("normal")
class NormalDelayBuilder:
    def __init__(self, raw_config: dict, N: int, rng: np.random.RandomState):
        try:
            # YAMLの "min" や "max" などのキーをスキーマに合わせてマッピング
            mapped_config = {
                "mean": raw_config.get("mean"),
                "std": raw_config.get("std"),
                "min_delay": raw_config.get("min"),
                "max_delay": raw_config.get("max")
            }
            self.cfg = EmpiricalDelaySchema(**mapped_config)
        except ValidationError as e:
            raise ValueError(f"[NormalDelayBuilder] Configuration error:\n{e}")
        
        self.N = N
        self.rng = rng

    def generate(self, mask: np.ndarray) -> np.ndarray:
        """指定された平均7ms、標準偏差1msの分布などを生成し、範囲内に収める"""
        delays = self.rng.normal(
            loc=self.cfg.mean, 
            scale=self.cfg.std, 
            size=(self.N, self.N)
        )
        
        # 物理的にあり得ない負の遅延や、想定外に大きすぎる遅延を防ぐため、
        # 4〜10msの範囲などでクリップする（厳密な制限ではなく範囲の担保）
        delays = np.clip(delays, self.cfg.min_delay, self.cfg.max_delay)
        
        # intへのキャストや、結合がない部分(mask==0)をゼロにする処理
        delays_int = delays.astype(np.int32)
        return delays_int * (mask != 0)