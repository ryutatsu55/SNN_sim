import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from src.core.registry import SPATIAL_MODELS
from src.core.layout import LayoutPlan
from pathlib import Path

class BaseSpace(ABC):
    """空間座標を生成する基底クラス"""
    def __init__(self, config: Dict[str, Any], num_neurons: int, rng: np.random.RandomState, layout=None):
        self.config = config
        self.num_neurons = num_neurons
        self.rng = rng
        # NetworkLayout。ニューロン種ごとの意図的バイアスや無相関化(シャッフル)を
        # 具象クラス側で実装したい場合に self.layout.items() / ids_by_mode() を参照する。
        self.layout = layout

    @classmethod
    def describe_layout(cls, config, total_neurons: int) -> Optional[LayoutPlan]:
        """任意フック: この空間モデルが構造層/割当方式を提供する場合に LayoutPlan を返す。

        `NetworkLayout.from_config` が(明示的な config.layout が無いフィールドについて)
        参照する。空間構造そのものが構造層に一致するモデル(データ由来の層、モジュール
        分割など)は、これをオーバーライドして層や `assignment` を宣言できる。

        - 既定は None → config.layout / 既定挙動(sequential・単一層)にフォールバック。
        - **決定論的に**導出すること(GeNN ビルド無しの再構築性を保つため)。データ
          ファイルや自身の config(space.yaml)から層数・層サイズ・割当を決めてよい。
        """
        return None

    @abstractmethod
    def generate(self) -> Optional[np.ndarray]:
        """
        Returns:
            np.ndarray: 形状 (num_neurons, D) の座標配列
        """
        pass

@SPATIAL_MODELS.register("no_space")
class NoSpace(BaseSpace):
    def generate(self):
        """空間配置を必要としない場合は None を返す"""
        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)
        coords[:,:] = np.nan

        return coords

@SPATIAL_MODELS.register("grid_2d")
class Grid2DSpace(BaseSpace):
    def generate(self):
        """指定された範囲内に一定間隔で2次元座標を生成する"""
        x_range = self.config.x_range
        y_range = self.config.y_range
        
        nx = int(np.ceil(np.sqrt(self.num_neurons)))
        ny = int(np.ceil(self.num_neurons / nx))
        
        # 各軸の一定間隔の座標を生成
        x_steps = np.linspace(x_range[0], x_range[1], nx)
        y_steps = np.linspace(y_range[0], y_range[1], ny)
        
        # メッシュグリッドを作成
        X, Y = np.meshgrid(x_steps, y_steps)
        
        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)
        
        # 生成したグリッド座標を1次元に平坦化し、必要なニューロン数だけ切り出して割り当て
        coords[:, 0] = X.ravel()[:self.num_neurons]
        coords[:, 1] = Y.ravel()[:self.num_neurons]
        
        return coords
    
@SPATIAL_MODELS.register("random_2d")
class Random2DSpace(BaseSpace):
    def generate(self):
        """指定された範囲内に一様乱数で2次元座標を生成する"""

        x_range = self.config.x_range
        y_range = self.config.y_range

        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)
        coords[:, 0] = self.rng.uniform(x_range[0], x_range[1], self.num_neurons)
        coords[:, 1] = self.rng.uniform(y_range[0], y_range[1], self.num_neurons)

        return coords

@SPATIAL_MODELS.register("block_2d")
class Block2DSpace(BaseSpace):
    def generate(self):
        """四角形のモジュール領域を定義し、一様ランダムに配置する"""
        x_range = self.config.x_range
        y_range = self.config.y_range
        
        gap_ratio = self.config.margin

        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)

        # --- 1. モジュール分割: NetworkLayout に複数の構造層があればそれを採用 ---
        #        (層 = モジュール。層が無い/単一のときは従来通り num_modules で均等分割)
        layers = self.layout.layers() if self.layout is not None else None
        if layers is not None and len(layers) > 1:
            num_modules = len(layers)
            neurons_per_module = np.array([l.num for l in layers], dtype=int)
        else:
            num_modules = self.config.num_modules
            neurons_per_module = np.full(num_modules, self.num_neurons // num_modules)
            # 余りが出た場合、先頭のモジュールから順に1つずつ追加して吸収する
            neurons_per_module[:self.num_neurons % num_modules] += 1

        # --- 2. グリッドの分割数とモジュール幾何 ---
        nx = int(np.ceil(np.sqrt(num_modules)))
        ny = int(np.ceil(num_modules / nx))

        total_w = x_range[1] - x_range[0]
        total_h = y_range[1] - y_range[0]

        gap_x = (total_w * gap_ratio)
        gap_y = (total_h * gap_ratio)

        module_w = (total_w - gap_x) / nx
        module_h = (total_h - gap_y) / ny
        
        # --- 3. 四角形領域の計算と内部へのランダム散布（連番割り当て） ---
        current_idx = 0
        for i in range(num_modules):
            n = neurons_per_module[i]
                
            # グリッド上のインデックス (ix, iy)
            ix = i % nx
            iy = i // nx
            
            # このモジュールの描画範囲 (四角形の境界) を計算
            min_x = x_range[0] + ix * (module_w + gap_x/nx) + gap_x/(2*nx)
            max_x = min_x + module_w
            
            min_y = y_range[0] + iy * (module_h + gap_y/ny) + gap_y/(2*ny)
            max_y = min_y + module_h
            
            # 連番のインデックス範囲に対して、四角形の範囲内に一様分布でランダムに散布
            coords[current_idx:current_idx+n, 0] = self.rng.uniform(min_x, max_x, size=n)
            coords[current_idx:current_idx+n, 1] = self.rng.uniform(min_y, max_y, size=n)
            
            # 次のモジュールへインデックスを進める
            current_idx += n
            
        return coords
    
@SPATIAL_MODELS.register("C.elegans")
class C_elegansSpace(BaseSpace):
    """C. elegans コネクトームの実座標を提供する空間モデル。

    ``ordered_coords.csv`` は ``Layer`` 列(IN1..IN4 など)で **層順に整列済み** で、
    行順がそのまま正準グローバルID順になる。したがって:

    - ``generate()`` … X,Y,Z の実 3D 座標(CSV 行順)を返す。
    - ``describe_layout()`` … ``Layer`` 列の連続ランを構造層(ブロック)として宣言し、
      興奮性/抑制性(population)は **ランダム割当** (``assignment="random"``) とする。
      これにより接続/重み行列は層でブロック対角に生成されつつ、E/I は層と無相関に散る。
    """

    _CSV_NAME = "ordered_coords.csv"

    @classmethod
    def _csv_path(cls) -> Path:
        return Path(__file__).parent / "data" / "c_elegans" / cls._CSV_NAME

    @classmethod
    def _load_ordered(cls, num_neurons: int) -> pd.DataFrame:
        """CSV を読み、先頭 num_neurons 行(= 正準グローバルID順)を返す。

        generate() と describe_layout() が同じ行集合・同じ順序を共有し、座標と層境界の
        整合を保証するための単一の読み込み口。
        """
        df = pd.read_csv(cls._csv_path())
        if num_neurons > len(df):
            raise ValueError(
                f"num_neurons ({num_neurons}) exceeds available neurons in "
                f"C. elegans data ({len(df)})"
            )
        return df.iloc[:num_neurons].reset_index(drop=True)

    @classmethod
    def describe_layout(cls, config, total_neurons: int) -> LayoutPlan:
        """Layer 列の連続ランを構造層とし、E/I をランダム割当にするプランを返す。"""
        df = cls._load_ordered(total_neurons)

        # Layer 列の連続ラン → [(層名, ニューロン数), ...]。データは層順に整列済み。
        pairs: List[tuple] = []
        for lname in df["Layer"].tolist():
            if pairs and pairs[-1][0] == lname:
                pairs[-1] = (lname, pairs[-1][1] + 1)
            else:
                pairs.append((lname, 1))

        # 同一層名が非連続に出現すると連続ブロック層にできない → 明示的にエラー。
        names = [n for n, _ in pairs]
        if len(names) != len(set(names)):
            raise ValueError(
                f"{cls._CSV_NAME} の Layer 列が層ごとに連続していません "
                f"(出現順: {names})。層順にソートしてください。"
            )

        return LayoutPlan(layers=pairs, assignment="random")

    def generate(self) -> np.ndarray:
        """C. elegans のニューロン座標データから実 3D 座標を読み込む(CSV 行順)。"""
        df = self._load_ordered(self.num_neurons)
        return df[['X', 'Y', 'Z']].values.astype(np.float32)
