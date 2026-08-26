import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from src.core.registry import SPATIAL_MODELS
from pathlib import Path

# 型注釈のためだけでなく、**AREA_MODELS の登録デコレータを発火させるための import** でもある。
# 既存のエントリポイントはどれも `import src.models.network.space` を持っているので、
# ここで area を引き込んでおけば全スクリプトでエリアが解決できる。
from .area import BaseArea

class BaseSpace(ABC):
    """空間座標を生成する基底クラス"""
    def __init__(self, config: Dict[str, Any], num_neurons: int, rng: np.random.RandomState, layout=None,
                 area: Optional[BaseArea] = None):
        self.config = config
        self.num_neurons = num_neurons
        self.rng = rng
        # NetworkLayout。ニューロン種ごとの意図的バイアスや無相関化(シャッフル)を
        # 具象クラス側で実装したい場合に self.layout.ids_by("polarity") などを参照する。
        self.layout = layout
        # BaseArea。soma を配置できる領域で、配置は area.sample(n, self.rng) を呼ぶ。
        # エリア自身は rng を持たないので、乱数の消費はこの呼び出し順にだけ依存する。
        self.area = area

    def describe_axes(self) -> Dict[str, Any]:
        """任意フック: この空間モデルが定義するカテゴリ/ソート軸を宣言する。

        NetworkBuilder が `generate()` の**直後**に呼び、戻り値を
        `NetworkLayout.add_axis()` へ注入する。生成後に呼ばれるので、自身が生成した
        `self.coords` から軸を導出してよい(例: 空間モジュール、深さ、中心からの距離)。

        Returns:
            {軸名: 長さ num_neurons の配列}。カテゴリ名(文字列)でもソート用の数値でも
            よい。既定は空 dict = 軸を定義しない。
        """
        return {}

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

@SPATIAL_MODELS.register("random_2d_n1000")
class Random2DSpaceN1000(Random2DSpace):
    """Beggs&Plenz 検証ラダー T0 (N=1000) 用の縮小空間。範囲は YAML から読む。"""
    pass

@SPATIAL_MODELS.register("random_2d_n4000")
class Random2DSpaceN4000(Random2DSpace):
    """Beggs&Plenz 検証ラダー T1 (N=4000) 用の縮小空間。範囲は YAML から読む。"""
    pass

@SPATIAL_MODELS.register("random_2d_n10000")
class Random2DSpaceN10000(Random2DSpace):
    """Beggs&Plenz 検証ラダー T2 (N=10000) 用の縮小空間。範囲は YAML から読む。"""
    pass

@SPATIAL_MODELS.register("random_circle_2d")
class RandomCircle2DSpace(BaseSpace):
    def generate(self):
        """半径 r の円盤内に一様乱数で2次元座標を生成する。

        単純に半径を uniform(0, r) で引くと面積要素 (r dr dθ) を無視するため
        中心に密集する。面積一様にするには r = R*sqrt(u) と補正する。
        """
        R = self.config.r

        radius = R * np.sqrt(self.rng.uniform(0.0, 1.0, self.num_neurons))
        theta = self.rng.uniform(0.0, 2.0 * np.pi, self.num_neurons)

        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)
        coords[:, 0] = radius * np.cos(theta)
        coords[:, 1] = radius * np.sin(theta)

        return coords

@SPATIAL_MODELS.register("area_uniform")
class AreaUniformSpace(BaseSpace):
    """`network.area` が定義する領域内に一様ランダムに配置する空間モデル。

    形状の知識は一切持たず、すべてエリアに委譲する(`area.sample()`)。円/矩形のような
    単純形状では解析的サンプリング、`composite` のような複雑形状では棄却サンプリングが
    エリア側で選ばれる。パラメータは空(形は areas.yaml が決める)。

    エリアが `part_of()` を持つ(= `CompositeArea`)場合、各ニューロンがどの part に
    落ちたかを **`module` 軸**として宣言する。これによりモジュール構造は専用の空間クラスを
    書かなくても、エリアの定義だけから自動的に得られる。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._module_labels: Optional[np.ndarray] = None

    def describe_axes(self) -> Dict[str, Any]:
        if self._module_labels is None:
            return {}
        return {"module": self._module_labels}

    def generate(self):
        if self.area is None:
            raise ValueError(
                "area_uniform は有界な network.area を必要とします。メイン config の network に"
                " area: disk などを指定してください (no_space は無界なので一様サンプリング"
                " できません)。"
            )

        # soma を置ける部分領域。既定は area そのもの(**同一オブジェクト**なので乱数の
        # 消費数も従来と変わらない)。`allow_soma: false` の part がある複合領域では
        # それを除いた union が返る。軸索側は NetworkBuilder が渡した area をそのまま
        # 使い続けるので、「soma はモジュール内、軸索はブリッジも通る」が実現する。
        # ダックタイピングで受けるのは他のエリア参照(part_of など)と同じ作法。
        region = getattr(self.area, "soma_area", self.area)

        xy = region.sample(self.num_neurons, self.rng)

        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)
        coords[:, :2] = xy

        # 実効密度の報告。Sumi et al. (2025) の培養は 400 neurons/mm^2 = 4e-4 /um^2 なので、
        # ニューロン数とエリアの大きさが噛み合っていない config に早く気づけるようにする。
        # 分母は **soma 配置領域**の面積 — N はそこにしか居ないので、これが実効密度になる。
        area_um2 = region.area_um2
        if area_um2:
            density = self.num_neurons / area_um2
            total_um2 = self.area.area_um2 if region is not self.area else None
            extent = (f"soma area={area_um2 * 1e-6:.3f} mm^2 / total {total_um2 * 1e-6:.3f} mm^2"
                      if total_um2 else f"area={area_um2 * 1e-6:.3f} mm^2")
            print(f"    Density: {density * 1e6:.1f} neurons/mm^2 "
                  f"(N={self.num_neurons}, {extent})")

        # module 軸(エリアが複合領域なら、どの part に落ちたか)。
        # **soma 配置領域の上で採る。** ブリッジがモジュールへ食い込む帯では、モジュール内の
        # 点でも全体領域の part_of は「より深い」ブリッジ part を返しうるので、全体領域で
        # 採ると soma が B0-1 とラベルされてしまう。region の part_names は親から
        # 引き継いだ名前なので、除外前と同じ M0, M1, ... が出る。
        part_of = getattr(region, "part_of", None)
        if callable(part_of):
            names = getattr(region, "part_names", None)
            idx = part_of(xy)
            self._module_labels = (
                np.asarray(names, dtype=object)[idx].astype(str) if names is not None
                else np.array([f"M{i}" for i in idx])
            )

        return coords

@SPATIAL_MODELS.register("block_2d")
class Block2DSpace(BaseSpace):
    """矩形モジュールに分割して一様ランダム配置する空間モデル。

    モジュール分割は自身の `config.num_modules` で決める(NetworkLayout の軸を読みに
    行かない)。分割結果は `describe_axes()` で **`module` 軸**として宣言し、
    NetworkBuilder 経由で NetworkLayout に注入される。以降の結合生成や解析は
    `layout.ids_by("module")` でこれを参照できる。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._module_labels: Optional[np.ndarray] = None

    def describe_axes(self) -> Dict[str, Any]:
        """モジュール分割を `module` 軸として宣言する(generate() 後に有効)。"""
        if self._module_labels is None:
            return {}
        return {"module": self._module_labels}

    def generate(self):
        """四角形のモジュール領域を定義し、一様ランダムに配置する"""
        x_range = self.config.x_range
        y_range = self.config.y_range

        gap_ratio = self.config.margin

        coords = np.zeros((self.num_neurons, 3), dtype=np.float32)

        # --- 1. モジュール分割 (自身の config で完結。層の概念とは独立) ---
        num_modules = self.config.num_modules
        neurons_per_module = np.full(num_modules, self.num_neurons // num_modules)
        # 余りが出た場合、先頭のモジュールから順に1つずつ追加して吸収する
        neurons_per_module[:self.num_neurons % num_modules] += 1

        # module 軸のラベル(describe_axes で NetworkLayout へ渡す)
        self._module_labels = np.repeat(
            [f"M{i}" for i in range(num_modules)], neurons_per_module
        )

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
    - ``describe_axes()`` … ``Layer`` 列を **``layer`` 軸**として宣言する。

    E/I を層と無相関に散らしたい場合は config 側で ``layout.assignment: random`` を
    指定する(population 軸の割当は NetworkLayout 自身の管轄であり、空間モデルは
    関与しない)。
    """

    _CSV_NAME = "ordered_coords.csv"

    @classmethod
    def _csv_path(cls) -> Path:
        return Path(__file__).parent / "data" / "c_elegans" / cls._CSV_NAME

    @classmethod
    def _load_ordered(cls, num_neurons: int) -> pd.DataFrame:
        """CSV を読み、先頭 num_neurons 行(= 正準グローバルID順)を返す。

        generate() と describe_axes() が同じ行集合・同じ順序を共有し、座標と層ラベルの
        整合を保証するための単一の読み込み口。
        """
        df = pd.read_csv(cls._csv_path())
        if num_neurons > len(df):
            raise ValueError(
                f"num_neurons ({num_neurons}) exceeds available neurons in "
                f"C. elegans data ({len(df)})"
            )
        return df.iloc[:num_neurons].reset_index(drop=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._layer_labels: Optional[np.ndarray] = None

    def describe_axes(self) -> Dict[str, Any]:
        """CSV の Layer 列を `layer` 軸として宣言する(generate() 後に有効)。"""
        if self._layer_labels is None:
            return {}
        return {"layer": self._layer_labels}

    def generate(self) -> np.ndarray:
        """C. elegans のニューロン座標データから実 3D 座標を読み込む(CSV 行順)。"""
        df = self._load_ordered(self.num_neurons)
        self._layer_labels = df["Layer"].to_numpy()
        return df[['X', 'Y', 'Z']].values.astype(np.float32)
