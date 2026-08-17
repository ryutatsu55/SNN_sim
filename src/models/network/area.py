"""ニューロンを配置し、軸索を閉じ込める 2 次元領域。

領域を **符号付き距離関数 (SDF)** で表すのが設計の要。`sdf(p)` が負なら内部、正なら外部で、
絶対値が境界までの距離になる。この 1 つの関数から

  - 内外判定      … `contains(p) = sdf(p) <= 0`
  - 外向き法線    … `normal(p) = normalize(grad(sdf)(p))`  (中心差分)
  - 一様サンプル  … 境界箱で棄却サンプリング

がすべて導け、さらに **min/max で合成できる** (和 = min、積 = max、差 = max(a, -b))。
そのため単純な円や矩形だけでなく、モジュール構造・トラック・穴あき領域といった任意の
複雑形状を `CompositeArea` ひとつで表現できる。軸索の境界処理 (壁沿いへの偏向) も
法線さえ取れれば形状によらず同じコードで動く。

`BaseSpace` (soma 配置) と `BaseConnection` (軸索の境界処理) の両方に渡される。
"""

import numpy as np
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from src.core.registry import AREA_MODELS


class BaseArea(ABC):
    """2 次元領域の基底クラス。

    **RNG を保持しない**のが設計上の要点。`NetworkBuilder.rng` は空間→結合→重み→遅延で
    順に共有される単一ストリームなので、area が構築時に 1 draw でも引くと下流すべての
    実現が変わってしまう (同じ seed の既存ネットワークが全部別物になる)。ストリームへの
    参照を持たないオブジェクトはそれを消費しえないので、`__init__` は rng を受け取らず、
    乱数は `sample()` の引数としてのみ渡す。

    `describe_axes()` は**意図的に持たない**。`NetworkBuilder._inject_axes()` の契約は
    「そのコンポーネントが今計算したものから導く長さ N のラベル配列」だが、area は座標が
    存在する前に構築され、ニューロンごとの値を何も持たない。軸を宣言するのは座標を持つ
    space 側の仕事 (`AreaUniformSpace` が `part_of()` を使って `module` 軸を出す)。
    """

    def __init__(self, config: Any, num_neurons: int = 0, layout=None):
        self.config = config
        self.num_neurons = num_neurons
        self.layout = layout

    # ------------------------------------------------------------------ 抽象

    @abstractmethod
    def sdf(self, points: np.ndarray) -> np.ndarray:
        """符号付き距離。(m, 2) -> (m,)。負=内部、正=外部、絶対値は境界までの距離。

        合成 (union/intersection/difference) が成り立つのはこの量が距離だから。
        厳密な距離でなくても符号と勾配の向きが正しければ内外判定と偏向は動くが、
        偏向の 1 ステップあたりの寄り方が変わるので、なるべく真の距離を返すこと。
        """

    @property
    @abstractmethod
    def bounds(self) -> np.ndarray:
        """軸平行境界箱 [[xmin, ymin], [xmax, ymax]]。無界なら ±inf。

        棄却サンプリングの箱としても使うので、無駄に大きいと採択率が落ちる。
        """

    # -------------------------------------------------- SDF から導かれる具象実装

    def contains(self, points: np.ndarray) -> np.ndarray:
        """(m, 2) -> (m,) bool。境界上 (sdf == 0) は内部として扱う。"""
        return np.asarray(self.sdf(np.asarray(points, dtype=np.float64))) <= 0.0

    # 法線を中心差分で取るときの刻み [um]。セグメント長 (100 um) よりずっと小さく、
    # かつ float64 の丸めに埋もれない程度の値。
    _NORMAL_EPS = 1e-2

    def normal(self, points: np.ndarray) -> np.ndarray:
        """外向き単位法線。(m, 2) -> (m, 2)。SDF の勾配を中心差分で取って正規化する。

        真の SDF なら勾配は領域の内外を問わずどこでも外向き単位ベクトルなので、
        境界の交点を二分探索で求めなくても、はみ出した点でそのまま評価してよい。
        """
        p = np.asarray(points, dtype=np.float64)
        h = self._NORMAL_EPS
        dx = (self.sdf(p + [h, 0.0]) - self.sdf(p - [h, 0.0])) / (2.0 * h)
        dy = (self.sdf(p + [0.0, h]) - self.sdf(p - [0.0, h])) / (2.0 * h)
        g = np.stack([dx, dy], axis=-1)
        norm = np.linalg.norm(g, axis=-1, keepdims=True)
        # 勾配が消える点 (領域の中心軸など) では向きが定まらない。ここへ来るのは
        # 境界処理の呼び出し元が「外にいる点」を渡す限り起きないが、0 除算は避ける。
        return np.divide(g, norm, out=np.zeros_like(g), where=norm > 1e-12)

    # 棄却サンプリングの 1 回あたりの倍率と上限試行回数。
    _REJECT_OVERSAMPLE = 4
    _REJECT_MAX_ROUNDS = 1000

    def sample(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        """領域内に一様分布する点を n 個返す。(n, 2)。

        汎用実装は境界箱での棄却サンプリング。任意形状で動く代わりに、**消費する乱数の
        個数が形状に依存する**ので、エリアだけを差し替えた 2 つの config は以降の
        重み/遅延の実現も変わる (後方互換の破壊ではないが、比較実験では意識すること)。
        円や矩形のように解析的に引ける形状は `sample()` を上書きしてこれを避ける。
        """
        if n <= 0:
            return np.zeros((0, 2), dtype=np.float64)
        lo, hi = self.bounds
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError(
                f"{type(self).__name__} は無界なので一様サンプリングできません。"
                " network.area に有界な領域 (disk / rect / composite) を指定してください。"
            )

        out: List[np.ndarray] = []
        got = 0
        for _ in range(self._REJECT_MAX_ROUNDS):
            want = n - got
            cand = rng.uniform(lo, hi, size=(want * self._REJECT_OVERSAMPLE + 8, 2))
            hit = cand[self.contains(cand)]
            if hit.size:
                out.append(hit)
                got += len(hit)
            if got >= n:
                break
        else:
            raise RuntimeError(
                f"{type(self).__name__}: 棄却サンプリングが {self._REJECT_MAX_ROUNDS} 回で"
                f" {n} 点に届きませんでした ({got} 点)。領域が境界箱に対して極端に小さいか、"
                " sdf / bounds が食い違っています。"
            )
        return np.concatenate(out, axis=0)[:n]

    # ---------------------------------------------------------------- 付随情報

    @property
    def is_bounded(self) -> bool:
        return bool(np.all(np.isfinite(self.bounds)))

    @property
    def area_um2(self) -> Optional[float]:
        """領域の面積 [um^2]。分からなければ None (密度の表示を諦めるだけ)。"""
        return None


@AREA_MODELS.register("no_space")
class NoSpaceArea(BaseArea):
    """領域の制約を置かない = 無界の平面。`space.yaml` の `no_space` と対になる名前。

    `sdf` が常に -inf なので `contains` は常に True になり、軸索の境界処理は一度も
    発火しない。確率ベースの結合モデル (`constant_prob`, `beggs_plenz` など) は領域を
    参照しないので、それらを使う config はこれを指定する。

    有界でないので `sample()` は使えない。soma をエリア内に配置したい (`area_uniform`)
    場合は `disk` / `rect` / 複合領域を選ぶこと。
    """

    def sdf(self, points: np.ndarray) -> np.ndarray:
        return np.full(len(np.atleast_2d(points)), -np.inf, dtype=np.float64)

    @property
    def bounds(self) -> np.ndarray:
        return np.array([[-np.inf, -np.inf], [np.inf, np.inf]], dtype=np.float64)

    def contains(self, points: np.ndarray) -> np.ndarray:
        return np.ones(len(np.atleast_2d(points)), dtype=bool)

    def sample(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        raise ValueError(
            "network.area='no_space' は無界なので一様サンプリングできません。"
            " network.area に有界な領域 (disk / rect / modular_4 など) を指定してください。"
        )


@AREA_MODELS.register("disk")
class DiskArea(BaseArea):
    """円板。config: `radius` [um]、`center` (省略時は原点)。"""

    def __init__(self, config, num_neurons: int = 0, layout=None):
        super().__init__(config, num_neurons, layout)
        self.radius = float(_get(config, "radius"))
        self.center = np.asarray(_get(config, "center", (0.0, 0.0)), dtype=np.float64)
        if self.radius <= 0:
            raise ValueError(f"disk area の radius は正である必要があります (got {self.radius})")

    def sdf(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return np.linalg.norm(p - self.center, axis=-1) - self.radius

    def normal(self, points: np.ndarray) -> np.ndarray:
        """解析的な外向き法線 (中心から見て放射方向)。"""
        d = np.asarray(points, dtype=np.float64) - self.center
        norm = np.linalg.norm(d, axis=-1, keepdims=True)
        return np.divide(d, norm, out=np.zeros_like(d), where=norm > 1e-12)

    @property
    def bounds(self) -> np.ndarray:
        return np.array([self.center - self.radius, self.center + self.radius])

    @property
    def area_um2(self) -> float:
        return float(np.pi * self.radius ** 2)

    def sample(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        """面積一様に円板内へ配置する。

        単純に半径を uniform(0, r) で引くと面積要素 (r dr dtheta) を無視するため中心に
        密集する。面積一様にするには r = R*sqrt(u) と補正する。

        **ドロー順は `space.RandomCircle2DSpace` と厳密に一致させてある** (uniform(0,1) を
        n 個 → uniform(0,2pi) を n 個)。同じ seed で `space: random_circle_2d` と
        `area: disk` + `space: area_uniform` がビット一致するので、既存 config を
        エリアベースへ移すときに実現が変わらない。順序を変えないこと。
        """
        if n <= 0:
            return np.zeros((0, 2), dtype=np.float64)
        radius = self.radius * np.sqrt(rng.uniform(0.0, 1.0, n))
        theta = rng.uniform(0.0, 2.0 * np.pi, n)
        return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1) + self.center


@AREA_MODELS.register("rect")
class RectArea(BaseArea):
    """軸平行な矩形。config: `x_range` [xmin, xmax]、`y_range` [ymin, ymax] ([um])。"""

    def __init__(self, config, num_neurons: int = 0, layout=None):
        super().__init__(config, num_neurons, layout)
        xr = np.asarray(_get(config, "x_range"), dtype=np.float64)
        yr = np.asarray(_get(config, "y_range"), dtype=np.float64)
        self.lo = np.array([xr[0], yr[0]])
        self.hi = np.array([xr[1], yr[1]])
        if np.any(self.hi <= self.lo):
            raise ValueError(f"rect area の範囲が空です: x_range={xr}, y_range={yr}")
        self._center = 0.5 * (self.lo + self.hi)
        self._half = 0.5 * (self.hi - self.lo)

    def sdf(self, points: np.ndarray) -> np.ndarray:
        # 矩形の標準的な SDF: 外側は角までのユークリッド距離、内側は最も近い辺までの距離。
        q = np.abs(np.asarray(points, dtype=np.float64) - self._center) - self._half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside

    @property
    def bounds(self) -> np.ndarray:
        return np.array([self.lo, self.hi])

    @property
    def area_um2(self) -> float:
        return float(np.prod(self.hi - self.lo))

    def sample(self, n: int, rng: np.random.RandomState) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 2), dtype=np.float64)
        return rng.uniform(self.lo, self.hi, size=(n, 2))


@AREA_MODELS.register("composite")
class CompositeArea(BaseArea):
    """複数の領域を SDF の集合演算で合成する。**複雑な領域はこれ 1 つで作る。**

    config:
        op:    "union" | "intersection" | "difference"
        parts: [{type: <AREA_MODELS のキー>, ...そのクラスのパラメータ}, ...]

    合成規則は SDF の標準的なもの:
        union        = min(d_0, d_1, ...)          … 和 (くっつける)
        intersection = max(d_0, d_1, ...)          … 積 (重なりだけ残す)
        difference   = max(d_0, -d_1, -d_2, ...)   … 差 (先頭から残りを削る = 穴あけ)

    `parts` は入れ子にできる (part 自体を `type: composite` にする) ので、いくらでも
    複雑な形にできる。例: 4 つの円 + 十字のブリッジ = モジュール構造。

    ※ `parts` の各要素は素の dict のまま渡ってくる (`ComponentConfig` は extra='allow'
      なので入れ子は pydantic モデル化されない)。子クラスが CLAUDE.md どおり属性アクセス
      できるよう `_build_part()` が SimpleNamespace に包み直す。**入れ子の spec を読む
      この 1 箇所だけが dict アクセス**で、それ以外は従来どおり属性アクセス。
    """

    _OPS = ("union", "intersection", "difference")

    def __init__(self, config, num_neurons: int = 0, layout=None):
        super().__init__(config, num_neurons, layout)
        self.op = str(_get(config, "op", "union"))
        if self.op not in self._OPS:
            raise ValueError(f"composite area の op は {self._OPS} のいずれかです (got {self.op!r})")

        specs = _get(config, "parts")
        if not specs:
            raise ValueError("composite area には parts が 1 つ以上必要です")
        if self.op == "difference" and len(specs) < 2:
            raise ValueError("composite area の op='difference' には parts が 2 つ以上必要です")

        self.parts: List[BaseArea] = [_build_part(s, num_neurons, layout) for s in specs]
        self.part_names: List[str] = [
            str(s.get("name", f"M{i}")) if isinstance(s, dict) else f"M{i}"
            for i, s in enumerate(specs)
        ]

    def _part_sdfs(self, points: np.ndarray) -> np.ndarray:
        """(k, m) の SDF 行列。行が part、列が点。"""
        return np.stack([p.sdf(points) for p in self.parts], axis=0)

    def sdf(self, points: np.ndarray) -> np.ndarray:
        d = self._part_sdfs(np.asarray(points, dtype=np.float64))
        if self.op == "union":
            return np.min(d, axis=0)
        if self.op == "intersection":
            return np.max(d, axis=0)
        # difference: 先頭から残りを削る
        return np.max(np.concatenate([d[:1], -d[1:]], axis=0), axis=0)

    def part_of(self, points: np.ndarray) -> np.ndarray:
        """各点がどの part に属するかの index。(m, 2) -> (m,) int。

        SDF が最小の part (= 最も深く内部にある part) を選ぶ。part が重なっている領域では
        より内側の方が採用される。`AreaUniformSpace` がこれを `module` 軸に変換する。
        """
        return np.argmin(self._part_sdfs(np.asarray(points, dtype=np.float64)), axis=0)

    # 面積の数値見積もりに使う 1 辺あたりの格子点数。512^2 = 26 万点で誤差 ~0.1%。
    _AREA_GRID = 512

    @property
    def area_um2(self) -> Optional[float]:
        """境界箱に等間隔の格子を敷いて面積を数値的に見積もる。

        合成領域の面積は解析的に出せない (part が重なるため) が、密度の妥当性チェックには
        1% 程度の精度で足りる。**乱数を使わず決定論的な格子で数える**のは、BaseArea が
        RandomState に触らないという不変条件を守るため。
        """
        lo, hi = self.bounds
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            return None
        g = self._AREA_GRID
        xs = np.linspace(lo[0], hi[0], g)
        ys = np.linspace(lo[1], hi[1], g)
        grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
        box = float(np.prod(hi - lo))
        return box * float(self.contains(grid).mean())

    @property
    def bounds(self) -> np.ndarray:
        b = np.stack([p.bounds for p in self.parts], axis=0)  # (k, 2, 2)
        if self.op == "union":
            return np.array([b[:, 0].min(axis=0), b[:, 1].max(axis=0)])
        if self.op == "intersection":
            return np.array([b[:, 0].max(axis=0), b[:, 1].min(axis=0)])
        # difference は先頭を削るだけなので、境界箱は先頭のもので足りる
        return self.parts[0].bounds


@AREA_MODELS.register("modular_4")
class Modular4Area(CompositeArea):
    """4 モジュール (円) を十字のブリッジで連結したモジュール構造。形は areas.yaml が持つ。

    **名前付きの複合領域を増やすときの型がこれ。** `ConfigManager.resolve()` は
    `network.area` の値を areas.yaml のプロファイルキーとしてもレジストリのキーとしても
    使うので、両方が一致していなければならない。したがって新しい複雑形状を足すときは

      1. `configs/components/areas.yaml` に `op` / `parts` を持つプロファイルを書く
      2. ここに `CompositeArea` の空サブクラスを同名で登録する

    の 2 手になる (connectors.py の `beggs_plenz` が `GaussianDistanceTypeTopology` の空
    サブクラスになっているのと同じ作法)。パラメータは YAML 側にだけ置く。
    """
    pass


@AREA_MODELS.register("modular_4_3mm")
class Modular4Area3mm(CompositeArea):
    """modular_4 を 3 mm 角スケールへ拡大したもの (面積 8.997 mm^2)。形は areas.yaml が持つ。"""
    pass


def _get(config: Any, name: str, default: Any = "__required__") -> Any:
    """config から値を取り出す。属性アクセスを基本にしつつ、入れ子 spec の dict も許す。"""
    if isinstance(config, dict):
        value = config.get(name, default)
    else:
        value = getattr(config, name, default)
    if isinstance(value, str) and value == "__required__":
        raise ValueError(f"area の設定に必須項目 {name!r} がありません")
    return value


def _build_part(spec: Dict[str, Any], num_neurons: int, layout) -> BaseArea:
    """`{type: <キー>, ...}` の入れ子 spec から子エリアを作る再帰ファクトリ。"""
    if isinstance(spec, BaseArea):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(f"composite area の parts の要素は mapping である必要があります (got {type(spec)})")
    if "type" not in spec:
        raise ValueError(f"composite area の part に type がありません: {spec}")
    cls = AREA_MODELS.get(spec["type"])
    # 子クラスは self.config.<attr> で読むので、dict を属性アクセスできる形に包み直す。
    child_cfg = SimpleNamespace(**{k: v for k, v in spec.items() if k != "type"})
    return cls(child_cfg, num_neurons, layout=layout)
