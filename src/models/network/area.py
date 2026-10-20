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

    **領域は 2 つある。** 軸索が動ける領域 (= このオブジェクト自身) と、soma を置いてよい
    部分領域 (`soma_area`) で、後者は前者の部分集合。既定では同一だが、`modular_grid` の
    ブリッジのように「軸索は通れるが細胞体は置かない通路」を作るときに分かれる。
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

    # 内外判定の許容差 [um]。**「境界上は内部」を実際に成り立たせるために必要**。
    # 壁の上を進む軸索の到達点は `p + s * u` のように計算で作られるので、正規化した u が
    # 厳密な単位長でないこと・補間係数が 2 進で表せないこと・45 度の面 (DiamondArea) の
    # SDF が両側に丸まることが重なり、数学的には sdf == 0 のはずの点が float64 の 1 ULP
    # (座標 ~1e3 um に対し ~4e-14 um) だけ正側へ転ぶ。許容差 0 だと `segment_inside()` は
    # 16 サンプルの all() なので 1 点でも転べば線分全体が「外」になり、`first_exit()` が
    # 0 を返して軸索が壁に貼り付き、そこで伸長が止まる。
    # 1e-9 um は丸め誤差の ~2.5e4 倍、ジオメトリのスケール (1e2〜1e4 um) の 1e-11 以下で、
    # 部分領域の空隙 (modular_4 で 40 um) の判定には何の影響も与えない。
    _CONTAINS_TOL = 1e-9

    def contains(self, points: np.ndarray) -> np.ndarray:
        """(m, 2) -> (m,) bool。境界上 (sdf == 0) は内部として扱う。

        比較が `<= 0.0` ではなく `<= _CONTAINS_TOL` なのは丸めのため (定数のコメント参照)。
        """
        return (np.asarray(self.sdf(np.asarray(points, dtype=np.float64)))
                <= self._CONTAINS_TOL)

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

    # 線分の内外判定を何点でサンプルするか。軸索セグメント (100 um) なら分解能 ~6.7 um。
    # これより細い空隙・くびれは検出できず飛び越えられてしまうので、そういう形状を扱う
    # ときは派生クラスで上げること。
    _SEGMENT_SAMPLES = 16

    def segment_inside(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """線分 a→b が**全体として**領域内にあるか。(m, 2), (m, 2) -> (m,) bool。

        `contains()` は点しか見ないので、凹形状や非連結な領域では両端点が内部でも途中が
        外に出る — つまり空隙を飛び越えられる。軸索の 1 ステップ (100 um) は
        `modular_4` の円とブリッジの隙間 (40 um) より長いので、端点だけの判定では
        孤立した部分領域の間を軸索が渡ってしまう。それを塞ぐための判定。

        汎用実装は線分上の等間隔サンプル。**凸領域では端点だけで厳密に決まる**ので、
        `DiskArea` / `RectArea` はこれを O(1) に上書きしている。
        """
        a = np.atleast_2d(np.asarray(a, dtype=np.float64))
        b = np.atleast_2d(np.asarray(b, dtype=np.float64))
        if len(a) == 0:
            return np.zeros(0, dtype=bool)
        s = np.linspace(0.0, 1.0, self._SEGMENT_SAMPLES)[None, :, None]
        pts = a[:, None, :] * (1.0 - s) + b[:, None, :] * s
        return self.contains(pts.reshape(-1, 2)).reshape(len(a), -1).all(axis=1)

    # 交点を詰める二分探索の反復回数。線分長の 1/2^K まで詰まるので、100 um セグメント・
    # K=12 なら 0.024 um。判定 1 回が `segment_inside` 1 回 (= 最大 _SEGMENT_SAMPLES 点) なので、
    # 上げると壁に当たった軸索 1 本あたりのコストが線形に増える。
    _BISECT_STEPS = 12

    def first_exit(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """線分 a→b のうち、**領域内に収まる最大の割合** t ∈ [0, 1] を返す。(m,)。

        領域を出ないなら 1.0。`a + t·(b−a)` までの部分線分は `segment_inside()` が True に
        なることが保証される — 判定は同じ `segment_inside()` を二分探索するので、
        「伸ばしてよいと判断した長さ」と「後から検証したときの内外判定」が食い違わない。
        点サンプルで交点を求めると、部分線分の方がサンプル間隔が細かいぶん厳しく、
        通したはずのものが後から外と判定されうる。

        `segment_inside()` が「出るか否か」だけを答えるのに対し、こちらは**どこまで進めるか**を
        答える。軸索が壁に沿って滑るにはこれが要る — 壁まで進み、そこでの法線で向きを変え、
        残りの長さを壁沿いに使う、という順序で初めて「壁に当たってから曲がる」になる
        (候補終点で法線を取ると、曲がる場所も向きもずれる)。

        部分線分が内部なら、その部分線分もまた内部 (前半は前半) なので判定は t について
        単調で、二分探索が使える。
        """
        a = np.atleast_2d(np.asarray(a, dtype=np.float64))
        b = np.atleast_2d(np.asarray(b, dtype=np.float64))
        m = len(a)
        if m == 0:
            return np.zeros(0, dtype=np.float64)

        d = b - a
        lo = np.zeros(m, dtype=np.float64)
        hi = np.ones(m, dtype=np.float64)
        hit = ~self.segment_inside(a, b)
        if not hit.any():
            return hi
        hi[~hit] = 0.0                      # 収まっている行は lo=hi=... で動かさない
        for _ in range(self._BISECT_STEPS):
            mid = 0.5 * (lo + hi)
            ok = self.segment_inside(a, a + mid[:, None] * d)
            lo = np.where(ok, mid, lo)
            hi = np.where(ok, hi, mid)
        return np.where(hit, lo, 1.0)

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

    @property
    def soma_area(self) -> "BaseArea":
        """soma を配置してよい部分領域。既定は**自分自身**。

        軸索の閉じ込め (`sdf` / `normal` / `segment_inside` / `first_exit`) は常にこの
        オブジェクト自身が担い、ここを参照するのは `AreaUniformSpace` だけ。両者を分けるのは
        「軸索は通れるが細胞体は置かない」通路 — `modular_grid` のブリッジ — を表すため。

        **既定が `self` (同一オブジェクト) なのは後方互換の要。** `sample()` の呼び先も
        乱数の消費数もこれまでと 1 draw も変わらないので、既存 config の実現は不変。
        """
        return self


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

    def segment_inside(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """平面全体が領域なので常に True。サンプリングを回す意味がない。"""
        return np.ones(len(np.atleast_2d(np.asarray(a, dtype=np.float64))), dtype=bool)

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

    def segment_inside(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """円板は凸なので、両端点が内部なら線分も内部。サンプリング不要で厳密。"""
        return self.contains(a) & self.contains(b)

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

    def segment_inside(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """矩形は凸なので、両端点が内部なら線分も内部。サンプリング不要で厳密。"""
        return self.contains(a) & self.contains(b)

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


@AREA_MODELS.register("diamond")
class DiamondArea(BaseArea):
    """菱形 (45 度回転した矩形 = L1 ボール)。config: `radius`、`center` (省略時は原点)。

    `radius` は**中心から頂点まで** (半対角線) の距離 [um]。スカラーなら等方、
    `[rx, ry]` なら異方。

    単体で使うより、**矩形との intersection で 45 度の面取りを作る**のが主用途:
    一辺 `side` の正方形と半対角線 `side - c` の菱形の積は、角を長さ `c` だけ 45 度に
    落とした八角形になる (`ModularGridArea` の `chamfer` がこれ)。矩形を回転させる仕組みを
    入れる代わりに菱形を 1 つ足すだけで済むのは、SDF が合成できるおかげ。

    SDF は Inigo Quilez の sdRhombus と同じ厳密式なので、面取り面でも法線が正しく取れる
    (軸索の壁沿い偏向がそのまま効く)。
    """

    def __init__(self, config, num_neurons: int = 0, layout=None):
        super().__init__(config, num_neurons, layout)
        r = np.atleast_1d(np.asarray(_get(config, "radius"), dtype=np.float64))
        self.half_diag = np.array([r[0], r[-1]], dtype=np.float64)   # スカラーなら等方
        self.center = np.asarray(_get(config, "center", (0.0, 0.0)), dtype=np.float64)
        if np.any(self.half_diag <= 0):
            raise ValueError(f"diamond area の radius は正である必要があります (got {r})")

    def sdf(self, points: np.ndarray) -> np.ndarray:
        p = np.abs(np.atleast_2d(np.asarray(points, dtype=np.float64)) - self.center)
        b = self.half_diag
        # 最寄りの辺までの距離。h は辺上の最近接点のパラメータ [-1, 1]。
        ndot = (b[0] - 2.0 * p[:, 0]) * b[0] - (b[1] - 2.0 * p[:, 1]) * b[1]
        h = np.clip(ndot / float(b @ b), -1.0, 1.0)
        q = p - 0.5 * b * np.stack([1.0 - h, 1.0 + h], axis=1)
        d = np.linalg.norm(q, axis=1)
        return d * np.sign(p[:, 0] * b[1] + p[:, 1] * b[0] - b[0] * b[1])

    def segment_inside(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """菱形は凸なので、両端点が内部なら線分も内部。"""
        return self.contains(a) & self.contains(b)

    @property
    def bounds(self) -> np.ndarray:
        return np.array([self.center - self.half_diag, self.center + self.half_diag])

    @property
    def area_um2(self) -> float:
        return float(2.0 * self.half_diag[0] * self.half_diag[1])


@AREA_MODELS.register("composite")
class CompositeArea(BaseArea):
    """複数の領域を SDF の集合演算で合成する。**複雑な領域はこれ 1 つで作る。**

    config:
        op:    "union" | "intersection" | "difference"
        parts: [{type: <AREA_MODELS のキー>, ...そのクラスのパラメータ}, ...]

    part には形状パラメータのほかに 2 つのメタキーを書ける:
        name:       part 名 (省略時 `M{i}`)。`AreaUniformSpace` の `module` 軸のラベルになる
        allow_soma: false にすると**その part には soma を置かない** (既定 true)。
                    軸索の閉じ込めには影響しないので、その part は「軸索だけが通れる通路」
                    になる。`soma_area` を参照。`op: union` のときだけ意味を持つ。

    合成規則は SDF の標準的なもの:
        union        = min(d_0, d_1, ...)          … 和 (くっつける)
        intersection = max(d_0, d_1, ...)          … 積 (重なりだけ残す)
        difference   = max(d_0, -d_1, -d_2, ...)   … 差 (先頭から残りを削る = 穴あけ)

    `parts` は入れ子にできる (part 自体を `type: composite` にする) ので、いくらでも
    複雑な形にできる。例: 4 つの円 + 十字のブリッジ = モジュール構造。

    合成結果は凹にも非連結にもなりうる (parts が重なっていなければ部分領域は孤立する) ので、
    `segment_inside()` は凸形状のような端点だけの近道が使えず、基底のサンプリング実装を使う。

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
        # soma を置いてよい part か (既定 True)。False の part は軸索だけが通れる通路になる。
        self.part_allows_soma: List[bool] = [
            bool(s.get("allow_soma", True)) if isinstance(s, dict) else True
            for s in specs
        ]
        self._soma_area: Optional[BaseArea] = None

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

    def _soma_area_um2(self) -> Optional[float]:
        """soma 配置領域の面積 [um^2] を解析的に出せるなら返す。既定は None。

        None なら `_SomaSubArea` が `CompositeArea` の格子カウント (~0.1% 誤差) に任せる。
        パラメータから面積が分かるサブクラス (`ModularGridArea`) だけが上書きする。
        """
        return None

    @property
    def soma_area(self) -> BaseArea:
        """`allow_soma: false` の part を除いた union。全 part が許可なら `self`。

        **全許可のとき同じオブジェクトを返すのが後方互換の要**で、`AreaUniformSpace` は
        これまでどおり領域そのものからサンプルする (乱数消費が 1 draw も変わらない)。

        `part_names` は親のものをそのまま引き継ぐので、`module` 軸のラベルは
        除外前と同じ名前 (`M0`, `M1`, ...) になる。
        """
        if all(self.part_allows_soma):
            return self
        if self._soma_area is not None:
            return self._soma_area
        if self.op != "union":
            raise ValueError(
                f"composite area の allow_soma: false は op='union' でのみ使えます"
                f" (got op={self.op!r})。intersection / difference では「その part を抜いた"
                " 領域」が定義できません。"
            )
        keep = [i for i, ok in enumerate(self.part_allows_soma) if ok]
        if not keep:
            raise ValueError(
                "composite area の全 part が allow_soma: false です。soma を置ける場所が"
                " ありません — 少なくとも 1 つの part を許可してください。"
            )
        self._soma_area = _SomaSubArea(
            [self.parts[i] for i in keep],
            [self.part_names[i] for i in keep],
            num_neurons=self.num_neurons,
            layout=self.layout,
            area_um2=self._soma_area_um2(),
        )
        return self._soma_area

    @property
    def bounds(self) -> np.ndarray:
        b = np.stack([p.bounds for p in self.parts], axis=0)  # (k, 2, 2)
        if self.op == "union":
            return np.array([b[:, 0].min(axis=0), b[:, 1].max(axis=0)])
        if self.op == "intersection":
            return np.array([b[:, 0].max(axis=0), b[:, 1].min(axis=0)])
        # difference は先頭を削るだけなので、境界箱は先頭のもので足りる
        return self.parts[0].bounds


class _SomaSubArea(CompositeArea):
    """親の part の一部だけを集めた union。**soma 配置領域を表すために内部で使う。**

    `AREA_MODELS` に登録しないのは、これが config から名指しできるプロファイルではなく
    `CompositeArea.soma_area` の実装内部だから。part は親が構築済みのものを共有する
    (`_build_part()` が `BaseArea` を素通しする) ので作り直さない。

    `part_names` に親の名前をそのまま持たせてあるので、`AreaUniformSpace` はこの領域の上で
    `part_of()` / `part_names` を読むだけで正しい `module` 軸が得られる。**全体領域で
    `part_of()` を採ってはいけない** — ブリッジがモジュールへ食い込む帯では、モジュール内の
    点でも「より深い」ブリッジ part が選ばれ、soma が `B0-1` とラベルされてしまう。
    """

    def __init__(self, parts, names, num_neurons: int = 0, layout=None,
                 area_um2: Optional[float] = None):
        super().__init__(SimpleNamespace(op="union", parts=list(parts)), num_neurons, layout)
        self.part_names = list(names)
        self._exact_area = area_um2

    @property
    def area_um2(self) -> Optional[float]:
        """解析値が渡されていればそれを、無ければ `CompositeArea` の格子カウントを返す。"""
        if self._exact_area is not None:
            return self._exact_area
        return CompositeArea.area_um2.fget(self)


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


@AREA_MODELS.register("modular_4_grid")
class Modular4GridArea(CompositeArea):
    """四隅の正方形を長方形のブリッジで格子状に連結したモジュール構造。形は areas.yaml が持つ。

    `modular_4` との違いは **parts が実際に重なっていること**で、全体が 1 つの連結成分に
    なる。したがってモジュール間の結合はブリッジを通った軸索だけが作り、`modular_4` の
    ように完全に独立したサブネットワークにはならない。面積は `modular_4` に揃えてあるので
    (2.230 mm^2 vs 2.227 mm^2)、同じ N で「孤立」と「格子連結」を比較できる。
    """
    pass


@AREA_MODELS.register("modular_grid")
class ModularGridArea(CompositeArea):
    """モジュール構造を**パラメータから生成する**格子エリア。

    `modular_4_grid` が形を areas.yaml に直書きするのに対し、こちらは数値だけを受け取って
    モジュールとブリッジの配置を自分で組み立てる。生成した parts は結局 `union` の
    `CompositeArea` なので、SDF 合成・`part_of`・`segment_inside`・`module` 軸といった
    仕組みは何ひとつ変わらない。**任意形状を parts で直接書く道は残したまま**、
    「格子状のモジュール構造」という頻出パターンだけを一撃で書けるようにするもの。

    config:
        side          … 正方形モジュールの一辺 [um]
        bridge_width  … ブリッジの線幅 [um]
        spacing       … 隣接モジュールの**中心間距離** [um] (辺と辺の隙間は spacing - side)
        num_modules   … モジュール数

        chamfer        … 省略可。正方形の角を 45 度に落とす長さ [um] (辺に沿って測った量)。
                         既定は side/4。0 なら面取りなしの正方形
        bridge_overlap … 省略可。ブリッジをモジュールへ食い込ませる深さ [um]。既定は
                         min(bridge_width, side/2)。**0 にしてはいけない** — parts が
                         接するだけだと union が連結にならず、`modular_4` と同じ
                         「隣なのに繋がらない」状態になる
        rows, cols     … 省略可。格子の行数・列数を明示する
        soma_in_bridge … 省略可。ブリッジに soma を置くか (既定 True = 従来どおり)。
                         False にするとブリッジ part に `allow_soma: false` が立ち、
                         **soma はモジュール内だけ・軸索はブリッジを通る**という構成になる。
                         `module` 軸から `B*` ラベルが消えるのも狙いのひとつ

    **格子の形は num_modules から自動で決まる**: rows * cols == num_modules を満たす組の
    うち最も正方形に近いもの (rows <= cols)。穴のない完全な矩形格子になるので、
    4 -> 2x2、6 -> 2x3、9 -> 3x3、5 -> 1x5 (素数は一列)。意図と違うときだけ
    `rows` / `cols` を書く。

    ブリッジは**上下左右の隣接モジュール間にだけ**張られる (斜めは張らない) ので、
    対角のモジュールへは必ず 2 ホップかかる。

    **面取りは「正方形 ∩ 菱形」で作る。** 一辺 side の正方形と半対角線 (side - chamfer) の
    `DiamondArea` の積が、角を chamfer だけ 45 度に落とした八角形になる。矩形を回転させる
    仕組みを持ち込まずに済むのは SDF が合成できるからで、モジュール 1 つが入れ子の
    `composite` になるだけなので `part_of` から見た part の数は変わらない。

    part の並びは「モジュールを行優先 (下の行から) で M0..M{n-1}、続いてブリッジ」。
    part_names もそれに合わせて `M0`, `M1`, ... / `B0-1`, `B0-2`, ... になるので、
    `AreaUniformSpace` の `module` 軸にそのまま出る。
    """

    def __init__(self, config, num_neurons: int = 0, layout=None):
        side = float(_get(config, "side"))
        bridge_width = float(_get(config, "bridge_width"))
        spacing = float(_get(config, "spacing"))
        n = int(_get(config, "num_modules"))

        if n < 1:
            raise ValueError(f"modular_grid の num_modules は 1 以上である必要があります (got {n})")

        chamfer = float(_get(config, "chamfer", side / 4.0))
        # parts が「接する」だけでは union が連結にならないので、必ず食い込ませる。
        overlap = float(_get(config, "bridge_overlap", min(bridge_width, side / 2.0)))
        _validate_grid_params(side, bridge_width, spacing, chamfer, overlap, kind="modular_grid")
        overlap = min(overlap, side / 2.0)

        rows = _get(config, "rows", None)
        cols = _get(config, "cols", None)
        if rows is None or cols is None:
            rows, cols = self._grid_shape(n)
        rows, cols = int(rows), int(cols)
        if rows * cols < n:
            raise ValueError(
                f"modular_grid: rows*cols ({rows}x{cols}) が num_modules ({n}) より少ないです。"
            )

        # ブリッジは「軸索は通れるが細胞体は置かない通路」にもできる。閉じ込めには効かない
        # ので、軸索側の挙動 (ブリッジ経由のモジュール間結合) はどちらでも変わらない。
        soma_in_bridge = bool(_get(config, "soma_in_bridge", True))

        gap = spacing - side          # 辺と辺の隙間 = ブリッジの露出長

        def center(i, j):
            return (j - (cols - 1) / 2.0) * spacing, (i - (rows - 1) / 2.0) * spacing

        # 行優先 (下の行から) にモジュール番号を振る。cell[(i, j)] -> モジュール index
        cell = {}
        specs: List[Dict[str, Any]] = []
        for idx in range(n):
            i, j = divmod(idx, cols)
            cell[(i, j)] = idx
            cx, cy = center(i, j)
            specs.append(_module_spec(f"M{idx}", cx, cy, side, chamfer))

        # 上下左右の隣にだけブリッジを張る (斜めは張らない)
        for (i, j), idx in sorted(cell.items()):
            for di, dj in ((0, 1), (1, 0)):
                nb = cell.get((i + di, j + dj))
                if nb is None:
                    continue
                specs.append(_bridge_spec(
                    f"B{idx}-{nb}", center(i, j), center(i + di, j + dj),
                    side=side, bridge_width=bridge_width, overlap=overlap,
                    allow_soma=soma_in_bridge,
                ))

        super().__init__(SimpleNamespace(op="union", parts=specs), num_neurons, layout)
        # 合成用に組み立てた config ではなく、ユーザーが書いた値を残す
        self.config = config
        self.side, self.bridge_width, self.spacing = side, bridge_width, spacing
        self.num_modules, self.rows, self.cols = n, rows, cols
        self.chamfer, self.bridge_overlap, self.gap = chamfer, overlap, gap
        self.n_bridges = len(specs) - n
        self.soma_in_bridge = soma_in_bridge

    @staticmethod
    def _grid_shape(n: int) -> tuple:
        """rows * cols == n を満たす組のうち最も正方形に近いもの (rows <= cols)。

        穴のない完全な矩形格子だけを作るので、モジュール数が素数なら一列になる。
        """
        best = (1, n)
        for r in range(1, int(np.sqrt(n)) + 1):
            if n % r == 0:
                best = (r, n // r)
        return best

    @property
    def area_um2(self) -> float:
        """解析的な面積。ブリッジの食い込み分はモジュールの内側なので二重計上しない。

        モジュール 1 つは side^2 から 45 度の角 4 つ (直角二等辺三角形、脚 chamfer) を
        落としたもの。`CompositeArea` の格子カウント (~0.1% 誤差) を使わずに済むので、
        `AreaUniformSpace` が出す密度もパラメータから決まる厳密値になる。
        """
        module = self.side ** 2 - 2.0 * self.chamfer ** 2
        return float(self.num_modules * module + self.n_bridges * self.gap * self.bridge_width)

    def _soma_area_um2(self) -> float:
        """soma 配置領域 (= モジュールだけ) の解析面積。`area_um2` からブリッジの露出分を除く。

        `soma_in_bridge: false` のとき `AreaUniformSpace` が出す密度の分母になる。
        N はモジュール内にしか居ないので、実効密度はこちらで割った値。
        """
        return float(self.num_modules * (self.side ** 2 - 2.0 * self.chamfer ** 2))


@AREA_MODELS.register("hierarchical_modular_grid")
class HierarchicalModularGridArea(CompositeArea):
    """**2 階層**のモジュール格子。`modular_grid` をもう 1 段深くしたもの。

    `modular_grid` の `num_modules: 4` が作る 2x2 クラスタを **4 つ格子状に並べ**、
    各クラスタの**内側 1 モジュール** (全体の中心に最も近いもの) 同士を同種のブリッジで
    格子状に繋ぐ。結果は 16 モジュール / 20 ブリッジ / 連結成分 1 つ::

        □ = モジュール         ── = クラスタ内ブリッジ (16 本)
        ■ = 内側 4 モジュール   ══ = クラスタ間ブリッジ ( 4 本)

           □──□     □──□
           │  │     │  │
           □──■═════■──□        4x4 格子の全直交辺 24 本から、
              ║     ║           境界の外側 4 辺を抜いた形にあたる
           □──■═════■──□
           │  │     │  │
           □──□     □──□

    狙いは**階層的なモジュール性**で、クラスタ内は密・クラスタ間は内側モジュールを
    経由する 1 本の細い経路だけになる。`modular_grid` を 16 モジュールで使うと格子上の
    どのモジュールも対等に繋がってしまい、この 2 段構造は作れない。

    config (`modular_grid` と同じものに `block_spacing` が加わる):
        side          … 正方形モジュールの一辺 [um]
        bridge_width  … ブリッジの線幅 [um]
        spacing       … **クラスタ内**モジュールの中心間距離 [um]

        block_spacing  … 省略可。**クラスタ中心間**の距離 [um]。既定 2*spacing
        chamfer        … 省略可。角を 45 度に落とす長さ [um]。既定 side/4
        bridge_overlap … 省略可。ブリッジをモジュールへ食い込ませる深さ [um]
        soma_in_bridge … 省略可。ブリッジに soma を置くか (既定 True)

    **4x4 固定** — `num_modules` / `rows` / `cols` は受け付けない。「内側のモジュール」が
    素直に定まるのは 2x2 のクラスタを 2x2 に並べたときだけなので、格子形を可変にすると
    どれを繋ぐかが恣意的になる。任意個の格子が要るなら `modular_grid` を使うこと。

    **`block_spacing` の既定 `2*spacing` は「均一な 4x4 格子」**になる (クラスタ間の隙間が
    クラスタ内と同じ)。階層をはっきりさせたいならこれより大きくする — 内側モジュールの
    中心間距離が `block_spacing - spacing` なので、クラスタ間ブリッジだけが長く伸びる。

    part の並びは モジュール 16 → クラスタ内ブリッジ 16 → クラスタ間ブリッジ 4。
    名前は `C{c}-M{m}` / `BC{c}-{m1}-{m2}` / `BX{c1}-{c2}` で、`AreaUniformSpace` の
    `module` 軸にそのまま出る。**`C0-M0` 形式は文字列ソートでクラスタごとのブロックに
    なる**ので、`order_by("module")` や `group_connection_probability` が追加の軸なしで
    階層を見せられる。
    """

    # 各クラスタで全体の中心に最も近いモジュール。2x2 のクラスタを 2x2 に並べた形では
    # 「クラスタ自身の中心に対する点対称」= 3 - c になる (c0 は左下なので右上の M3、…)。
    _INNER = {0: 3, 1: 2, 2: 1, 3: 0}

    # クラスタ / モジュールとも 2x2 固定。index 0 が下・左なのは ModularGridArea と同じ。
    _GRID = 2

    def __init__(self, config, num_neurons: int = 0, layout=None):
        for forbidden in ("num_modules", "rows", "cols"):
            if _get(config, forbidden, None) is not None:
                raise ValueError(
                    f"hierarchical_modular_grid は 4 クラスタ x 4 モジュールの固定形なので"
                    f" {forbidden!r} を受け付けません。格子形を変えたいなら modular_grid を"
                    " 使ってください。"
                )

        side = float(_get(config, "side"))
        bridge_width = float(_get(config, "bridge_width"))
        spacing = float(_get(config, "spacing"))
        block_spacing = float(_get(config, "block_spacing", 2.0 * spacing))

        chamfer = float(_get(config, "chamfer", side / 4.0))
        overlap = float(_get(config, "bridge_overlap", min(bridge_width, side / 2.0)))
        _validate_grid_params(side, bridge_width, spacing, chamfer, overlap,
                              kind="hierarchical_modular_grid")
        overlap = min(overlap, side / 2.0)

        # 内側モジュール同士の中心間距離は block_spacing - spacing。これが side を
        # 超えないと、隣り合うクラスタの内側モジュールが重なってしまう。
        if block_spacing - spacing <= side:
            raise ValueError(
                f"hierarchical_modular_grid の block_spacing は spacing + side より大きい"
                f" 必要があります (got block_spacing={block_spacing}, spacing={spacing},"
                f" side={side})。隣接クラスタの内側モジュールの中心間距離は"
                f" block_spacing - spacing = {block_spacing - spacing} で、これが side 以下だと"
                " モジュールが重なります。"
            )

        soma_in_bridge = bool(_get(config, "soma_in_bridge", True))

        g = self._GRID
        # 格子 index (0 が下・左) -> 中心座標。クラスタもモジュールも同じ式で、
        # 刻みが block_spacing か spacing かだけが違う。
        def offset(i, j, pitch):
            return (j - (g - 1) / 2.0) * pitch, (i - (g - 1) / 2.0) * pitch

        def module_center(c, m):
            bx, by = offset(*divmod(c, g), block_spacing)
            mx, my = offset(*divmod(m, g), spacing)
            return bx + mx, by + my

        # 2x2 格子の直交隣接対を (a, b) の昇順で返す。ModularGridArea と同じ走査順
        # ((0,1) = 右, (1,0) = 上) なので、a は必ず b の左か下になる。
        def neighbours():
            cell = {divmod(k, g): k for k in range(g * g)}
            for (i, j), k in sorted(cell.items()):
                for di, dj in ((0, 1), (1, 0)):
                    nb = cell.get((i + di, j + dj))
                    if nb is not None:
                        yield k, nb

        specs: List[Dict[str, Any]] = []
        # 1. モジュール 16 枚 (クラスタが外側、モジュールが内側)
        for c in range(g * g):
            for m in range(g * g):
                cx, cy = module_center(c, m)
                specs.append(_module_spec(f"C{c}-M{m}", cx, cy, side, chamfer))

        # 2. クラスタ内ブリッジ 16 本 (各クラスタで 4-サイクル)
        for c in range(g * g):
            for m, nb in neighbours():
                specs.append(_bridge_spec(
                    f"BC{c}-{m}-{nb}", module_center(c, m), module_center(c, nb),
                    side=side, bridge_width=bridge_width, overlap=overlap,
                    allow_soma=soma_in_bridge,
                ))

        # 3. クラスタ間ブリッジ 4 本 — 内側モジュール同士だけを繋ぐ。クラスタ対 1 つに
        #    つき 1 本なので、名前は対だけで一意になる。
        for c, nb in neighbours():
            specs.append(_bridge_spec(
                f"BX{c}-{nb}",
                module_center(c, self._INNER[c]), module_center(nb, self._INNER[nb]),
                side=side, bridge_width=bridge_width, overlap=overlap,
                allow_soma=soma_in_bridge,
            ))

        super().__init__(SimpleNamespace(op="union", parts=specs), num_neurons, layout)
        # 合成用に組み立てた config ではなく、ユーザーが書いた値を残す
        self.config = config
        self.side, self.bridge_width, self.spacing = side, bridge_width, spacing
        self.block_spacing = block_spacing
        self.chamfer, self.bridge_overlap = chamfer, overlap
        self.soma_in_bridge = soma_in_bridge
        self.num_clusters = self.modules_per_cluster = g * g
        self.num_modules = self.num_clusters * self.modules_per_cluster
        self.n_intra_bridges = len(specs) - self.num_modules - self.num_clusters
        self.n_inter_bridges = self.num_clusters
        self.n_bridges = self.n_intra_bridges + self.n_inter_bridges
        # 辺と辺の隙間 = ブリッジの露出長。クラスタ内と間で違う。
        self.gap = spacing - side
        self.inter_gap = block_spacing - spacing - side

    def module_index(self, cluster: int, module: int) -> int:
        """`(クラスタ, モジュール)` -> `parts` / `part_names` の index。"""
        return cluster * self.modules_per_cluster + module

    @property
    def area_um2(self) -> float:
        """解析的な面積。ブリッジの食い込み分はモジュールの内側なので二重計上しない。

        `ModularGridArea.area_um2` との違いは、露出長の違うブリッジが 2 種類あること
        だけ (クラスタ内は `gap`、クラスタ間は `inter_gap`)。
        """
        module = self.side ** 2 - 2.0 * self.chamfer ** 2
        return float(self.num_modules * module
                     + self.n_intra_bridges * self.gap * self.bridge_width
                     + self.n_inter_bridges * self.inter_gap * self.bridge_width)

    def _soma_area_um2(self) -> float:
        """soma 配置領域 (= モジュールだけ) の解析面積。`soma_in_bridge: false` の分母。"""
        return float(self.num_modules * (self.side ** 2 - 2.0 * self.chamfer ** 2))

@AREA_MODELS.register("hierarchical_modular_grid2")
class HierarchicalModularGridArea2(HierarchicalModularGridArea):
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


# ======================================================================================
# 格子状モジュール構造の部品 (ModularGridArea / HierarchicalModularGridArea が共有)
#
# 「正方形モジュール + それを繋ぐ矩形ブリッジ」という形は 1 段でも 2 段でも同じなので、
# spec を組み立てる式をここに 1 つだけ置く。**式を書き換えると同じ seed の既存
# ネットワークが別物になる** (座標が変われば棄却サンプリングのドロー数も変わり、
# 重みと遅延まで全部ずれる — CLAUDE.md の注意 14) ので、幾何を触るときは意図的に。
# ======================================================================================

def _validate_grid_params(side: float, bridge_width: float, spacing: float,
                          chamfer: float, overlap: float, *, kind: str) -> None:
    """格子モジュール構造に共通のパラメータ検証。`kind` はメッセージに出すプロファイル名。"""
    if side <= 0:
        raise ValueError(f"{kind} の side は正である必要があります (got {side})")
    if bridge_width <= 0:
        raise ValueError(f"{kind} の bridge_width は正である必要があります (got {bridge_width})")
    if spacing <= side:
        raise ValueError(
            f"{kind} の spacing は**中心間距離**なので side より大きい必要があります"
            f" (got spacing={spacing}, side={side})。辺と辺の隙間は spacing - side です。"
        )
    if not 0.0 <= chamfer < side / 2.0:
        raise ValueError(
            f"{kind} の chamfer は [0, side/2) である必要があります"
            f" (got {chamfer}, side={side})。side/2 では正方形が菱形に潰れます。"
        )
    if bridge_width > side - 2.0 * chamfer:
        raise ValueError(
            f"{kind}: bridge_width ({bridge_width}) が面取り後の辺の直線部"
            f" ({side - 2.0 * chamfer}) を超えています。ブリッジが面取り面にはみ出すので、"
            " bridge_width を細くするか chamfer を小さくしてください。"
        )
    if overlap <= 0:
        raise ValueError(
            f"{kind} の bridge_overlap は正である必要があります"
            " (0 だとブリッジと正方形が接するだけで連結にならない)。"
        )


def _module_spec(name: str, cx: float, cy: float, side: float, chamfer: float) -> Dict[str, Any]:
    """モジュール 1 つの spec。面取りありなら「正方形 ∩ 菱形」、なしなら素の正方形。"""
    half = side / 2.0
    square = {"type": "rect",
              "x_range": [cx - half, cx + half],
              "y_range": [cy - half, cy + half]}
    if chamfer <= 0.0:
        return {"name": name, **square}
    # 半対角線 side - chamfer の菱形で角を落とす (辺に沿って chamfer だけ削れる)
    return {"type": "composite", "name": name, "op": "intersection",
            "parts": [square,
                      {"type": "diamond", "center": [cx, cy],
                       "radius": side - chamfer}]}


def _bridge_spec(name: str, a, b, *, side: float, bridge_width: float,
                 overlap: float, allow_soma: bool) -> Dict[str, Any]:
    """軸平行に並んだ 2 つのモジュール中心 a → b を繋ぐ矩形ブリッジの spec。

    `a` は必ず `b` の左か下 (呼び出し側が隣接を +x / +y 方向にだけ辿るため)。
    両端は `overlap` だけモジュールへ食い込ませる — 接するだけでは union が連結にならない。
    """
    (ax, ay), (bx, by) = a, b
    half = side / 2.0
    meta = {"name": name, "allow_soma": allow_soma}
    if abs(bx - ax) > abs(by - ay):   # 横ブリッジ
        return {"type": "rect", **meta,
                "x_range": [ax + half - overlap, bx - half + overlap],
                "y_range": [ay - bridge_width / 2.0, ay + bridge_width / 2.0]}
    return {"type": "rect", **meta,   # 縦ブリッジ
            "x_range": [ax - bridge_width / 2.0, ax + bridge_width / 2.0],
            "y_range": [ay + half - overlap, by - half + overlap]}
