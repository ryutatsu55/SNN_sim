import itertools
import numpy as np
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Optional, Dict, Any
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform
from src.core.registry import CONNECTION_MODELS

from .area import BaseArea

class BaseConnection(ABC):
    """シナプス結合の有無(マスク)を決定する基底クラス"""

    # 密な (N,N) マスクを作らずに COO を直接生成できるか。True を宣言したクラスは
    # generate_sparse() を実装しなければならない。NetworkBuilder は結合/重み/遅延の
    # 3 段すべてが True のときだけ疎生成経路を選ぶ。
    supports_sparse: bool = False

    def __init__(self, config: Dict[str, Any], num_neurons: int, coords: Optional[np.ndarray], rng: np.random.RandomState, layout=None,
                 area: Optional[BaseArea] = None):
        self.config = config
        self.num_neurons = num_neurons
        self.coords = coords
        self.rng = rng
        # NetworkLayout。ニューロン種ごとの意図的バイアスや無相関化(シャッフル)を
        # 具象クラス側で実装したい場合に self.layout.ids_by("polarity") などを参照する。
        self.layout = layout
        # BaseArea。幾何的に結合を作るモデル(軸索伸長など)が、軸索を領域内に
        # 閉じ込めるために使う。確率ベースのモデルは参照しない。
        self.area = area

    def describe_axes(self) -> Dict[str, Any]:
        """任意フック: このコンポーネントが定義するカテゴリ/ソート軸を宣言する。

        NetworkBuilder が `generate()` / `generate_sparse()` の**直後**に呼び、戻り値を
        `NetworkLayout.add_axis()` へ注入する。生成後に呼ばれるので、自身が計算した
        座標・マスク等から軸を導出してよい(例: コミュニティ検出、次数によるランク付け)。

        Returns:
            {軸名: 長さ num_neurons の配列}。カテゴリ名(文字列)でもソート用の数値でも
            よい。既定は空 dict = 軸を定義しない。
        """
        return {}

    @abstractmethod
    def generate(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: 形状 (num_neurons, num_neurons) の結合マスク (0 or 1の np.int8 配列など)
        """
        pass

    def generate_sparse(self) -> tuple[np.ndarray, np.ndarray]:
        """結合を COO (rows, cols) で返す。行優先ソート済みであること。

        Returns:
            (rows, cols): それぞれ int32 のグローバル pre/post ID
        """
        raise NotImplementedError(
            f"{type(self).__name__} は疎生成に対応していません (supports_sparse=False)。"
        )

@CONNECTION_MODELS.register("constant_prob")
class ConstantProbabilityTopology(BaseConnection):
    def generate(self):
        """空間配置を無視し、純粋な確率で結合マスクを生成"""
        prob = self.config.p
        
        mask = self.rng.random((self.num_neurons, self.num_neurons)) < prob
        if not self.config.allow_self_connections:
            np.fill_diagonal(mask, 0) # 自己結合（自分自身へのシナプス）を排除
        return mask.astype(np.int8)

@CONNECTION_MODELS.register("constant_prob_full")
class FullProbabilityTopology(ConstantProbabilityTopology):
    """論文再現用の全ペア結合候補。具体値はYAMLプロファイルから読む。"""
    pass

@CONNECTION_MODELS.register("constant_prob_full_autapse")
class FullAutapseProbabilityTopology(ConstantProbabilityTopology):
    """自己結合あり検証用の全ペア結合候補。具体値はYAMLプロファイルから読む。"""
    pass

@CONNECTION_MODELS.register("constant_prob_sparse")
class SparseProbabilityTopology(ConstantProbabilityTopology):
    """Akita SoC安定化候補用の疎な確率結合。具体値はYAMLプロファイルから読む。"""
    pass

@CONNECTION_MODELS.register("optional_connect")
class OptionalConnection(BaseConnection):
    def generate(self):
        """任意の結合を指定する"""        
        mask = np.zeros((self.num_neurons, self.num_neurons))
        mask[self.config.src_ID, self.config.tgt_ID] = 1
        return mask.astype(np.int8)
    
@CONNECTION_MODELS.register("C.elegans")
class C_elegansConnection(BaseConnection):
    def generate(self):
        """synapse_mask.csvから接続マスクを読み込む。
        元データにおいては0以外はすべて接続（1）として定義する。
        """
        csv_path = Path(__file__).parent / "data" / "c_elegans" / "synapse_mask.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"synapse_mask.csv not found at {csv_path}")

        # synapse_mask.csvを読み込む
        mask_data = np.loadtxt(csv_path, delimiter=",", dtype=np.int8)

        # 形状チェック
        if mask_data.shape != (self.num_neurons, self.num_neurons):
            raise ValueError(
                f"synapse_mask shape {mask_data.shape} does not match "
                f"expected ({self.num_neurons}, {self.num_neurons})"
            )

        # 0以外をすべて1に変換
        mask = (mask_data != 0).astype(np.int8)

        return mask

@CONNECTION_MODELS.register("distance_based")
class DistanceBasedTopology(BaseConnection):
    def generate(self):
        """空間座標間の距離に基づき、ガウス分布で減衰する確率結合を生成"""
        if self.coords is None:
            raise ValueError("DistanceBasedTopology requires spatial coordinates (coords cannot be None).")

        # N x N の距離行列を一括計算
        dist_matrix = squareform(pdist(self.coords))
        
        # 距離に応じた結合確率を計算
        prob_matrix = self.config.max_prob * np.exp(-(dist_matrix**2) / (2 * self.config.sigma**2))
        
        # 確率に基づいてマスクを生成
        mask = self.rng.random((self.num_neurons, self.num_neurons)) < prob_matrix
        
        return mask.astype(np.int8)

@CONNECTION_MODELS.register("gaussian_distance_type")
class GaussianDistanceTypeTopology(BaseConnection):
    """種別別(E→E, E→I, I→E, I→I)のガウス型距離依存結合。

    Beggs & Plenz (2003) 再現用(docs/refs/SNN_PA~1.MD §3)。結合確率は
        P(結合 | 距離 d) = p0_xy * exp(-d^2 / (2 * sigma_xy^2))
    で、送信種別 x(E/I) × 受信種別 y(E/I) の 4 ブロックごとに独立の
    (sigma_xy, p0_xy) を用いる。興奮性/抑制性のグローバルID集合は
    layout.ids_by("polarity") から取得する。

    config(フラットなスカラーフィールド):
        sigma_ee/p0_ee, sigma_ei/p0_ei, sigma_ie/p0_ie, sigma_ii/p0_ii  … [um] と確率
        allow_self_connections (bool, 既定 False) … False で対角(オートシナプス)を除去

    大規模ネットワーク向けに generate_sparse() を実装しており、密な (N,N) を作らずに
    COO を直接生成する。乱数ストリームの消費順は密版と完全に一致する(下記参照)。
    """

    supports_sparse = True

    # 疎生成時の 1 行ブロックあたりの一時配列サイズの目安 [bytes]。
    _ROW_BLOCK_BYTES = 128 << 20

    def _validate_inputs(self):
        if self.coords is None:
            raise ValueError(
                "GaussianDistanceTypeTopology requires spatial coordinates (coords cannot be None)."
            )
        if self.layout is None:
            raise ValueError(
                "GaussianDistanceTypeTopology requires a NetworkLayout to resolve E/I populations."
            )

    def generate_sparse(self):
        """行ブロックごとに距離・確率・乱数を作って COO を組み立てる。

        乱数ストリームの同一性:
            RandomState.random_sample は指定形状ぶんの double を Mersenne ストリームから
            逐次消費して C 順に reshape する。よって連続した行ブロック [0,nb), [nb,2nb), ...
            に対して (nb, N) を順に引くことは、(N, N) を一括で引くのとビット単位で同一の
            値が同一の (i, j) に対応する。これを壊さないため:
              - prob == 0 のペアでもドローを省略しない(ブロック全体を無条件にドロー)
              - 自己結合の除去は比較の「後」に行う(密版の fill_diagonal と同じく
                対角のドローは消費して捨てる)
              - 空の id 集合に対する continue は確率の埋め込みだけをスキップし、
                ドローはスキップしない
        """
        self._validate_inputs()

        ids = self.layout.ids_by("polarity")
        exc = ids["excitatory"]
        inh = ids["inhibitory"]
        n = self.num_neurons
        coords = np.ascontiguousarray(self.coords, dtype=np.float64)

        is_exc = np.zeros(n, dtype=bool)
        is_exc[exc] = True
        is_inh = np.zeros(n, dtype=bool)
        is_inh[inh] = True
        cols_e = np.nonzero(is_exc)[0]
        cols_i = np.nonzero(is_inh)[0]

        block = max(1, min(n, self._ROW_BLOCK_BYTES // max(1, 8 * n)))
        allow_self = getattr(self.config, "allow_self_connections", False)

        out_rows: list[np.ndarray] = []
        out_cols: list[np.ndarray] = []

        for start in range(0, n, block):
            stop = min(start + block, n)
            dist = cdist(coords[start:stop], coords)
            prob = np.zeros((stop - start, n), dtype=np.float64)

            local_e = np.nonzero(is_exc[start:stop])[0]
            local_i = np.nonzero(is_inh[start:stop])[0]
            for src_local, tgt_global, sigma, p0 in (
                (local_e, cols_e, self.config.sigma_ee, self.config.p0_ee),
                (local_e, cols_i, self.config.sigma_ei, self.config.p0_ei),
                (local_i, cols_e, self.config.sigma_ie, self.config.p0_ie),
                (local_i, cols_i, self.config.sigma_ii, self.config.p0_ii),
            ):
                if src_local.size == 0 or tgt_global.size == 0:
                    continue
                block_idx = np.ix_(src_local, tgt_global)
                d = dist[block_idx]
                prob[block_idx] = p0 * np.exp(-(d ** 2) / (2.0 * sigma ** 2))

            hit = self.rng.random((stop - start, n)) < prob
            if not allow_self:
                # 密版の np.fill_diagonal(mask, 0) と等価。ドローは既に消費済み。
                hit[np.arange(stop - start), np.arange(start, stop)] = False

            rows, cols = np.nonzero(hit)
            if rows.size:
                out_rows.append((rows + start).astype(np.int32))
                out_cols.append(cols.astype(np.int32))

        if not out_rows:
            empty = np.array([], dtype=np.int32)
            return empty, empty
        return np.concatenate(out_rows), np.concatenate(out_cols)

    def generate(self):
        self._validate_inputs()

        ids = self.layout.ids_by("polarity")
        exc = ids["excitatory"]
        inh = ids["inhibitory"]

        # N x N の距離行列を一括計算(DistanceBasedTopology と同手法)
        dist_matrix = squareform(pdist(self.coords))

        prob_matrix = np.zeros((self.num_neurons, self.num_neurons), dtype=np.float64)

        # 送信種別(行) × 受信種別(列)の 4 ブロックにそれぞれの (sigma, p0) を適用
        blocks = (
            (exc, exc, self.config.sigma_ee, self.config.p0_ee),
            (exc, inh, self.config.sigma_ei, self.config.p0_ei),
            (inh, exc, self.config.sigma_ie, self.config.p0_ie),
            (inh, inh, self.config.sigma_ii, self.config.p0_ii),
        )
        for src_ids, tgt_ids, sigma, p0 in blocks:
            if len(src_ids) == 0 or len(tgt_ids) == 0:
                continue
            block_idx = np.ix_(src_ids, tgt_ids)
            d = dist_matrix[block_idx]
            prob_matrix[block_idx] = p0 * np.exp(-(d ** 2) / (2.0 * sigma ** 2))

        mask = self.rng.random((self.num_neurons, self.num_neurons)) < prob_matrix

        if not getattr(self.config, "allow_self_connections", False):
            np.fill_diagonal(mask, 0)  # オートシナプス禁止 (MD §8)

        return mask.astype(np.int8)

@CONNECTION_MODELS.register("beggs_plenz")
class BeggsPlenzGaussianTopology(GaussianDistanceTypeTopology):
    """Beggs & Plenz (2003) 再現用の種別別ガウス結合プロファイル。具体値は YAML から読む。"""
    pass

@CONNECTION_MODELS.register("axon_growth")
class AxonGrowthTopology(BaseConnection):
    """軸索の伸長過程から結合を生成する。

    Sumi et al. (2025) Front. Neurosci. 19:1570783 §2.8 / Orlandi et al. (2013)。
    確率を距離の関数として与えるのではなく、**軸索を折れ線として実際に伸ばし、
    樹状突起円と交差したら結合する**。

    1. 軸索の全長 L を Rayleigh 分布から引く (平均 `mean_axon_length`)
    2. セグメント本数 = floor(L / `segment_length`)。0 本 (= 出力を持たない) も論文どおり許容
    3. 方向は角度のランダムウォーク: theta[k] = theta[k-1] + N(0, `angle_sigma`)。
       theta[0] は一様。sigma が小さい (0.1 rad) ので "pseudo-straight" な軌跡になる
    4. セグメントが 1 点でも `network.area` の外を通るなら境界処理 (既定は壁沿いへの偏向)
       を行い、**軸索は領域外へ出ない**。判定は端点ではなく線分全体
       (`BaseArea.segment_inside`) なので、セグメント長より狭い空隙も飛び越えない
    5. セグメントと細胞体の距離が `dendrite_radius` 以下で、**かつ接触点から細胞体まで
       領域内を見通せる**なら、その交差ごとに独立に確率 `connection_prob` で結合を張る

    config:
        mean_axon_length, segment_length, angle_sigma, dendrite_radius [um]
        connection_prob                         … 交差 1 回あたりの結合確率
        boundary: deflect | reflect | stop      … 領域境界に当たったときの挙動
        max_deflect (int)                       … 偏向の再試行回数。超えたらその軸索は打ち切り
        allow_self_connections (bool, 既定 False)

    `generate()` は `generate_sparse()` に委譲して散布するだけなので、密と疎は定義上一致する
    (`GaussianDistanceTypeTopology` が乱数ストリームを手で揃えているのとは対照的)。
    """

    supports_sparse = True

    # 交差判定を何セグメントずつ処理するか。1 セグメントあたりの候補が ~50 なので、
    # 8192 で 1 ブロック ~40 万行。
    _SEGMENT_BLOCK = 8192

    _BOUNDARY_MODES = ("deflect", "reflect", "stop")

    def _params(self):
        c = self.config
        p = SimpleNamespace(
            mean_axon_length=float(getattr(c, "mean_axon_length", 1100.0)),
            segment_length=float(getattr(c, "segment_length", 100.0)),
            angle_sigma=float(getattr(c, "angle_sigma", 0.1)),
            dendrite_radius=float(getattr(c, "dendrite_radius", 150.0)),
            connection_prob=float(getattr(c, "connection_prob", 0.2)),
            boundary=str(getattr(c, "boundary", "deflect")),
            max_deflect=int(getattr(c, "max_deflect", 4)),
            allow_self=bool(getattr(c, "allow_self_connections", False)),
        )
        if p.boundary not in self._BOUNDARY_MODES:
            raise ValueError(
                f"axon_growth の boundary は {self._BOUNDARY_MODES} のいずれかです (got {p.boundary!r})"
            )
        for name in ("mean_axon_length", "segment_length", "angle_sigma", "dendrite_radius"):
            if getattr(p, name) <= 0:
                raise ValueError(f"axon_growth の {name} は正である必要があります (got {getattr(p, name)})")
        if not 0.0 <= p.connection_prob <= 1.0:
            raise ValueError(f"axon_growth の connection_prob は [0, 1] です (got {p.connection_prob})")
        if p.max_deflect < 1:
            raise ValueError(f"axon_growth の max_deflect は 1 以上です (got {p.max_deflect})")
        return p

    def _validate_inputs(self):
        if self.coords is None:
            raise ValueError(
                "AxonGrowthTopology は空間座標を必要とします (network.space に no_space 以外を"
                " 指定してください)。"
            )
        if np.asarray(self.coords).shape[1] < 2:
            raise ValueError("AxonGrowthTopology は 2 次元以上の座標を必要とします。")
        if self.area is None:
            raise ValueError(
                "AxonGrowthTopology は network.area を必要とします (軸索を領域内に閉じ込めるため)。"
            )

    def _grow_axons(self, p):
        """軸索を折れ線として伸ばし、セグメント列を返す。

        境界処理があるためセグメント方向は逐次依存で、単純な cumsum にはできない。
        代わりに**ステップ index について反復し、ニューロン方向はベクトル化**する
        (ステップ数は平均 11、最大でも数十なので反復回数は小さい)。

        乱数の消費は rayleigh(N) -> uniform(N) -> 各ステップで normal(伸長中の本数) の順で、
        本数は n_seg から決まるので完全に決定的。偏向自体は乱数を引かない。

        Returns:
            (seg_start, seg_end, seg_owner, arc_length)
            seg_* は (S, 2) / (S,) で **(owner, step) の昇順**にソート済み。
            arc_length は実際に伸びた長さ (N,) [um] (打ち切られた軸索は短くなる)。
        """
        soma = np.ascontiguousarray(np.asarray(self.coords)[:, :2], dtype=np.float64)
        n = self.num_neurons

        # 1. 全長と本数。Rayleigh の平均は scale * sqrt(pi/2)。
        lengths = self.rng.rayleigh(p.mean_axon_length / np.sqrt(np.pi / 2.0), n)
        n_seg = np.floor(lengths / p.segment_length).astype(np.int64)
        # 2. 初期方向は等方ランダム
        theta = self.rng.uniform(0.0, 2.0 * np.pi, n)

        pos = soma.copy()
        alive = n_seg > 0

        starts: list = []
        ends: list = []
        owners: list = []
        steps: list = []

        max_steps = int(n_seg.max()) if n else 0
        for k in range(max_steps):
            act = np.nonzero(alive & (k < n_seg))[0]
            if act.size == 0:
                break

            # 3. 角度のランダムウォーク
            theta[act] += self.rng.normal(0.0, p.angle_sigma, act.size)
            u = np.stack([np.cos(theta[act]), np.sin(theta[act])], axis=1)
            base = pos[act]
            cand = base + p.segment_length * u

            # 4. 境界処理。領域外に出た軸索だけを壁沿いへ寄せる。
            #    判定は端点ではなく**線分全体**で行う。端点だけを見ると、セグメント長
            #    (100 um) より狭い空隙は「行き先が内部」なので通過してしまい、軸索が
            #    孤立した部分領域へ飛び移ってしまう (modular_4 の円↔ブリッジ間 40 um)。
            if p.boundary != "stop":
                for _ in range(p.max_deflect):
                    out = ~self.area.segment_inside(base, cand)
                    if not out.any():
                        break
                    nv = self.area.normal(cand[out])       # 外向き単位法線
                    uo = u[out]
                    dot = (uo * nv).sum(axis=1, keepdims=True)
                    if p.boundary == "deflect":
                        ut = uo - dot * nv                 # 接線成分だけ残す = 壁沿い
                    else:                                  # reflect
                        ut = uo - 2.0 * dot * nv           # 鏡面反射
                    nrm = np.linalg.norm(ut, axis=1, keepdims=True)
                    # 壁に正面衝突すると接線成分が消えて向きが決まらない。法線を 90° 回して逃がす。
                    fallback = np.stack([-nv[:, 1], nv[:, 0]], axis=1)
                    u[out] = np.where(nrm > 1e-9, ut / np.maximum(nrm, 1e-12), fallback)
                    cand[out] = base[out] + p.segment_length * u[out]
                theta[act] = np.arctan2(u[:, 1], u[:, 0])

            # 偏向しても領域内に収まらないもの (凹の袋小路) はここで打ち切る。
            outside = ~self.area.segment_inside(base, cand)
            if outside.any():
                alive[act[outside]] = False

            keep = np.nonzero(~outside)[0]
            if keep.size == 0:
                continue
            kept = act[keep]
            starts.append(base[keep])
            ends.append(cand[keep])
            owners.append(kept)
            steps.append(np.full(kept.size, k, dtype=np.int64))
            pos[kept] = cand[keep]

        if not owners:
            empty2 = np.zeros((0, 2), dtype=np.float64)
            return empty2, empty2, np.zeros(0, dtype=np.int64), np.zeros(n, dtype=np.float64)

        seg_start = np.concatenate(starts, axis=0)
        seg_end = np.concatenate(ends, axis=0)
        seg_owner = np.concatenate(owners)
        seg_step = np.concatenate(steps)

        # ステップごとに積んだので今は (step, owner) 順。(owner, step) 順に並べ替えて、
        # 折れ線としての順序と、以降の乱数消費順を決定論的にする。
        order = np.lexsort((seg_step, seg_owner))
        seg_start, seg_end, seg_owner = seg_start[order], seg_end[order], seg_owner[order]

        arc_length = np.bincount(seg_owner, minlength=n).astype(np.float64) * p.segment_length
        return seg_start, seg_end, seg_owner, arc_length

    def generate_sparse(self):
        self._validate_inputs()
        p = self._params()

        seg_start, seg_end, seg_owner, arc_length = self._grow_axons(p)
        self._arc_length = arc_length

        soma = np.ascontiguousarray(np.asarray(self.coords)[:, :2], dtype=np.float64)
        empty = np.array([], dtype=np.int32)
        if seg_owner.size == 0:
            return empty, empty

        # 樹状突起円との交差判定。セグメント中点から半径 (セグメント長/2 + 樹状突起半径) 以内に
        # 細胞体があることが必要条件なので、まず KD-tree でその候補を絞る。
        tree = cKDTree(soma)
        query_r = 0.5 * p.segment_length + p.dendrite_radius

        pre_list: list = []
        post_list: list = []

        for s0 in range(0, seg_owner.size, self._SEGMENT_BLOCK):
            s1 = min(s0 + self._SEGMENT_BLOCK, seg_owner.size)
            a = seg_start[s0:s1]
            b = seg_end[s0:s1]
            own = seg_owner[s0:s1]

            cand = tree.query_ball_point(0.5 * (a + b), query_r)
            counts = np.fromiter((len(c) for c in cand), dtype=np.int64, count=len(cand))
            total = int(counts.sum())
            if total == 0:
                continue
            flat_j = np.fromiter(itertools.chain.from_iterable(cand), dtype=np.int64, count=total)
            seg_idx = np.repeat(np.arange(len(cand), dtype=np.int64), counts)

            # query_ball_point の返す順序は未定義。ここで固定しないと再現性が壊れる。
            order = np.lexsort((flat_j, seg_idx))
            seg_idx, flat_j = seg_idx[order], flat_j[order]

            # 点-線分の厳密な距離。t は線分上の最近接点のパラメータ [0, 1]。
            ab = b[seg_idx] - a[seg_idx]
            ap = soma[flat_j] - a[seg_idx]
            ab2 = np.einsum("ij,ij->i", ab, ab)
            t = np.clip(np.einsum("ij,ij->i", ap, ab) / np.maximum(ab2, 1e-12), 0.0, 1.0)
            perp = ap - t[:, None] * ab
            hit = np.einsum("ij,ij->i", perp, perp) <= p.dendrite_radius ** 2

            # 自己結合は**抽選の前に**落とす。密版が疎版に委譲する構成なので、
            # GaussianDistanceTypeTopology のように対角のドローを消費して捨てる必要がない。
            if not p.allow_self:
                hit &= own[seg_idx] != flat_j
            if not hit.any():
                continue

            # 樹状突起も領域の外へは出られない。ユークリッド距離だけで判定すると、
            # 半径 (150 um) より狭い空隙の向こう側にある細胞体に届いてしまう
            # (modular_4 では孤立した円とブリッジの間 40 um)。接触点から細胞体までが
            # 領域内を通ること = 見通しが立つことを要求する。**抽選より前**に落とすのは、
            # 「そもそも接触していない」ものに乱数を消費させないため (自己結合と同じ方針)。
            contact = a[seg_idx] + t[:, None] * ab
            hit[hit] &= self.area.segment_inside(contact[hit], soma[flat_j[hit]])
            if not hit.any():
                continue
            seg_idx, flat_j = seg_idx[hit], flat_j[hit]

            # 交差 1 回につき独立に 1 回抽選する (Orlandi et al. 2013)。同じ相手を複数の
            # セグメントが横切れば、その回数だけ試行が行われる。
            accept = self.rng.random(seg_idx.size) < p.connection_prob
            if not accept.any():
                continue
            pre_list.append(own[seg_idx[accept]])
            post_list.append(flat_j[accept])

        if not pre_list:
            return empty, empty

        pairs = np.stack([np.concatenate(pre_list), np.concatenate(post_list)], axis=1)
        # 重複除去 + 行優先ソート (np.unique(axis=0) は辞書式に並べる)
        pairs = np.unique(pairs, axis=0)
        return pairs[:, 0].astype(np.int32), pairs[:, 1].astype(np.int32)

    def generate(self):
        """密な (N, N) マスク。疎版を呼んで散布するだけなので、密と疎は定義上一致する。"""
        rows, cols = self.generate_sparse()
        mask = np.zeros((self.num_neurons, self.num_neurons), dtype=np.int8)
        mask[rows, cols] = 1
        return mask

    def describe_axes(self) -> Dict[str, Any]:
        """実際に伸びた軸索の長さを数値軸として宣言する (generate 後に有効)。

        `layout.order_by("axon_length")` で軸索長順に並べ替えられるので、軸索長と出次数の
        関係や、境界で打ち切られた軸索の分布を解析できる。
        """
        arc = getattr(self, "_arc_length", None)
        if arc is None:
            return {}
        return {"axon_length": arc}


@CONNECTION_MODELS.register("prob_based_block")
class BlockRandomTopology(BaseConnection):
    def generate(self):
        """ブロック分割ベースの確率結合を生成する。

        モジュール境界は、NetworkLayout に `module` 軸があればそれを用いる(空間モデルが
        `describe_axes()` で宣言したもの)。無ければ config.num_modules で等分する。
        num_modules=1(または単一モジュール)では単一ブロックのランダム結合として振る舞う。
        モジュールのIDが連続か散在かは問わない。
        """
        within_module_connection_prob = self.config.within_module_connection_prob
        between_module_connection_prob = self.config.between_module_connection_prob
        allow_self_connections = self.config.allow_self_connections

        for prob_name, prob_value in (
            ("within_module_connection_prob", within_module_connection_prob),
            ("between_module_connection_prob", between_module_connection_prob),
        ):
            if not isinstance(prob_value, (int, float)) or isinstance(prob_value, bool):
                raise ValueError(f"{prob_name} must be a real number.")
            if not 0.0 <= prob_value <= 1.0:
                raise ValueError(f"{prob_name} must be between 0.0 and 1.0.")

        if not isinstance(allow_self_connections, bool):
            raise ValueError("allow_self_connections must be a boolean.")

        module_ids = self._module_partition()
        num_modules = len(module_ids)
        mask = np.zeros((self.num_neurons, self.num_neurons), dtype=np.int8)

        def write_block(src_idx: int, tgt_idx: int, prob: float) -> None:
            """モジュール src_idx → tgt_idx のブロックに確率 prob の結合を書き込む。

            乱数の消費は「ブロック形状ぶんを1回 rand」なので、モジュール分割が同じなら
            ID が連続でも散在でも同一の実現になる。
            """
            block = self.rng.rand(len(module_ids[src_idx]), len(module_ids[tgt_idx])) < prob
            mask[np.ix_(module_ids[src_idx], module_ids[tgt_idx])] = block.astype(np.int8)

        # モジュール内の結合を生成
        for module_idx in range(num_modules):
            write_block(module_idx, module_idx, within_module_connection_prob)

        # モジュール間の結合を生成: 各モジュールを隣接モジュールと双方向に接続する。
        for module_idx in range(num_modules):
            # module_idx + 1 == num_modules のとき IndexError になる。よって % 演算子で
            # 0 に戻す(環状に隣接させる)。
            next_idx = (module_idx + 1) % num_modules
            write_block(module_idx, next_idx, between_module_connection_prob)
            write_block(next_idx, module_idx, between_module_connection_prob)

        if not allow_self_connections:
            np.fill_diagonal(mask, 0)

        return mask

    def _module_partition(self) -> list[np.ndarray]:
        """モジュール分割をグローバルID集合のリストとして返す。

        NetworkLayout に `module` 軸があればそれを採用し、無ければ config.num_modules で
        グローバルID空間を等分する。ID が連続かどうかは問わない。
        """
        if self.layout is not None and self.layout.has_axis("module"):
            return list(self.layout.ids_by("module").values())

        num_modules = self.config.num_modules
        if not isinstance(num_modules, int) or isinstance(num_modules, bool):
            raise ValueError("num_modules must be an integer.")
        if num_modules < 1:
            raise ValueError("num_modules must be at least 1.")
        if num_modules > self.num_neurons:
            raise ValueError("num_modules must not exceed num_neurons.")

        module_ids = []
        for module_idx in range(num_modules):
            start = int(module_idx * self.num_neurons / num_modules)
            end = int((module_idx + 1) * self.num_neurons / num_modules)
            module_ids.append(np.arange(start, end))
        return module_ids
