import itertools
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
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

# 軸索の折れ線 (AxonGeometry) を書き出す npz の名前。`save()` と `load()` が対で使う。
# 「どの軸索がどのブリッジを通ったか」= 損傷実験で必要になる記録。
AXONS_NAME = "axon_geometry.npz"


@dataclass(frozen=True)
class AxonGeometry:
    """軸索の折れ線と、各シナプスがどこで接触したかを持つプレーンなデータ。

    `AxonGrowthTopology.axon_geometry()` が返す。**結合そのものではなく、結合が
    どうやってできたかの記録**なので、シミュレーションには一切使わない (描画と解析用)。
    折れ線を復元するのは受け取った側の仕事で、ここには配列しか入れない
    (`src/utils/plotting` がダックタイピングで読めるようにするため)。

    セグメントは `(owner, step)` 昇順に並んでいるので、ニューロン i の軸索は
    `seg_start[offsets[i]:offsets[i+1]]` が始点列、`seg_end[offsets[i+1]-1]` が終端。
    連続するセグメントは端点を共有する (`seg_start[k+1] == seg_end[k]`)。

    Attributes:
        seg_start, seg_end: (S, 2) セグメントの両端 [um]
        seg_owner:          (S,)   セグメントの持ち主 (pre ニューロンのグローバル ID)
        offsets:            (N+1,) owner ごとのセグメント範囲。軸索を持たない owner は空区間
        pre, post:          (M,)   結合ペア。generate_sparse() の戻り値と同一・同順
        contact_seg:        (M,)   その結合を作ったセグメントの**グローバル** index
        contact_t:          (M,)   接触点のセグメント上パラメータ [0, 1]
    """

    seg_start: np.ndarray
    seg_end: np.ndarray
    seg_owner: np.ndarray
    offsets: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    contact_seg: np.ndarray
    contact_t: np.ndarray

    def save(self, path: "Path | str") -> "Path":
        """8 本の配列を npz として `path` へ書き出し、そのパスを返す。

        **置き場所 (どの run ディレクトリか) は呼び出し側が決める。** ここが知っているのは
        「軸索の記録をどう直列化するか」と、その名前 (`AXONS_NAME`) までで、`outputs/` の
        規約は持たない (`NetworkLayout.save_axes()` と同じ切り分け)。

        軸索の軌跡は seed から再現できるが、再現には同じコード・同じエリア・同じ
        `segment_length` が要る (CLAUDE.md 注記 16b / 16d)。損傷実験の対象は
        **その run で実際に伸びた軸索**なので、記録として残す。

        書き出す配列はフィールドそのままで、dtype も値も変換しない。連続セグメントが
        端点を共有する分だけ座標は 2 倍冗長だが、保存形式がその不変条件に依存しない
        ほうが壊れにくい。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})
        return path

    @classmethod
    def load(cls, path: "Path | str") -> "AxonGeometry":
        """`save()` が書いた npz から復元する。

        配列が 1 本でも欠けていれば、**何が無いのかを挙げて ValueError**。部分的に
        読めるふりをすると、折れ線と接触点の対応が黙って壊れた状態で解析へ流れる。
        """
        path = Path(path)
        names = [f.name for f in fields(cls)]
        with np.load(path, allow_pickle=False) as data:
            missing = [name for name in names if name not in data.files]
            if missing:
                raise ValueError(
                    f"軸索の記録として読めません ({path}): {', '.join(missing)} がありません。"
                    f" AxonGeometry.save() が書いた npz を渡してください。"
                )
            return cls(**{name: data[name] for name in names})


@CONNECTION_MODELS.register("axon_growth")
class AxonGrowthTopology(BaseConnection):
    """軸索の伸長過程から結合を生成する。

    Sumi et al. (2025) Front. Neurosci. 19:1570783 §2.8 / Orlandi et al. (2013)。
    確率を距離の関数として与えるのではなく、**軸索を折れ線として実際に伸ばし、
    樹状突起円と交差したら結合する**。

    1. 軸索の全長 L を Rayleigh 分布から引く (平均 `mean_axon_length`)
    2. ステップ数 = floor(L / `segment_length`)。0 本 (= 出力を持たない) も論文どおり許容
    3. 方向は角度のランダムウォーク: theta[k] = theta[k-1] + N(0, `angle_sigma`)。
       theta[0] は一様。sigma が小さい (0.1 rad) ので "pseudo-straight" な軌跡になる
    4. `network.area` の外へは出ない。壁に当たったら **advance-and-slide** —
       交点まで進み、**そこでの**法線で向きを変え、残りの長さを壁沿いに使う
       (`_grow_axons` 参照)。1 ステップが壁で 2 本以上のセグメントに割れることがある。
       滑り直しの打ち切りは回数ではなく**状態が不変になったこと**で判定する
    5. セグメントと細胞体の距離が `dendrite_radius` 以下で、**かつ接触点から細胞体まで
       領域内を見通せる**なら結合の候補。抽選は**樹状突起円への侵入 1 回につき 1 回**で、
       確率は `connection_prob` (論文の規則。詳細は `generate_sparse`)

    config:
        mean_axon_length, segment_length, angle_sigma, dendrite_radius [um]
        connection_prob                         … 樹状突起円への侵入 1 回あたりの結合確率
        boundary: deflect | reflect | stop      … 領域境界に当たったときの挙動
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

    # 1 ステップぶんの伸長を「壁まで進む → 壁沿いに残りを使う」に分ける際の最小長 [um]。
    # これ未満しか進めなかった移動はセグメントとして出さない (長さ 0 の線分を作らない)。
    _MIN_ADVANCE = 1e-6

    # 壁沿いの滑り直しの上限。本来の停止条件は「位置・向き・残りがすべて不変 = 以降ずっと
    # no-op」の検出 (`_grow_axons` 参照) で、これは循環に対する保険にすぎない。
    # 実測の必要回数: 円板 (滑らかな凸壁) は最大 3 回、hierarchical_modular_grid は最大
    # 105 回 — 曲面へ引き戻しながら滑る行がナノメートル未満の残りを刻み続けるため。
    # 256 なら後者でもビット単位で 512 と一致する (grid で segments/arc/synapses が完全一致)。
    _MAX_SLIDE_ITERS = 256

    def _grow_axons(self, p):
        """軸索を折れ線として伸ばし、セグメント列を返す。

        1 ステップ = 弧長 `segment_length` ぶんの伸長。壁に当たらなければ 1 本の直線
        セグメントになり、当たれば **advance-and-slide** で複数本に割れる:

            1. 交点まで進む            (`BaseArea.first_exit` で交点を求める)
            2. **交点での**法線で向きを付け替える (deflect = 接線成分だけ残す)
            3. **残りの長さ**で 1 に戻る

        曲がるのは壁に着いてからで、1 ステップの弧長は壁に当たっても保存される
        (向きが変わるだけで速度は落ちない)。

        **打ち切りは回数ではなく「状態が不変になったこと」で決める。** 進めた場合は残りが
        減るので反復は進むが、`t = 0` (1 歩も進めない) だと残りは減らない。そのとき向きも
        変わらなければ (位置, 向き, 残り) が完全に同じなので、以降は何回まわしても同じ
        結果にしかならない — そこで打ち切る。射影 `u - (u·n)n` は冪等なので、多角形の角
        では 2 回目の反復でちょうどこの不変状態になる。**1 回目の偏向は切らない**こと:
        壁に正面から当たった軸索は `t = 0` でも向きが変わり、次の反復で壁沿いに進む。
        `_MAX_SLIDE_ITERS` は万一の循環に対する保険で、通常は効かない。

        境界処理があるためセグメント方向は逐次依存で、単純な cumsum にはできない。
        代わりに**ステップ index について反復し、ニューロン方向はベクトル化**する
        (ステップ数は平均 11、最大でも数十なので反復回数は小さい)。

        乱数の消費は rayleigh(N) -> uniform(N) -> 各ステップで normal(伸長中の本数) の順で、
        本数は n_seg から決まるので完全に決定的。境界処理は乱数を引かない。

        Returns:
            (seg_start, seg_end, seg_owner, arc_length)
            seg_* は (S, 2) / (S,) で **(owner, step, 壁で割れた順) の昇順**にソート済み。
            壁沿いに割れたセグメントは `segment_length` より短いので、arc_length は
            本数ではなく**実際の長さの総和** (N,) [um]。
        """
        soma = np.ascontiguousarray(np.asarray(self.coords)[:, :2], dtype=np.float64)
        n = self.num_neurons
        eps = self._MIN_ADVANCE * p.segment_length

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
        keys: list = []          # (step, 壁で割れた順) を 1 本の整数に畳んだ並べ替えキー
        # 並べ替えキー k * sub_stride + sub が (k, sub) の辞書順と一致するための桁。
        sub_stride = self._MAX_SLIDE_ITERS + 2

        def emit(a, b, who, key):
            starts.append(a.copy())
            ends.append(b.copy())
            owners.append(who)
            keys.append(np.full(who.size, key, dtype=np.int64))

        max_steps = int(n_seg.max()) if n else 0
        for k in range(max_steps):
            act = np.nonzero(alive & (k < n_seg))[0]
            if act.size == 0:
                break

            # 3. 角度のランダムウォーク
            theta[act] += self.rng.normal(0.0, p.angle_sigma, act.size)
            u = np.stack([np.cos(theta[act]), np.sin(theta[act])], axis=1)
            base = pos[act].copy()
            remaining = np.full(act.size, p.segment_length)
            live = np.ones(act.size, dtype=bool)
            on_wall = np.zeros(act.size, dtype=bool)   # 壁の上にいるか (次の一歩は壁沿い)
            advanced = np.zeros(act.size, dtype=bool)  # このステップで少しでも進めたか

            # 4. 壁まで進んでは滑る、を残りが尽きるまで繰り返す。
            for sub in range(self._MAX_SLIDE_ITERS):
                idx = np.nonzero(live & (remaining > eps))[0]
                if idx.size == 0:
                    break
                b0, u0, r0 = base[idx], u[idx], remaining[idx]
                cand = b0 + r0[:, None] * u0
                # この反復で実際に前へ進めたか (行ごと)。進めず向きも変わらない行は
                # 状態が不変なので、この後 live から外して打ち切る。
                moved = np.zeros(idx.size, dtype=bool)

                # 壁の上から接線方向へ出た直後は、壁が曲がっていると即座に外へ出てしまう
                # (円の接線は接点以外すべて外側)。SDF で表面へ引き戻すと「曲面に沿って
                # 滑る」動きになる。直線の壁なら sdf=0 のままなので何も起きない。
                snapped = np.zeros(idx.size, dtype=bool)
                slide = np.nonzero(on_wall[idx])[0]
                if slide.size:
                    d = self.area.sdf(cand[slide])
                    off = np.nonzero(d > 0.0)[0]
                    if off.size:
                        pts = cand[slide[off]]
                        pulled = pts - d[off][:, None] * self.area.normal(pts)
                        # SDF が真の距離とは限らない (union は min なので下界) ため、引き戻しで
                        # かえって遠のくことがある。1 ステップの弧長を超えないよう頭を押さえる。
                        rows = slide[off]
                        v = pulled - b0[rows]
                        vlen = np.linalg.norm(v, axis=1)
                        over = vlen > r0[rows]
                        if over.any():
                            scale = np.where(over, r0[rows] / np.maximum(vlen, 1e-12), 1.0)
                            pulled = b0[rows] + v * scale[:, None]
                        cand[rows] = pulled
                        snapped[rows] = True

                inside = self.area.segment_inside(b0, cand)
                clear = np.nonzero(inside)[0]
                if clear.size:                       # 壁に当たらない = そのまま伸ばして終わり
                    g = idx[clear]
                    emit(b0[clear], cand[clear], act[g], k * sub_stride + sub)
                    base[g] = cand[clear]
                    # 表面へ引き戻したぶんだけ弦は短い (円 R=1500・100 um ステップで 0.02 um)。
                    # 端数は次の反復で使われるので、実際に進んだ長さで残りを減らす。
                    remaining[g] = np.maximum(
                        r0[clear] - np.linalg.norm(cand[clear] - b0[clear], axis=1), 0.0)
                    # 表面へ引き戻した移動は壁の上で終わっている。壁から離れたことにすると、
                    # 次の反復で接線がまた外へ出て 1 回ぶん無駄になる。
                    on_wall[g] = snapped[clear]
                    advanced[g] = True
                    moved[clear] = True

                blocked = np.nonzero(~inside)[0]
                if blocked.size == 0:
                    continue
                g = idx[blocked]
                b1, u1, r1 = b0[blocked], u0[blocked], r0[blocked]
                c1 = cand[blocked]

                # 交点まで進む。first_exit は必ず領域内の点を返すので貼り付かない。
                t = self.area.first_exit(b1, c1)
                x = b1 + t[:, None] * (c1 - b1)
                # 進めた距離が無視できるほど短いときは、セグメントを作らず**位置も動かさない**
                # (動かすと折れ線が微小に途切れる)。向きだけ付け替えて次の反復へ。
                grew = np.nonzero(t * r1 > eps)[0]
                if grew.size:
                    gg = g[grew]
                    emit(b1[grew], x[grew], act[gg], k * sub_stride + sub)
                    base[gg] = x[grew]
                    remaining[gg] = r1[grew] * (1.0 - t[grew])
                    advanced[gg] = True
                    moved[blocked[grew]] = True

                if p.boundary == "stop":             # 壁で止まる = そこで伸長終了
                    live[g] = False
                    continue
                on_wall[g] = True

                # 交点での法線で向きを付け替える。deflect は壁に食い込む成分だけを消す。
                nv = self.area.normal(x)
                dot = (u1 * nv).sum(axis=1, keepdims=True)
                if p.boundary == "deflect":
                    ut = u1 - dot * nv               # 接線成分だけ残す = 壁沿い
                else:                                # reflect
                    ut = u1 - 2.0 * dot * nv         # 鏡面反射
                nrm = np.linalg.norm(ut, axis=1, keepdims=True)
                # 壁に正面衝突すると接線成分が消えて向きが決まらない。法線を 90° 回して逃がす。
                fallback = np.stack([-nv[:, 1], nv[:, 0]], axis=1)
                u_new = np.where(nrm > 1e-9, ut / np.maximum(nrm, 1e-12), fallback)

                # 進めず、向きも変わらなかった行は (位置, 向き, 残り) が完全に同じ =
                # 以降の反復は no-op にしかならないので、このステップを打ち切る。
                # 射影は冪等なので、多角形の角ではここに落ちる (u·n = 0 -> ut = u)。
                # **1 回目の偏向は向きが変わるので frozen にならない** — 壁に正面から
                # 当たった軸索が次の反復で壁沿いに進む経路は残る。
                frozen = (~moved[blocked]) & ((u_new * u1).sum(axis=1) > 1.0 - 1e-12)
                live[g[frozen]] = False
                u[g] = u_new

            # 5. **1 mm も進めなかった**軸索だけを打ち切る (凹の袋小路)。
            #    少しでも進めたなら生かす — 曲面を追う都合で 1 ステップが数 um 足りずに
            #    終わることがあり、それを袋小路と同一視すると全長が目減りしてしまう。
            #    本当の袋小路なら次のステップで進行量 0 になり、そこで止まる。
            stuck = ~advanced
            if stuck.any():
                alive[act[stuck]] = False
            pos[act] = base
            theta[act] = np.arctan2(u[:, 1], u[:, 0])

        if not owners:
            empty2 = np.zeros((0, 2), dtype=np.float64)
            return empty2, empty2, np.zeros(0, dtype=np.int64), np.zeros(n, dtype=np.float64)

        seg_start = np.concatenate(starts, axis=0)
        seg_end = np.concatenate(ends, axis=0)
        seg_owner = np.concatenate(owners)
        seg_key = np.concatenate(keys)

        # ステップごとに積んだので今は (step, owner) 順。(owner, step, 割れた順) に並べ替えて、
        # 折れ線としての順序と、以降の乱数消費順を決定論的にする。
        order = np.lexsort((seg_key, seg_owner))
        seg_start, seg_end, seg_owner = seg_start[order], seg_end[order], seg_owner[order]

        # 壁沿いに割れたセグメントは短いので、本数 x segment_length では長さにならない。
        arc_length = np.bincount(
            seg_owner, weights=np.linalg.norm(seg_end - seg_start, axis=1), minlength=n,
        ).astype(np.float64)
        return seg_start, seg_end, seg_owner, arc_length

    def generate_sparse(self):
        """軸索を伸ばし、樹状突起円への**侵入 1 回につき 1 回**抽選して COO を返す。

        抽選の単位が「侵入」であることが要点 (Sumi et al. 2025 §2.8 "whenever the axon of
        neuron i crossed the area covered by the dendritic tree of neuron j with a 20%
        probability" / Orlandi et al. 2013 "invaded")。セグメントごとに引くと、同じ 1 回の
        侵入を刻んだ本数だけ試行することになり、実効確率が `segment_length` に依存して
        跳ね上がる (100 um 刻みで約 2.2 倍、25 um 刻みで約 3.9 倍)。侵入単位なら結果は
        刻みにほぼ依存しない = 離散化ではなく幾何が結合を決める。
        """
        self._validate_inputs()
        p = self._params()

        seg_start, seg_end, seg_owner, arc_length = self._grow_axons(p)
        self._arc_length = arc_length
        self._geometry = None

        soma = np.ascontiguousarray(np.asarray(self.coords)[:, :2], dtype=np.float64)
        empty = np.array([], dtype=np.int32)
        if seg_owner.size == 0:
            return empty, empty

        # 樹状突起円との交差判定。セグメント中点から半径 (セグメント長/2 + 樹状突起半径) 以内に
        # 細胞体があることが必要条件なので、まず KD-tree でその候補を絞る。
        tree = cKDTree(soma)
        query_r = 0.5 * p.segment_length + p.dendrite_radius

        # 見つけた「侵入」を貯める。抽選はブロック処理が終わってから一括で行う
        # (ブロック分割は性能のための内部都合なので、それが実現に影響しないように)。
        inv_pre: list = []
        inv_post: list = []
        inv_seg: list = []      # 侵入が始まったセグメントの全体 index
        inv_t: list = []        # そのセグメント上の接触位置 [0, 1]
        # ブロック末尾のセグメントで継続中の侵入 (owner*N + post で符号化)。
        # 次のブロックの先頭セグメントへ続いていれば、それは同じ 1 回の侵入。
        carry = np.zeros(0, dtype=np.int64)

        for s0 in range(0, seg_owner.size, self._SEGMENT_BLOCK):
            s1 = min(s0 + self._SEGMENT_BLOCK, seg_owner.size)
            # carry は「直前のブロックの末尾から続いている侵入」だけを指す。ここで空に
            # しておくことで、途中の continue で抜けたブロックが古い carry を持ち越さない。
            prev_carry, carry = carry, np.zeros(0, dtype=np.int64)
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
            # t も一緒に絞る。接触位置を後で復元するため 3 本の対応を崩さない。
            seg_idx, flat_j, t = seg_idx[hit], flat_j[hit], t[hit]

            # --- 「樹状突起円への侵入 1 回につき 1 抽選」に畳む ---
            # 論文の規則は *侵入* が単位 (Sumi et al. 2025 §2.8: "whenever the axon of
            # neuron i crossed the area covered by the dendritic tree of neuron j with a
            # 20% probability" / Orlandi et al. 2013: "invaded")。セグメントごとに引くと
            # 同じ 1 回の侵入を刻んだ本数だけ試行してしまい、実効確率が segment_length に
            # 依存して跳ね上がる (100 um 刻みで約 2.2 倍、25 um 刻みで約 3.9 倍)。
            # 侵入 = 同じ (軸索, 相手) に**連続したセグメント index** で触れている一続き。
            # 円を一度出てから入り直せば index が飛ぶので、別の侵入として数えられる。
            g_seg = s0 + seg_idx
            g_own = own[seg_idx]
            order = np.lexsort((g_seg, flat_j, g_own))
            g_seg, g_own, flat_j, t = g_seg[order], g_own[order], flat_j[order], t[order]

            start_of_run = np.ones(g_seg.size, dtype=bool)
            start_of_run[1:] = (
                (g_own[1:] != g_own[:-1])
                | (flat_j[1:] != flat_j[:-1])
                | (g_seg[1:] != g_seg[:-1] + 1)
            )
            # ブロックの切れ目で分断された侵入は、前ブロックで既に抽選済み。
            if prev_carry.size:
                head = np.nonzero(start_of_run & (g_seg == s0))[0]
                if head.size:
                    keys = g_own[head] * self.num_neurons + flat_j[head]
                    start_of_run[head[np.isin(keys, prev_carry)]] = False
            tail = g_seg == (s1 - 1)
            carry = g_own[tail] * self.num_neurons + flat_j[tail]

            firsts = np.nonzero(start_of_run)[0]
            inv_pre.append(g_own[firsts])
            inv_post.append(flat_j[firsts])
            inv_seg.append(g_seg[firsts])
            inv_t.append(t[firsts])

        empty_geom = (empty, empty, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64))
        if not inv_pre:
            self._geometry = self._make_geometry(seg_start, seg_end, seg_owner, *empty_geom)
            return empty, empty

        pre = np.concatenate(inv_pre)
        post = np.concatenate(inv_post)
        c_seg = np.concatenate(inv_seg)
        c_t = np.concatenate(inv_t)

        # 侵入を (軸索, 相手, 侵入開始セグメント) の正準順に並べてから 1 侵入 1 抽選。
        # ブロック順のまま引くと、_SEGMENT_BLOCK を変えただけで乱数の割り当てが変わり、
        # 同じ seed で別のネットワークになってしまう。
        order = np.lexsort((c_seg, post, pre))
        pre, post, c_seg, c_t = pre[order], post[order], c_seg[order], c_t[order]
        accept = self.rng.random(pre.size) < p.connection_prob
        if not accept.any():
            self._geometry = self._make_geometry(seg_start, seg_end, seg_owner, *empty_geom)
            return empty, empty
        pre, post, c_seg, c_t = pre[accept], post[accept], c_seg[accept], c_t[accept]

        # 同じ相手に 2 回侵入して 2 回とも当たると重複するので畳む。並びは既に
        # (pre, post, seg) 昇順なので、残るのは**最初に当たった侵入**。
        keep = np.ones(pre.size, dtype=bool)
        keep[1:] = (pre[1:] != pre[:-1]) | (post[1:] != post[:-1])
        rows = pre[keep].astype(np.int32)
        cols = post[keep].astype(np.int32)
        self._geometry = self._make_geometry(
            seg_start, seg_end, seg_owner, rows, cols, c_seg[keep], c_t[keep],
        )
        return rows, cols

    def _make_geometry(self, seg_start, seg_end, seg_owner, pre, post, contact_seg, contact_t):
        """セグメント列を owner ごとの区間に切る索引を付けて AxonGeometry にまとめる。

        セグメントは (owner, step) 昇順なので、owner ごとの本数を累積するだけで区間になる。
        """
        counts = np.bincount(seg_owner, minlength=self.num_neurons)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return AxonGeometry(
            seg_start=seg_start, seg_end=seg_end, seg_owner=seg_owner, offsets=offsets,
            pre=np.asarray(pre, dtype=np.int64), post=np.asarray(post, dtype=np.int64),
            contact_seg=np.asarray(contact_seg, dtype=np.int64),
            contact_t=np.asarray(contact_t, dtype=np.float64),
        )

    def axon_geometry(self):
        """伸ばした軸索の折れ線と各シナプスの接触位置を返す (generate 後に有効)。

        生成前は None。`describe_axes()` が生成前に {} を返すのと同じ作法で、
        呼ぶ側 (可視化スクリプト) は None を「この図は描けない」として扱う。
        """
        return getattr(self, "_geometry", None)

    def generate(self):
        """密な (N, N) マスク。疎版を呼んで散布するだけなので、密と疎は定義上一致する。"""
        rows, cols = self.generate_sparse()
        mask = np.zeros((self.num_neurons, self.num_neurons), dtype=np.int8)
        mask[rows, cols] = 1
        return mask

    def describe_axes(self) -> Dict[str, Any]:
        """実際に伸びた軸索の長さ [um] を数値軸として宣言する (generate 後に有効)。

        壁沿いに割れたセグメントは短いので、これは本数 x segment_length ではなく
        折れ線の実長。打ち切られた軸索は名目長 floor(L/seg)*seg より短くなる。

        `layout.order_by("axon_length")` で軸索長順に並べ替えられるので、軸索長と出次数の
        関係や、境界で打ち切られた軸索の分布を解析できる。
        """
        arc = getattr(self, "_arc_length", None)
        if arc is None:
            return {}
        return {"axon_length": arc}


@CONNECTION_MODELS.register("axon_growth_fine")
class AxonGrowthFineTopology(AxonGrowthTopology):
    """`axon_growth` の `segment_length` を細かくした版。挙動は親と同一で、値は YAML から。

    `profile_name` は YAML のキーであると同時にレジストリのキーなので、値だけ違う
    プロファイルにも空のサブクラスが要る (`BeggsPlenzGaussianTopology` と同じ作法)。
    細かい Δs が要るのは、ブリッジのような細い通路を軸索に通させたいとき。

    **`angle_sigma` は `0.1 * sqrt(Δs / 100)` で合わせること。** 持続長
    `Lp = 2Δs/σ² = 20 mm` が軌跡の直線性そのものなので、Δs だけ変えると別のモデルになる。
    再計算の規則・検証手順・Δs を変えると副次的に動く量 (弧長の切り捨て、
    `_SEGMENT_SAMPLES` による分解能、`_MAX_SLIDE_ITERS` の必要回数、コスト) は
    `docs/technical/axon_growth_segment_length.md` にまとめてある。
    """
    pass


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
