"""ニューロン間の機能的結合 —— 発火率時系列のゼロラグ Pearson 相関。

Watanabe et al. (2026) *Communications Engineering* **5**:150 の "Neuronal correlation"
(式 1) をそのまま実装したもの。ニューロン i, j の相関は

    CC_ij = Σ_t [r_i(t) − r̄_i][r_j(t) − r̄_j]
            / sqrt(Σ_t [r_i(t) − r̄_i]²) / sqrt(Σ_t [r_j(t) − r̄_j]²)

で、ネットワークの平均相関は **対角を除いた**全ペアにわたる CC_ij の平均。

## r(t) の作り方

スパイク時刻を `RATE_BIN_MS` のビンで数えて発火率にする。Pearson 相関は線形変換に
不変なので「カウント」でも「Hz」でも値は同じ。

**ビン幅は値を決めるパラメータで、r(b) には 3 つの領域がある。**

    b << 30 ms   r ∝ b        測っているのは b そのもの (情報が増えない)
    b ≈ 30-250   ほぼ頭打ち   相互共分散を取り切っている
    b >> 250 ms  r が低下     各ニューロン固有の遅い揺らぎが分母に入る (なまり)

境目の 30 ms は相互相関ヒストグラムの半値全幅と一致する。

既定の 50 ms は**論文のフレームレート (20 fps) と同じ幅**。頭打ちの領域に入っており、
偶然との分離も最大 (30 ms) とほぼ変わらない。根拠の数値は
`docs/technical/criticality_notes.md`。

## 分散 0 のニューロン

窓の中で 1 発も撃たない (あるいは全ビンで同数撃つ) ニューロンは分母が 0 になり、
相関は**未定義**。0 で埋めずに `nan` を返し、平均からも除く —— 「相関が無い」と
「測れない」は別で、前者に畳むと損傷で黙ったニューロンが平均を 0 側へ引き、後者に
畳むと逆に平均が上がる。どちらへ倒すかを図と指標が別々に決めないよう、ここで
`nan` に固定して**除いた本数を一緒に返す**。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 発火率時系列のビン幅 [ms]。論文のフレームレート (20 fps) に合わせた幅 (上の docstring)。
RATE_BIN_MS = 50.0


def firing_rate_series(times, ids, num_neurons: int, duration_ms: float,
                       bin_ms: float = RATE_BIN_MS) -> np.ndarray:
    """(num_neurons, num_bins) の瞬時発火率 [Hz]。

    `times` は**窓の先頭を 0 とするローカル時刻** [ms]。窓の外のスパイクは落とす。
    """
    duration_ms = float(duration_ms)
    bin_ms = float(bin_ms)
    if duration_ms <= 0 or bin_ms <= 0:
        raise ValueError(f"duration_ms と bin_ms は正である必要があります "
                         f"({duration_ms}, {bin_ms})。")
    num_neurons = int(num_neurons)
    num_bins = max(int(duration_ms // bin_ms), 1)

    times = np.asarray(times, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    inside = (times >= 0.0) & (times < num_bins * bin_ms) & (ids >= 0) & (ids < num_neurons)
    bin_of = (times[inside] / bin_ms).astype(np.int64)
    counts = np.bincount(ids[inside] * num_bins + bin_of,
                         minlength=num_neurons * num_bins)
    return counts.reshape(num_neurons, num_bins) / (bin_ms / 1000.0)


@dataclass(frozen=True)
class StandardizedRates:
    """平均 0・ノルム 1 に直した発火率時系列と、それが定義できたニューロン。

    行列も平均も**これ 1 つから**導く。Pearson 相関は標準化したベクトルの内積なので、
    2 つの出口が別々に式を持たずに済む。
    """
    z: np.ndarray            # (num_defined, num_bins)
    defined: np.ndarray      # 長さ num_neurons の bool。分散 0 のニューロンが False

    @property
    def num_defined(self) -> int:
        return int(self.z.shape[0])

    @property
    def num_undefined(self) -> int:
        return int(self.defined.size - self.num_defined)


def standardize_rates(rates) -> StandardizedRates:
    """発火率時系列を平均 0・ノルム 1 に直す。分散 0 の行は落とす。"""
    rates = np.asarray(rates, dtype=np.float64)
    centred = rates - rates.mean(axis=1, keepdims=True)
    norm = np.sqrt(np.einsum("ij,ij->i", centred, centred))
    defined = norm > 0.0
    return StandardizedRates(z=centred[defined] / norm[defined, None], defined=defined)


def neuronal_correlation(times, ids, num_neurons: int, duration_ms: float,
                         bin_ms: float = RATE_BIN_MS) -> np.ndarray:
    """(N, N) のゼロラグ Pearson 相関行列。

    分散 0 のニューロンの行と列は `nan`。対角は (定義できたニューロンでは) 1。

    **N² のメモリを取る。** 平均値だけが欲しいなら `mean_neuronal_correlation()` を
    使うこと (行列を作らずに同じ値を出す)。
    """
    num_neurons = int(num_neurons)
    standardized = standardize_rates(
        firing_rate_series(times, ids, num_neurons, duration_ms, bin_ms))
    matrix = np.full((num_neurons, num_neurons), np.nan, dtype=np.float64)
    index = np.nonzero(standardized.defined)[0]
    if index.size:
        # 丸め誤差で |CC| が 1 をわずかに超えることがある。色の範囲を外れるだけだが、
        # 「相関が 1 を超えた」と読めてしまうので畳んでおく。
        block = np.clip(standardized.z @ standardized.z.T, -1.0, 1.0)
        np.fill_diagonal(block, 1.0)
        matrix[np.ix_(index, index)] = block
    return matrix


def mean_neuronal_correlation(times, ids, num_neurons: int, duration_ms: float,
                              bin_ms: float = RATE_BIN_MS) -> tuple[float, int]:
    """ネットワークの平均相関と、それを支えたペア数。

    **対角を除く**全ペアの平均 (論文の定義)。分散 0 のニューロンを含むペアは
    未定義なので数に入らない。

    行列は作らない: 標準化したベクトルの和を使うと
    `Σ_{i≠j} z_i·z_j = ‖Σ_i z_i‖² − n` なので O(N·T) で済む。
    `neuronal_correlation()` の非対角平均と厳密に一致する。

    Returns:
        `(平均相関, ペア数)`。定義できたニューロンが 1 本以下なら `(nan, 0)`。
    """
    standardized = standardize_rates(
        firing_rate_series(times, ids, int(num_neurons), duration_ms, bin_ms))
    n = standardized.num_defined
    if n < 2:
        return float("nan"), 0
    total = standardized.z.sum(axis=0)
    off_diagonal_sum = float(total @ total) - float(n)
    num_pairs = n * (n - 1)
    return off_diagonal_sum / num_pairs, int(num_pairs)


def mean_of_matrix(matrix) -> float:
    """相関行列の**対角を除いた**平均。`nan` のペアは除く。

    `mean_neuronal_correlation()` と同じ値を、既に行列を持っているときに出す口。
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    off_diagonal = ~np.eye(matrix.shape[0], dtype=bool)
    values = matrix[off_diagonal]
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")
