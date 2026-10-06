"""実効結合 —— 一般化移動エントロピー (GTE) と、その損傷前比 (RTE)。

Watanabe et al. (2026) *Communications Engineering* **5**:150 の式 (2)(3) をそのまま
実装したもの。元は Stetter et al. (2012) の GTE。

## TE の定義

ニューロン Y から X へ渡る情報量を、2 値化したスパイク列 x_t, y_t から

    TE(Y→X) = Σ P(x_{t+1}, x_t^(k), y_{t+1}^(k))
              · log2[ P(x_{t+1} | x_t^(k), y_{t+1}^(k)) / P(x_{t+1} | x_t^(k)) ]

    x_t^(k)     = (x_t, x_{t-1}, …, x_{t-k+1})        X 自身の過去 k フレーム
    y_{t+1}^(k) = (y_{t+1}, y_t, …, y_{t-k+2})        Y の**同じフレームを含む** k フレーム

で測る。Y が**同じフレーム**を含むのが GTE の「瞬時の項」で、フレーム (50 ms) が
シナプス遅延よりずっと長いため、1 フレームの中で起きた伝達を拾うためにある。
**この項は向きを区別しない** —— 同じフレームで一緒に撃つだけのペアは、Y→X と X→Y の
両方に同じだけの TE を持つ。向きを言えるのは、1 フレーム以上遅れた成分だけ。

**大域状態での条件づけはしない。** Stetter の原法は集団活動の高いフレームを除くが、
Watanabe の Methods は式 (2) だけを書いているので、それに合わせる。集団バーストに
共通して参加するペアは、直接の伝達が無くても TE を持ちうる。

## 2 値化

フレーム `TE_BIN_MS` の中に 1 発以上あれば 1。論文は推定発火率 r(t) に Schmitt トリガ
(上 1 Hz / 下 0.5 Hz) を掛けるが、スパイク数から作った r(t) は 0 か 20 Hz 以上なので
同じものになる。

## z スコアと RTE

z スコアは**窓ごと**に、対角を除いた全ペアの TE で標準化したもの。全結合が一律に
半減しても z は変わらないので、水準の変化は RTE が受け持つ:

    C   = 損傷前の窓で z > `RELEVANT_Z` だった結合の集合 (固定)
    RTE = (1/|C|) Σ_{(Y,X)∈C} TE(Y→X) / TE₀(Y→X)

**C は損傷前に固定する**ので、損傷後に生まれた結合は RTE に入らない。RTE は
「元の経路がどれだけ残っているか」で、代替経路の形成はこの指標には現れない。

行列の向きは結合の COO と同じ **`te[source, target]`** (行 = 送る側、列 = 受ける側)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.analysis.correlation import firing_rate_series

# フレームの幅 [ms]。論文のイメージングのフレーム (20 fps) と同じ。
TE_BIN_MS = 50.0
# マルコフ次数 k。論文と同じ。
MARKOV_ORDER = 2
# 「機能的に意味のある結合」とみなす z スコアの下限。論文と同じ。
RELEVANT_Z = 1.0


def binarize_spikes(times, ids, num_neurons: int, duration_ms: float,
                    bin_ms: float = TE_BIN_MS) -> np.ndarray:
    """(num_neurons, num_frames) の 0/1 列。フレームの中に 1 発以上あれば 1。

    `times` は**窓の先頭を 0 とするローカル時刻** [ms]。
    """
    return firing_rate_series(times, ids, num_neurons, duration_ms, bin_ms) > 0


def _history_code(binary: np.ndarray, start: int, stop: int, order: int) -> np.ndarray:
    """各フレーム t ∈ [start, stop) の (b_t, b_{t-1}, …, b_{t-order+1}) を整数に符号化する。"""
    code = np.zeros((binary.shape[0], stop - start), dtype=np.int64)
    for lag in range(order):
        code = code * 2 + binary[:, start - lag:stop - lag]
    return code


def transfer_entropy_matrix(binary: np.ndarray, order: int = MARKOV_ORDER) -> np.ndarray:
    """全ペアの TE [bit] を (N, N) で返す。`te[source, target]`。対角は `nan`。

    1 回の行列積で全ペアの同時頻度を数える: 受け手 X の状態
    (x_{t+1}, x_t^(k)) と送り手 Y の状態 y_{t+1}^(k) をそれぞれ one-hot にし、
    時間方向に内積を取る。メモリは O(N² · 2^(2k+1))。
    """
    binary = np.asarray(binary, dtype=np.int64)
    num_neurons, num_frames = binary.shape
    if num_frames <= order:
        raise ValueError(f"フレーム数 {num_frames} がマルコフ次数 {order} 以下です。")

    # 使うフレーム: x_{t+1} を予測する t+1 ∈ [order, num_frames)
    start, stop = order, num_frames
    x_next = binary[:, start:stop]
    x_past = _history_code(binary, start - 1, stop - 1, order)   # x_t^(k)
    y_hist = _history_code(binary, start, stop, order)           # y_{t+1}^(k)

    num_past = 2 ** order
    target_state = x_next * num_past + x_past                    # (N, T) ∈ [0, 2·2^k)
    num_target_states = 2 * num_past

    def _one_hot(code, size):
        out = np.zeros((code.shape[0], size, code.shape[1]), dtype=np.float32)
        np.put_along_axis(out, code[:, None, :], 1.0, axis=1)
        return out.reshape(code.shape[0] * size, code.shape[1])

    # counts[target, (a, b), source, c]  a = x_{t+1}, b = x_t^(k), c = y_{t+1}^(k)
    # float32 の積和は 2^24 フレームまで整数として正確。
    counts = _one_hot(target_state, num_target_states) @ _one_hot(y_hist, num_past).T
    counts = counts.astype(np.float64).reshape(num_neurons, 2, num_past, num_neurons, num_past)

    total = float(stop - start)
    p_abc = counts / total                               # P(a, b, c)
    p_bc = p_abc.sum(axis=1, keepdims=True)              # P(b, c)
    p_ab = p_abc.sum(axis=4, keepdims=True)              # P(a, b)
    p_b = p_ab.sum(axis=1, keepdims=True)                # P(b)

    # TE = Σ P(a,b,c) log2[ P(a,b,c) P(b) / (P(b,c) P(a,b)) ]。P(a,b,c) = 0 の項は 0。
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (p_abc * p_b) / (p_bc * p_ab)
        terms = np.where(p_abc > 0, p_abc * np.log2(ratio), 0.0)
    te_target_source = terms.sum(axis=(1, 2, 4))         # (target, source)

    te = te_target_source.T.copy()                       # (source, target)
    # 丸め誤差で負になる分を畳む (TE は条件付き相互情報量なので非負)。
    np.maximum(te, 0.0, out=te)
    np.fill_diagonal(te, np.nan)
    return te


def transfer_entropy(times, ids, num_neurons: int, duration_ms: float,
                     bin_ms: float = TE_BIN_MS, order: int = MARKOV_ORDER) -> np.ndarray:
    """スパイク列から直接 `transfer_entropy_matrix()` を求める。"""
    return transfer_entropy_matrix(
        binarize_spikes(times, ids, num_neurons, duration_ms, bin_ms), order)


def zscore_offdiagonal(te: np.ndarray) -> np.ndarray:
    """対角を除いた全ペアで標準化した TE。対角は `nan` のまま。

    全ペアが同じ値 (分散 0) なら全部 `nan`。
    """
    te = np.asarray(te, dtype=np.float64)
    values = te[np.isfinite(te)]
    sd = float(values.std()) if values.size else 0.0
    if sd <= 0.0:
        return np.full_like(te, np.nan)
    return (te - float(values.mean())) / sd


def relevant_connections(te_before: np.ndarray, threshold: float = RELEVANT_Z) -> np.ndarray:
    """損傷前の TE で z > `threshold` の結合。(N, N) の bool、`[source, target]`。"""
    z = zscore_offdiagonal(te_before)
    with np.errstate(invalid="ignore"):
        return np.isfinite(z) & (z > threshold)


@dataclass(frozen=True)
class RelativeTE:
    """1 つの窓の RTE。"""
    rte: float              # C 上の TE / TE₀ の平均
    num_connections: int    # |C|
    mean_te: float          # 窓の全ペアの TE の平均 [bit] (C に限らない水準)


def relative_transfer_entropy(te: np.ndarray, te_before: np.ndarray,
                              relevant: np.ndarray) -> RelativeTE:
    """`relevant` (損傷前に固定した C) の上で、TE を損傷前の値で割った比の平均。

    **C は `te_before` から作ったものを渡すこと。** 窓ごとに C を作り直すと、
    損傷後に生まれた結合が混ざって「元の経路が残っているか」を測らなくなる。
    """
    te = np.asarray(te, dtype=np.float64)
    te_before = np.asarray(te_before, dtype=np.float64)
    relevant = np.asarray(relevant, dtype=bool)
    if te.shape != te_before.shape or te.shape != relevant.shape:
        raise ValueError(f"形が違います: te {te.shape}, te_before {te_before.shape},"
                         f" relevant {relevant.shape}")
    mask = relevant & (te_before > 0)
    finite = te[np.isfinite(te)]
    mean_te = float(finite.mean()) if finite.size else float("nan")
    if not mask.any():
        return RelativeTE(rte=float("nan"), num_connections=0, mean_te=mean_te)
    return RelativeTE(rte=float(np.mean(te[mask] / te_before[mask])),
                      num_connections=int(mask.sum()), mean_te=mean_te)
