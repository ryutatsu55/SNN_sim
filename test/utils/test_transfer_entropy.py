"""移動エントロピー (`analysis/transfer_entropy.py`) の性質を固定する。

1. **素朴な数え上げと一致する** —— 行列積による一括計算が式 (2) そのものであること
2. **遅れて駆動する向きにだけ TE が立つ** —— 向きを取り違えると実効結合の図が逆を指す
3. **同じフレームの項は向きを区別しない** —— GTE の性質。docstring の注意の裏付け
4. **RTE は C の上の比の平均で、C は損傷前に固定** —— 一律に半減すれば 0.5
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pytest

from src.utils.analysis.transfer_entropy import (RelativeTE, binarize_spikes,
                                                 relative_transfer_entropy,
                                                 relevant_connections,
                                                 transfer_entropy_matrix,
                                                 zscore_offdiagonal)


def _brute_te(y, x, k=2):
    """式 (2) をそのまま数える。`x[t+1]`、x の過去 k、y の同じフレームを含む過去 k。"""
    states = Counter()
    for t1 in range(k, len(x)):
        states[(x[t1], tuple(x[t1 - 1 - lag] for lag in range(k)),
                tuple(y[t1 - lag] for lag in range(k)))] += 1
    n = sum(states.values())
    p_bc, p_ab, p_b = defaultdict(float), defaultdict(float), defaultdict(float)
    for (a, b, c), count in states.items():
        p = count / n
        p_bc[(b, c)] += p
        p_ab[(a, b)] += p
        p_b[b] += p
    return sum((count / n) * np.log2((count / n) * p_b[b] / (p_bc[(b, c)] * p_ab[(a, b)]))
               for (a, b, c), count in states.items())


def _driven(seed=0, frames=4000):
    """0 が 1 を 1 フレーム遅れで、2 を同じフレームで駆動する。3 は独立。"""
    rng = np.random.default_rng(seed)
    binary = (rng.random((4, frames)) < 0.15).astype(np.int64)
    binary[1, 1:] |= binary[0, :-1] & (rng.random(frames - 1) < 0.7)
    binary[2] |= binary[0] & (rng.random(frames) < 0.7)
    return binary


def test_matches_brute_force():
    binary = _driven()
    te = transfer_entropy_matrix(binary)
    for source in range(4):
        for target in range(4):
            if source == target:
                assert np.isnan(te[source, target])
            else:
                assert te[source, target] == pytest.approx(
                    _brute_te(binary[source], binary[target]), abs=1e-12)


def test_lagged_drive_has_direction():
    te = transfer_entropy_matrix(_driven())
    # te[source, target]: 0 → 1 は 1 → 0 よりずっと大きい
    assert te[0, 1] > 10 * te[1, 0]
    # 独立なペアはほぼ 0
    assert te[0, 3] < 0.01 and te[3, 0] < 0.01


def test_same_frame_drive_is_symmetric():
    te = transfer_entropy_matrix(_driven())
    assert te[0, 2] > 0.05
    assert te[2, 0] == pytest.approx(te[0, 2], rel=0.3)


def test_binarize_counts_any_spike_in_frame():
    binary = binarize_spikes(np.array([10.0, 20.0, 60.0]), np.array([0, 0, 1]),
                             num_neurons=2, duration_ms=100.0, bin_ms=50.0)
    np.testing.assert_array_equal(binary, [[True, False], [False, True]])


def test_zscore_ignores_diagonal():
    te = np.array([[np.nan, 1.0, 2.0], [3.0, np.nan, 4.0], [5.0, 6.0, np.nan]])
    z = zscore_offdiagonal(te)
    assert np.isnan(np.diag(z)).all()
    off = z[~np.eye(3, dtype=bool)]
    assert off.mean() == pytest.approx(0.0) and off.std() == pytest.approx(1.0)


def test_rte_is_mean_ratio_on_fixed_set():
    rng = np.random.default_rng(1)
    before = rng.random((20, 20))
    np.fill_diagonal(before, np.nan)
    relevant = relevant_connections(before)
    assert 0 < relevant.sum() < 20 * 19

    halved = relative_transfer_entropy(before / 2, before, relevant)
    assert isinstance(halved, RelativeTE)
    assert halved.rte == pytest.approx(0.5)
    assert halved.num_connections == relevant.sum()
    # C の外だけが増えても RTE は動かない (新しい経路は数えない)
    grown = before.copy()
    grown[~relevant & np.isfinite(grown)] *= 3
    assert relative_transfer_entropy(grown, before, relevant).rte == pytest.approx(1.0)
