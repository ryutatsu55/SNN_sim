"""ニューロン間相関 (Watanabe et al. 2026 式 1) の性質を固定する。

縛るのは 4 つ:

1. **定義が `np.corrcoef` と一致すること。** 式を書き下しているので、標準実装との
   一致が唯一の外部基準。
2. **行列の非対角平均と、行列を作らない閉じた形が厳密に一致すること。** 指標 (metrics)
   は後者、図は前者を使うので、ずれると「図のタイトルと CSV の値が違う」が起きる。
3. **分散 0 のニューロンが `nan` であり、平均から除かれること。** 0 に畳むと損傷で
   黙ったニューロンが平均を押し下げ、含めると平均が押し上げられる。どちらも
   「相関が変わった」と読めてしまうので、未定義は未定義のまま扱う。
4. **ビン幅が値を決めること。** 既定の 30 ms は「偶然との分離が最大になる幅」として
   実測で選んだ値であって、任意に変えてよい表示パラメータではない。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.analysis.correlation import (RATE_BIN_MS, firing_rate_series,
                                            mean_neuronal_correlation, mean_of_matrix,
                                            neuronal_correlation)

DURATION_MS = 20000.0
NUM_NEURONS = 12


def _spikes(seed: int = 0, silent: tuple[int, ...] = ()):
    """共通の駆動を持つ前半と、独立に撃つ後半。`silent` のニューロンは 1 発も撃たない。"""
    rng = np.random.default_rng(seed)
    drive = np.sort(rng.uniform(0.0, DURATION_MS, 400))
    times, ids = [], []
    for neuron in range(NUM_NEURONS):
        if neuron in silent:
            continue
        if neuron < NUM_NEURONS // 2:
            t = drive + rng.normal(0.0, 5.0, drive.size)
        else:
            t = rng.uniform(0.0, DURATION_MS, 300)
        t = np.sort(t[(t >= 0.0) & (t < DURATION_MS)])
        times.append(t)
        ids.append(np.full(t.size, neuron))
    order = np.argsort(np.concatenate(times))
    return np.concatenate(times)[order], np.concatenate(ids)[order]


def test_matches_numpy_corrcoef():
    """式 1 は Pearson 相関そのもの。標準実装と機械精度で一致すること。"""
    times, ids = _spikes()
    matrix = neuronal_correlation(times, ids, NUM_NEURONS, DURATION_MS)
    rates = firing_rate_series(times, ids, NUM_NEURONS, DURATION_MS)
    assert np.allclose(matrix, np.corrcoef(rates), atol=1e-12)


def test_closed_form_mean_equals_matrix_mean():
    """行列を作らない平均が、非対角平均と厳密に一致すること。

    指標は前者、図のタイトルは後者を使う。ずれると同じ窓に 2 つの値が出る。
    """
    times, ids = _spikes()
    matrix = neuronal_correlation(times, ids, NUM_NEURONS, DURATION_MS)
    mean, pairs = mean_neuronal_correlation(times, ids, NUM_NEURONS, DURATION_MS)
    assert pairs == NUM_NEURONS * (NUM_NEURONS - 1)
    assert mean == pytest.approx(mean_of_matrix(matrix), abs=1e-12)


def test_shared_drive_is_more_correlated_than_independent_firing():
    """共通入力を持つ群の相関が、独立に撃つ群より高いこと (指標の向きの確認)。"""
    times, ids = _spikes()
    matrix = neuronal_correlation(times, ids, NUM_NEURONS, DURATION_MS)
    half = NUM_NEURONS // 2
    shared = matrix[:half, :half][~np.eye(half, dtype=bool)]
    independent = matrix[half:, half:][~np.eye(NUM_NEURONS - half, dtype=bool)]
    assert shared.mean() > 0.5
    assert abs(independent.mean()) < 0.1


def test_silent_neurons_are_undefined_not_zero():
    """1 発も撃たないニューロンは `nan`。平均にもペア数にも入らないこと。"""
    silent = (3, 9)
    times, ids = _spikes(silent=silent)
    matrix = neuronal_correlation(times, ids, NUM_NEURONS, DURATION_MS)
    for neuron in silent:
        assert np.all(np.isnan(matrix[neuron]))
        assert np.all(np.isnan(matrix[:, neuron]))

    active = NUM_NEURONS - len(silent)
    mean, pairs = mean_neuronal_correlation(times, ids, NUM_NEURONS, DURATION_MS)
    assert pairs == active * (active - 1)
    assert mean == pytest.approx(mean_of_matrix(matrix), abs=1e-12)
    assert np.isfinite(mean)


def test_all_silent_gives_nan_and_no_pairs():
    """誰も撃たない窓は「相関 0」ではなく「測れない」。"""
    mean, pairs = mean_neuronal_correlation(np.array([]), np.array([], dtype=int),
                                            NUM_NEURONS, DURATION_MS)
    assert pairs == 0
    assert np.isnan(mean)


def test_bin_width_changes_the_value():
    """ビン幅は値を決めるパラメータ。条件をまたいで比べる間は既定から動かさないこと。"""
    times, ids = _spikes()
    coarse, _ = mean_neuronal_correlation(times, ids, NUM_NEURONS, DURATION_MS,
                                          bin_ms=RATE_BIN_MS * 20)
    fine, _ = mean_neuronal_correlation(times, ids, NUM_NEURONS, DURATION_MS,
                                        bin_ms=RATE_BIN_MS)
    assert coarse != pytest.approx(fine, abs=1e-6)


def test_rate_series_drops_spikes_outside_the_window():
    """窓の外のスパイクは数えないこと (窓の先頭を 0 とするローカル時刻が前提)。"""
    span = 600.0                       # ビン幅で割り切れる長さ
    times = np.array([-1.0, 0.0, span - 1.0, span + 1.0])
    ids = np.zeros(4, dtype=int)
    rates = firing_rate_series(times, ids, 1, span, RATE_BIN_MS)
    assert rates.shape == (1, int(span // RATE_BIN_MS))
    assert rates.sum() * (RATE_BIN_MS / 1000.0) == pytest.approx(2.0)


def test_rate_series_drops_the_remainder_of_the_window():
    """ビン幅で割り切れない端数は落ちること。

    窓の末尾が最大 1 ビンぶん捨てられる。`record_window_ms` がビン幅の整数倍でない
    設定では、この端数のスパイクが相関に入らない。
    """
    span = 100.0
    rates = firing_rate_series(np.array([95.0]), np.array([0]), 1, span, 30.0)
    assert rates.shape == (1, 3)       # 0-90 ms の 3 ビンだけ。90-100 ms は落ちる
    assert rates.sum() == 0.0
