"""同期イベントの粗さ (`analysis/synchrony.py`) の性質を固定する。

この指標が存在する理由は「平均相関が落ちたとき、同期構造が壊れたのか発火率が落ちた
だけなのかを分ける」こと。だから縛るのは 2 つ:

1. **参加率には発火率で決まる上限がある** —— イベントより遅く撃つニューロンは、
   全部のイベントには参加できない。
2. **その上限で割った値 (= 発火のうち同期イベントに乗っている割合) が、発火率だけの
   変化では動かない**こと。ここが動くと「構造が変わった」の判定ができなくなる。

ビン幅を平均発火率の逆数にしない理由も固定する (ほぼ全ビンがイベント判定になる)。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.analysis.synchrony import (ACTIVE_FRACTION, event_columns,
                                          population_events)

N = 20
DURATION_MS = 60000.0
BIN_MS = 100.0


def _events(event_times, participants, rng):
    """`event_times` の各時刻で `participants` 本だけが撃つスパイク列を作る。"""
    times, ids = [], []
    for t in event_times:
        who = rng.choice(N, size=participants, replace=False)
        times.append(np.full(participants, t) + rng.uniform(0, BIN_MS * 0.5, participants))
        ids.append(who)
    return np.concatenate(times), np.concatenate(ids)


def test_participation_matches_the_constructed_fraction():
    """作った参加率がそのまま出ること。"""
    rng = np.random.default_rng(0)
    event_times = np.arange(0.0, DURATION_MS, 1000.0)
    times, ids = _events(event_times, participants=N // 2, rng=rng)
    result = population_events(times, ids, N, DURATION_MS, BIN_MS)
    assert result.participation == pytest.approx(0.5, abs=0.02)
    assert result.event_rate_hz == pytest.approx(1.0, abs=0.05)


def test_ceiling_binds_when_neurons_fire_slower_than_events():
    """イベントより遅く撃つニューロンは、全イベントには参加できないこと。

    1 本のニューロンは 1 回のイベントにしか参加しないので、参加率は上限に張り付く。
    """
    rng = np.random.default_rng(1)
    event_times = np.arange(0.0, DURATION_MS, 500.0)      # 2 Hz のイベント
    times, ids = _events(event_times, participants=N // 4, rng=rng)
    result = population_events(times, ids, N, DURATION_MS, BIN_MS)
    assert result.participation_ceiling < 1.0
    assert result.participation == pytest.approx(result.participation_ceiling, rel=0.15)
    assert result.synchronous_fraction == pytest.approx(1.0, rel=0.15)


def test_synchronous_fraction_is_one_without_background_firing():
    """全ての発火がイベント中なら同期割合は 1。

    発火率を落としても、イベントの外で撃たない限りここは動かない —— これが
    「発火率の変化では動かない」という性質の中身。
    """
    rng = np.random.default_rng(2)
    fast = _events(np.arange(0.0, DURATION_MS, 500.0), N // 2, rng)
    slow = _events(np.arange(0.0, DURATION_MS, 1000.0), N // 2, rng)
    a = population_events(*fast, N, DURATION_MS, BIN_MS)
    b = population_events(*slow, N, DURATION_MS, BIN_MS)
    assert b.neuron_rate_hz < a.neuron_rate_hz          # 発火率は落ちている
    assert a.synchronous_fraction == pytest.approx(1.0, abs=0.02)
    assert b.synchronous_fraction == pytest.approx(1.0, abs=0.02)


def test_synchronous_fraction_drops_with_scattered_firing():
    """イベントの外に散発的な発火を足すと同期割合が下がること (構造の変化)。"""
    rng = np.random.default_rng(3)
    event_times = np.arange(0.0, DURATION_MS, 500.0)
    times, ids = _events(event_times, N // 2, rng)
    clean = population_events(times, ids, N, DURATION_MS, BIN_MS)
    # 同じイベントに、イベント間へばら撒いた発火を足す
    extra_t = rng.uniform(0.0, DURATION_MS, times.size)
    extra_i = rng.integers(0, N, times.size)
    noisy = population_events(np.concatenate([times, extra_t]),
                              np.concatenate([ids, extra_i]), N, DURATION_MS, BIN_MS)
    assert noisy.neuron_rate_hz > clean.neuron_rate_hz   # 発火率は上がっている
    assert noisy.synchronous_fraction < clean.synchronous_fraction


def test_inverse_rate_bin_makes_every_bin_an_event():
    """**ビン幅を平均発火率の逆数にしない理由。** ほぼ全ビンがイベントになる。

    1 ビンあたり平均 1 発になる幅では、閾値 20% をほぼ常に超えてしまい、
    参加率が条件の違いに反応しなくなる。
    """
    rng = np.random.default_rng(4)
    times, ids = _events(np.arange(0.0, DURATION_MS, 500.0), N // 2, rng)
    inverse_rate_ms = 1000.0 / (times.size / N / (DURATION_MS / 1000.0))
    coarse = population_events(times, ids, N, DURATION_MS, inverse_rate_ms)
    fine = population_events(times, ids, N, DURATION_MS, BIN_MS)
    # 粗いビンではイベント率が「ビンの数」そのものに張り付く
    assert coarse.event_rate_hz == pytest.approx(1000.0 / inverse_rate_ms, rel=0.05)
    assert coarse.participation > fine.participation


def test_silent_window_is_undefined_not_zero():
    """1 本も撃たない窓は参加率 `nan` (「同期が無い」ではなく「測れない」)。"""
    result = population_events(np.array([]), np.array([], dtype=int), N, DURATION_MS)
    assert result.event_rate_hz == 0.0
    assert np.isnan(result.participation)
    assert np.isnan(result.synchronous_fraction)


def test_event_columns_names_are_stable():
    """`metrics.csv` の列名を固定する (再解析が列を変えないこと)。"""
    rng = np.random.default_rng(5)
    times, ids = _events(np.arange(0.0, DURATION_MS, 500.0), N // 2, rng)
    columns = event_columns(times, ids, N, DURATION_MS)
    assert set(columns) == {"event_bin_ms", "event_rate_hz", "event_participation",
                            "event_participation_ceiling", "synchronous_fraction",
                            "active_fraction_median"}
    assert 0.0 < ACTIVE_FRACTION < 1.0


# --- 群ごとの同期 (active_fraction_series / golomb_chi) ------------------------------

from src.utils.analysis.synchrony import (active_fraction_series, chi_floor,  # noqa: E402
                                          golomb_chi)


def test_active_fraction_counts_only_members():
    times = np.array([10.0, 20.0, 30.0, 150.0])
    ids = np.array([0, 1, 5, 1])
    # members = {0, 1, 2, 3}: ビン 0 で 0 と 1 (5 は群の外)、ビン 1 で 1
    fraction = active_fraction_series(times, ids, [0, 1, 2, 3], 200.0, bin_ms=100.0)
    np.testing.assert_allclose(fraction, [0.5, 0.25])


def test_chi_is_one_when_all_fire_together_and_low_when_independent():
    rng = np.random.default_rng(0)
    members = np.arange(N)
    # 全員が同じ時刻に撃つ
    event_times = np.sort(rng.uniform(0, DURATION_MS, 200))
    times = np.repeat(event_times, N)
    ids = np.tile(members, event_times.size)
    assert golomb_chi(times, ids, members, DURATION_MS) == pytest.approx(1.0)
    # 独立なポアソン
    count = 200 * N
    times = rng.uniform(0, DURATION_MS, count)
    ids = rng.integers(0, N, count)
    chi = golomb_chi(times, ids, members, DURATION_MS)
    assert chi == pytest.approx(chi_floor(N), rel=0.3)


def test_chi_ignores_neurons_outside_group():
    rng = np.random.default_rng(1)
    event_times = np.sort(rng.uniform(0, DURATION_MS, 100))
    times = np.repeat(event_times, 5)
    ids = np.tile(np.arange(5), event_times.size)
    # 群の外 (ID 10..19) にばらばらの発火を足しても、群 0..4 の χ は 1 のまま
    noise_t = rng.uniform(0, DURATION_MS, 2000)
    noise_id = rng.integers(10, 20, 2000)
    chi = golomb_chi(np.concatenate([times, noise_t]), np.concatenate([ids, noise_id]),
                     np.arange(5), DURATION_MS)
    assert chi == pytest.approx(1.0)


def test_chi_nan_when_group_silent():
    assert np.isnan(golomb_chi(np.array([5.0]), np.array([9]), [0, 1], DURATION_MS))
