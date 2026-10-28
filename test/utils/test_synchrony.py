"""同期イベントの粗さ (`analysis/synchrony.py`) の性質を固定する。

この指標が存在する理由は「平均相関が落ちたとき、同期構造が壊れたのか発火率が落ちた
だけなのかを分ける」こと。だから縛るのは 2 つ:

1. **参加率には発火率で決まる上限がある** —— イベントより遅く撃つニューロンは、
   全部のイベントには参加できない。
2. **その上限で割った値 (= 発火のうち同期イベントに乗っている割合) が、発火率だけの
   変化では動かない**こと。ここが動くと「構造が変わった」の判定ができなくなる。

ビン幅を平均発火率の逆数にしない理由も固定する (ほぼ全ビンがイベント判定になる)。
黙った窓では粗さを 0 ではなく未定義 (`nan`) とする。

同じモジュールの他の指標も固定する:

- **群ごとの同期 (Golomb の χ)** は、全員が同時に撃てば 1、独立なら小さい。群の外のニューロンを数えず、
  群が黙っていれば `nan`
- **ネットワークバースト**は連続するイベントビンを 1 回にまとめ、参加率はバースト全体で数える。
  バーストの外の発火も数える
- **全体イベント**は、独立な発火からは見つからず、埋め込んだ深くて広いイベントは見つけ、
  1 モジュールだけのイベントは局所として分ける。結果は決定的で、黙った窓では 0 件
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
    assert set(columns) == {"event_rate_hz", "event_participation",
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


# --- ネットワークバースト (network_bursts) -----------------------------------------

from src.utils.analysis.synchrony import burst_columns, network_bursts  # noqa: E402


def test_consecutive_event_bins_are_one_burst():
    # 0.3 s 続くバースト (100 ms ビン 3 つ) を 10 s ごとに 6 回。全員が各ビンで撃つ。
    rng = np.random.default_rng(0)
    times, ids = [], []
    for start in np.arange(6) * 10000.0 + 1000.0:
        for offset in (10.0, 110.0, 210.0):
            times += list(start + offset + rng.uniform(0, 50, N))
            ids += list(range(N))
    b = network_bursts(np.array(times), np.array(ids), N, DURATION_MS, bin_ms=BIN_MS)
    assert b.num_bursts == 6
    assert b.burst_rate_hz == pytest.approx(6 / (DURATION_MS / 1000))
    assert b.width_ms == pytest.approx(300.0)
    assert b.participation == pytest.approx(1.0)
    assert b.outside_fraction == pytest.approx(0.0)


def test_participation_counts_neurons_over_the_whole_burst():
    # 1 回のバーストのうち、前半のビンで半分、後半のビンで残り半分が撃つ -> 参加率 1.0
    half = N // 2
    times = np.concatenate([np.full(half, 1010.0), np.full(N - half, 1110.0)])
    ids = np.arange(N)
    b = network_bursts(times, ids, N, DURATION_MS, bin_ms=BIN_MS)
    assert b.num_bursts == 1
    assert b.width_ms == pytest.approx(200.0)
    assert b.participation == pytest.approx(1.0)


def test_spikes_outside_bursts_are_counted():
    # バースト 1 回 (全員) + 散発 1 発 (別の時刻、閾値に届かない)
    times = np.concatenate([np.full(N, 1010.0), [5005.0]])
    ids = np.concatenate([np.arange(N), [0]])
    b = network_bursts(times, ids, N, DURATION_MS, bin_ms=BIN_MS)
    assert b.num_bursts == 1
    assert b.outside_fraction == pytest.approx(1 / (N + 1))


def test_no_burst_gives_zero_rate_and_nan_shape():
    cols = burst_columns(np.array([5.0]), np.array([0]), N, DURATION_MS)
    assert cols["burst_count"] == 0 and cols["burst_rate_hz"] == 0.0
    assert np.isnan(cols["burst_width_ms"]) and np.isnan(cols["burst_participation"])
    assert cols["burst_outside_fraction"] == pytest.approx(1.0)


from src.utils.analysis.synchrony import global_event_columns, global_events  # noqa: E402


def _poisson(rng, n, rate_hz, duration_ms):
    """独立な Poisson 発火 (n 本、ローカル時刻 ms)。"""
    counts = rng.poisson(rate_hz * duration_ms / 1000.0, n)
    ids = np.repeat(np.arange(n), counts)
    times = rng.uniform(0.0, duration_ms, ids.size)
    return times, ids


def test_independent_firing_has_no_global_events():
    """発火率が高くても、同時性が無ければ全体イベントは出ない (固定閾値の burst との違い)。"""
    rng = np.random.default_rng(1)
    times, ids = _poisson(rng, 64, 5.0, 60000.0)
    ev = global_events(times, ids, 64, 60000.0)
    assert ev.global_rate_hz < 0.05
    assert ev.very_deep_rate_hz == 0.0


def test_planted_global_events_are_found_deep_and_wide():
    """全員が揃って撃つイベントを 0.5 Hz で埋めると、その頻度で深く広く見つかる。"""
    rng = np.random.default_rng(2)
    n, dur = 64, 60000.0
    bg_t, bg_i = _poisson(rng, n, 0.5, dur)
    onsets = np.arange(1000.0, dur, 2000.0)
    ev_t = np.concatenate([o + rng.uniform(0, 30, n) for o in onsets])
    ev_i = np.tile(np.arange(n), onsets.size)
    groups = np.repeat(np.arange(4), n // 4)
    ev = global_events(np.concatenate([bg_t, ev_t]), np.concatenate([bg_i, ev_i]), n, dur, groups)
    assert ev.global_rate_hz == pytest.approx(onsets.size / 60.0, rel=0.1)
    assert ev.global_depth > 0.9
    assert ev.deep_group_fraction == pytest.approx(1.0)


def test_single_module_events_are_local_not_global():
    """1 モジュールだけが揃って撃つイベントは local に数え、全体には数えない。"""
    rng = np.random.default_rng(3)
    n, dur = 64, 60000.0
    bg_t, bg_i = _poisson(rng, n, 0.5, dur)
    onsets = np.arange(1000.0, dur, 2000.0)
    members = np.arange(16)              # モジュール 0 だけ
    ev_t = np.concatenate([o + rng.uniform(0, 30, members.size) for o in onsets])
    ev_i = np.tile(members, onsets.size)
    groups = np.repeat(np.arange(4), n // 4)
    ev = global_events(np.concatenate([bg_t, ev_t]), np.concatenate([bg_i, ev_i]), n, dur, groups)
    assert ev.local_rate_hz == pytest.approx(onsets.size / 60.0, rel=0.1)
    assert ev.global_rate_hz == 0.0
    # 全体で見れば 1/4 しか参加していないが、そのモジュールの中では全員が撃っている
    assert ev.local_group_depth > 0.9


def test_group_depth_separates_spread_from_depth_within_groups():
    """3/4 のモジュールが全員で撃つイベント: 全体の深さは約 3/4、加わった群の中では約 1。"""
    rng = np.random.default_rng(5)
    n, dur = 64, 60000.0
    bg_t, bg_i = _poisson(rng, n, 0.5, dur)
    onsets = np.arange(1000.0, dur, 2000.0)
    members = np.arange(48)              # モジュール 0-2
    ev_t = np.concatenate([o + rng.uniform(0, 30, members.size) for o in onsets])
    ev_i = np.tile(members, onsets.size)
    groups = np.repeat(np.arange(4), n // 4)
    ev = global_events(np.concatenate([bg_t, ev_t]), np.concatenate([bg_i, ev_i]), n, dur, groups)
    assert ev.global_rate_hz == pytest.approx(onsets.size / 60.0, rel=0.1)
    assert ev.global_depth == pytest.approx(0.75, abs=0.05)
    assert ev.global_group_depth > 0.9


def test_global_events_are_deterministic_and_silent_window_is_zero():
    rng = np.random.default_rng(4)
    times, ids = _poisson(rng, 32, 2.0, 30000.0)
    a = global_event_columns(times, ids, 32, 30000.0)
    b = global_event_columns(times, ids, 32, 30000.0)
    np.testing.assert_equal(a, b)       # nan どうしも等しいとみなす
    empty = global_events(np.array([]), np.array([], dtype=int), 32, 30000.0)
    assert empty.event_rate_hz == 0.0 and empty.global_rate_hz == 0.0
