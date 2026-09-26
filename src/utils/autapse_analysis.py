"""単一細胞比較の時間区間・発火指標。PyGeNNには依存しない。"""

import numpy as np


def steps_for(value, dt, name="time"):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name}: 有限の非負時間と正のdtが必要です")
    steps = round(value / dt)
    if not np.isclose(steps * dt, value, atol=1e-8, rtol=0):
        raise ValueError(f"{name}: dtの整数倍を指定してください")
    return steps


def spike_metrics(times, start, stop, end, tail, burst_isi=10.0, burst_min=3):
    """スパイクは更新開始時刻。全区間を左閉・右開として数える。"""
    times = np.asarray(times, dtype=float)
    if (
        not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0)
        or np.any(times < 0)
        or np.any(times >= end)
    ):
        raise ValueError("スパイク時刻は観測内で有限かつ狭義単調増加が必要です")
    if not (0 <= start < stop < end and 0 < tail <= end - stop):
        raise ValueError("刺激・観測・末尾窓の区間が不正です")
    if burst_isi <= 0 or burst_min < 2:
        raise ValueError("バースト閾値は正、最小発火数は2以上が必要です")
    during = times[(times >= start) & (times < stop)]
    post = times[(times >= stop) & (times < end)]

    isi = np.diff(post)
    tails = post[post >= end - tail]
    groups = np.split(post, np.flatnonzero(isi > burst_isi) + 1)

    if tails.size:
        status = "active_at_observation_tail"
    elif post.size:
        status = "post_spikes_without_tail_activity"
    elif during.size:
        status = "no_post_spikes"
    else:
        status = "no_evoked_spikes"
    second_half = during[during >= (start + stop) / 2]

    return {
        "pre_spikes": int(np.count_nonzero(times < start)),
        "during_spikes": int(during.size),
        "post_spikes": int(post.size),
        "during_rate_hz": float(during.size * 1000 / (stop - start)),
        "late_stimulus_rate_hz": float(second_half.size * 2000 / (stop - start)),
        "post_rate_hz": float(post.size * 1000 / (end - stop)),
        "first_post_latency_ms": float(post[0] - stop) if post.size else None,
        "last_post_latency_ms": float(post[-1] - stop) if post.size else None,
        "post_isi_mean_ms": float(isi.mean()) if isi.size else None,
        "post_isi_cv": float(isi.std() / isi.mean()) if isi.size >= 2 else None,
        "post_bursts": sum(len(group) >= burst_min for group in groups),
        "tail_spikes": int(tails.size),
        "status": status,
        "terminal_silence_ms": float(end - times[-1]) if times.size else float(end),
    }


def threshold_bracket(amplitudes, fired):
    """単調な発火有無から、未発火下端と発火上端を返す。"""
    amplitudes = np.asarray(amplitudes, dtype=float)
    fired = np.asarray(fired, dtype=bool)
    if (
        amplitudes.ndim != 1
        or fired.shape != amplitudes.shape
        or amplitudes.size < 2
        or amplitudes[0] != 0
        or not np.all(np.isfinite(amplitudes))
        or np.any(np.diff(amplitudes) <= 0)
    ):
        raise ValueError("閾値探索は0から始まる有限の昇順振幅列が必要です")
    if fired[0]:
        raise ValueError("無入力で発火しています。閾値正規化を停止します")
    hits = np.flatnonzero(fired)
    if not hits.size:
        raise ValueError("探索範囲内で発火閾値を検出できません")
    first = hits[0]
    if not np.all(fired[first:]):
        raise ValueError("発火有無が単調ではありません。探索範囲の再検討が必要です")
    return {
        "lower_nonfiring": float(amplitudes[first - 1]),
        "upper_firing": float(amplitudes[first]),
    }


def matched_dendritic_delay(requested_ms, dt_ms, reference_dt_ms=None):
    """GeNNの伝達更新1dtを含む到着時間を、基準実験と一致させる。"""
    offset = dt_ms if reference_dt_ms is None else reference_dt_ms
    delay = requested_ms + offset - dt_ms

    return steps_for(delay, dt_ms, "arrival-matched dendritic delay") * dt_ms
