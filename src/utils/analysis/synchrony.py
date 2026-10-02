"""集団同期の粗さ —— 「1 回の同期イベントに何割が参加するか」。

ペア相関がなぜその値になるかを説明する量。実測では **平均相関 ≒ 参加率** で、
両者はほぼ 1 対 1 に動く。相関だけ見ていると「同期構造が変わった」のか
「発火率が落ちて参加できなくなった」のかが区別できない。

## 参加率には発火率で決まる上限がある

同期イベントが `f_event` [Hz] で起きていても、1 本のニューロンが `f_neuron` [Hz] でしか
撃たないなら、参加できるのは最大でも `f_neuron / f_event` の割合。これが
`participation_ceiling`。**参加率が低いことがそのまま「同期が弱い」を意味しない** ——
発火率が足りなくて参加できないだけかもしれない。

上限で割った `synchronous_fraction = participation / ceiling` は、式を整理すると
**発火のうち同期イベントに乗っている割合** (イベント中の発火 / 全発火) に等しい。
散発的な発火が増えればここが下がる。

    参加率が下がり、同期割合も下がった   -> 同期構造が壊れた
    参加率が下がり、同期割合は保たれた   -> 発火率が落ちて参加できなくなっただけ

## ビン幅は固定

**平均発火率の逆数を使ってはいけない。** 1 ビンあたり平均 1 発になる幅だと、ほぼ全ての
ビンがイベント判定になり (実測で 99%)、損傷の前後で参加率が動かなくなる。率の変化を
除きたいなら上の `synchronous_fraction` を見ること —— ビン幅で正規化すると、率の変化も
同期構造の変化も**両方**消える。

**相関とはビン幅が違う** (相関は 50 ms、こちらは 100 ms)。測っているものが違うため:
相関は「ペアが揃うか」なので 30〜50 ms で偶然との分離が最大になるが、こちらは
「集団の 20% が同時に撃つか」なので、2.5 Hz では 30 ms の窓に 51 本を詰められず
イベントがほとんど検出されない (実測でイベント頻度が 10 ms で 0.2 Hz、50 ms で 3.5 Hz、
100 ms で 3.9 Hz と頭打ち)。**「20% 以上が参加するイベント」は 50-100 ms の現象。**
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.analysis.correlation import firing_rate_series

# 同期イベントとみなす、同じビンで発火したニューロンの割合。
ACTIVE_FRACTION = 0.2
# イベント検出のビン幅 [ms]。**相関の `RATE_BIN_MS` とは別**(上の docstring)。
EVENT_BIN_MS = 100.0


@dataclass(frozen=True)
class PopulationEvents:
    """同期イベントの粗さ。全部 1 記録窓ぶん。"""
    bin_ms: float
    neuron_rate_hz: float        # 1 ニューロンが「ビンのどれかで撃つ」頻度
    event_rate_hz: float         # 同期イベントの頻度
    participation: float         # イベント時に撃っているニューロンの割合
    participation_ceiling: float  # 発火率で決まる参加率の上限 (f_neuron / f_event)
    # 発火のうち同期イベントに乗っている割合 (= participation / ceiling)
    synchronous_fraction: float
    active_fraction_median: float  # 全ビンでの活動割合の中央値 (閾値に依らない粗さ)


def population_events(times, ids, num_neurons: int, duration_ms: float,
                      bin_ms: float = EVENT_BIN_MS,
                      active_fraction: float = ACTIVE_FRACTION) -> PopulationEvents:
    """同期イベントの頻度と参加率。

    `times` は**窓の先頭を 0 とするローカル時刻** [ms]。イベントは「同じビンで
    `active_fraction` 以上のニューロンが撃った」ビン。

    ニューロンが 1 本も撃たない窓では全項目 `nan` (`event_rate_hz` は 0)。
    """
    num_neurons = int(num_neurons)
    fired = firing_rate_series(times, ids, num_neurons, duration_ms, bin_ms) > 0
    seconds = bin_ms / 1000.0
    active = fired.mean(axis=0)
    event = active >= float(active_fraction)

    neuron_rate = float(fired.mean()) / seconds
    event_rate = float(event.mean()) / seconds
    if not event.any() or event_rate <= 0.0:
        return PopulationEvents(
            bin_ms=float(bin_ms), neuron_rate_hz=neuron_rate, event_rate_hz=event_rate,
            participation=float("nan"), participation_ceiling=float("nan"),
            synchronous_fraction=float("nan"),
            active_fraction_median=float(np.median(active)))

    participation = float(active[event].mean())
    # 上限は 1 を超えない。発火率がイベント率より高ければ毎回参加できる。
    ceiling = min(neuron_rate / event_rate, 1.0)
    return PopulationEvents(
        bin_ms=float(bin_ms), neuron_rate_hz=neuron_rate, event_rate_hz=event_rate,
        participation=participation, participation_ceiling=ceiling,
        synchronous_fraction=participation / ceiling if ceiling > 0 else float("nan"),
        active_fraction_median=float(np.median(active)))


def event_columns(times, ids, num_neurons: int, duration_ms: float,
                  bin_ms: float = EVENT_BIN_MS) -> dict:
    """`metrics.csv` に入れる列。`population_events()` の結果を `event_*` の名前で返す。"""
    events = population_events(times, ids, num_neurons, duration_ms, bin_ms)
    return {
        "event_bin_ms": events.bin_ms,
        "event_rate_hz": events.event_rate_hz,
        "event_participation": events.participation,
        "event_participation_ceiling": events.participation_ceiling,
        "synchronous_fraction": events.synchronous_fraction,
        "active_fraction_median": events.active_fraction_median,
    }
