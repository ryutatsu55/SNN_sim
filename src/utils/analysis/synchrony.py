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


# ======================================================================================
# ネットワークバースト —— 連続するイベントビンを 1 回のバーストにまとめる
#
# `population_events()` はイベント**ビン**を数えるので、数百 ms 続くバーストを何回にも
# 数える。ここは `active_fraction` 以上のビンが**連続する区間**を 1 回とし、回数・幅・
# 1 回あたりの参加率を出す。疎で深いバースト (論文: 約 0.1 Hz、ほぼ全員参加) を
# 目標にするときの物差し。
# ======================================================================================


@dataclass(frozen=True)
class NetworkBursts:
    """ネットワークバーストの要約。全部 1 記録窓ぶん。"""
    bin_ms: float
    num_bursts: int
    burst_rate_hz: float          # バーストの頻度
    width_ms: float               # 1 回の幅の平均 (ビン数 × ビン幅)
    participation: float          # 1 回の間に 1 発以上撃ったニューロンの割合の平均
    outside_fraction: float       # 全スパイクのうちバーストの外で撃たれた割合


def network_bursts(times, ids, num_neurons: int, duration_ms: float,
                   bin_ms: float = EVENT_BIN_MS,
                   active_fraction: float = ACTIVE_FRACTION) -> NetworkBursts:
    """`active_fraction` 以上が撃ったビンの連続区間を 1 回のバーストとして数える。

    `times` は**窓の先頭を 0 とするローカル時刻** [ms]。バーストが無ければ幅と参加率は
    `nan`、頻度は 0、バーストの外の割合は 1 (スパイクが無ければ `nan`)。
    """
    num_neurons = int(num_neurons)
    counts = firing_rate_series(times, ids, num_neurons, duration_ms, bin_ms) * (bin_ms / 1000.0)
    fired = counts > 0
    event = fired.mean(axis=0) >= float(active_fraction)

    # 連続区間の [start, stop) を取り出す
    edges = np.diff(np.concatenate([[0], event.astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)

    total_spikes = float(counts.sum())
    inside_spikes = float(counts[:, event].sum())
    outside = (1.0 - inside_spikes / total_spikes) if total_spikes > 0 else float("nan")
    seconds = float(duration_ms) / 1000.0
    if starts.size == 0:
        return NetworkBursts(bin_ms=float(bin_ms), num_bursts=0, burst_rate_hz=0.0,
                             width_ms=float("nan"), participation=float("nan"),
                             outside_fraction=outside)
    participation = np.array([fired[:, a:b].any(axis=1).mean() for a, b in zip(starts, stops)])
    return NetworkBursts(
        bin_ms=float(bin_ms), num_bursts=int(starts.size),
        burst_rate_hz=float(starts.size) / seconds,
        width_ms=float(np.mean(stops - starts)) * float(bin_ms),
        participation=float(participation.mean()),
        outside_fraction=outside)


def burst_columns(times, ids, num_neurons: int, duration_ms: float,
                  bin_ms: float = EVENT_BIN_MS) -> dict:
    """`metrics.csv` に入れる列。`network_bursts()` の結果を `burst_*` の名前で返す。"""
    bursts = network_bursts(times, ids, num_neurons, duration_ms, bin_ms)
    return {
        "burst_count": bursts.num_bursts,
        "burst_rate_hz": bursts.burst_rate_hz,
        "burst_width_ms": bursts.width_ms,
        "burst_participation": bursts.participation,
        "burst_outside_fraction": bursts.outside_fraction,
    }


# ======================================================================================
# 群ごとの同期 —— 参加率の時系列と Golomb の χ
#
# 上の `population_events()` はネットワーク全体を 1 つの数に畳む。ここは**群
# (クラスタなど) を渡して**、その中だけで測る。損傷は特定の群を孤立させるので、
# 全体の平均では「どこの同期が崩れたか」が見えない。
# ======================================================================================

# χ のビン幅 [ms]。相関・移動エントロピーと同じ 50 ms (論文のフレーム)。
CHI_BIN_MS = 50.0


def _group_counts(times, ids, members, duration_ms: float, bin_ms: float) -> np.ndarray:
    """`members` だけの (群の人数, ビン数) の発火率時系列。行は `members` を昇順にした順。"""
    members = np.unique(np.asarray(members, dtype=np.int64))
    if members.size == 0:
        raise ValueError("members が空です。")
    times = np.asarray(times, dtype=np.float64)
    ids = np.asarray(ids, dtype=np.int64)
    in_group = np.isin(ids, members)
    return firing_rate_series(times[in_group], np.searchsorted(members, ids[in_group]),
                              members.size, duration_ms, bin_ms)


def active_fraction_series(times, ids, members, duration_ms: float,
                           bin_ms: float = EVENT_BIN_MS) -> np.ndarray:
    """`members` のうち各ビンで 1 発以上撃った割合。長さはビン数。

    `times` は**窓の先頭を 0 とするローカル時刻** [ms]、`members` はグローバル ID。
    """
    return (_group_counts(times, ids, members, duration_ms, bin_ms) > 0).mean(axis=0)


def golomb_chi(times, ids, members, duration_ms: float,
               bin_ms: float = CHI_BIN_MS) -> float:
    """Golomb の同期指標 χ。`members` の中だけで測る。

        χ² = Var_t[ 群の平均活動 ] / 平均_i( Var_t[ ニューロン i の活動 ] )

    活動は**ビンごとのスパイク数** (膜電位を全員ぶん記録していないので、スパイク版)。
    全員が揃って揺れれば 1、独立に撃てば約 1/√n (n = 群の人数) に落ちる —— 小さい群
    ほど底が高いので、群どうしを比べるときは `chi_floor(n)` と並べて読むこと。

    群の誰も揺れない (全員が 1 発も撃たない) 窓では `nan`。
    """
    counts = _group_counts(times, ids, members, duration_ms, bin_ms)
    single = float(counts.var(axis=1).mean())
    if single <= 0.0:
        return float("nan")
    return float(np.sqrt(counts.mean(axis=0).var() / single))


def chi_floor(num_members: int) -> float:
    """独立に撃つ `num_members` 本の群で χ が取る値 (の目安 1/√n)。"""
    return 1.0 / float(np.sqrt(num_members)) if num_members > 0 else float("nan")


# ======================================================================================
# 全体イベント —— 偶然の重なりを除いた深さと、モジュールへの広がり
#
# 上の `network_bursts()` は「20% 以上が同じビンで撃つ」を固定閾値で数えるので、
# 発火率が ≳1 Hz では偶然の重なりだけで閾値を越え、ほぼ連続する活動を「深いバースト」と
# 読んでしまう。ここは閾値をサロゲート (各ニューロンの発火列を独立に巡回シフトしたもの。
# 発火率と ISI は保ち、ニューロン間の同時性だけ壊す) から決め、参加率からも偶然の分を除く。
# イベントを「全体 / 1 モジュールだけ」に分けるので、疎で深い全体イベントの頻度と、
# それを薄めるモジュール内のイベントを別々に読める。
# ======================================================================================

# 全体イベントの検出のビン幅 [ms]。相関・χ と同じ 50 ms。
GLOBAL_EVENT_BIN_MS = 50.0
# サロゲートの活動割合のこの分位点を越えたビンを「群が活動」とする。
GLOBAL_EVENT_QUANTILE = 0.999
GLOBAL_EVENT_SURROGATES = 20
# 偶然の分を除いた参加率がこれ以上なら「深い」/「とても深い」。
DEEP_EVENT_DEPTH = 0.5
VERY_DEEP_EVENT_DEPTH = 0.8
# 加わった群の割合がこれ以上なら「広い」。深くて広いものが全体イベント。
GLOBAL_EVENT_GROUP_FRACTION = 0.75


@dataclass(frozen=True)
class GlobalEvents:
    """全体イベントの要約。全部 1 記録窓ぶん。頻度は Hz。"""
    bin_ms: float
    num_groups: int
    threshold_mean: float          # 群ごとの活動閾値 (サロゲートの分位点) の平均
    event_rate_hz: float           # どれかの群が活動している区間の頻度 (深さを問わない)
    local_rate_hz: float           # 1 群だけのイベントの頻度 (群が 1 つなら 0)
    deep_rate_hz: float            # 深さ ≥ DEEP_EVENT_DEPTH の頻度
    very_deep_rate_hz: float       # 深さ ≥ VERY_DEEP_EVENT_DEPTH の頻度
    deep_group_fraction: float     # 深いイベントに加わった群の割合の平均
    global_rate_hz: float          # 深くて広い (= 全体) イベントの頻度
    global_depth: float            # 全体イベントの深さ (偶然の分を除いた参加率) の平均
    global_participation: float    # 全体イベントの生の参加率の平均
    global_width_ms: float         # 全体イベントの幅の平均
    global_spike_fraction: float   # 全スパイクのうち全体イベント中に撃たれた割合
    # 加わった群の中だけで測った深さ。全体の深さ = 広がり × これ、に近い。
    # 「広がりが足りない」のか「各群が浅い」のかを分ける
    global_group_depth: float      # 全体イベントの、加わった群の中での深さの平均
    local_group_depth: float       # 1 群だけのイベントの、その群の中での深さの平均


def global_events(times, ids, num_neurons: int, duration_ms: float, groups=None,
                  bin_ms: float = GLOBAL_EVENT_BIN_MS,
                  quantile: float = GLOBAL_EVENT_QUANTILE,
                  num_surrogates: int = GLOBAL_EVENT_SURROGATES,
                  seed: int = 0) -> GlobalEvents:
    """イベントを検出し、深さ (偶然の分を除いた参加率) と群への広がりで分ける。

    `times` は**窓の先頭を 0 とするローカル時刻** [ms]。`groups` は長さ `num_neurons` の
    群ラベル (モジュールなど)。`None` なら全体を 1 群とみなす (広がりは常に 1)。

    - 群 g の活動割合 A_g(t) がサロゲートの `quantile` 分位点を越えたビンを「g が活動」とし、
      どれかの群が活動しているビンの連続区間を 1 イベントとする
    - 深さ = (part − chance) / (1 − chance)。part はイベント中に 1 発以上撃ったニューロンの
      割合、chance は各ニューロンの発火率とイベント幅から Poisson で見積もった偶然の参加率
    - 群の中での深さ = 同じ式を、そのイベントで活動した群のメンバーだけで計算した値 (群ごとに
      出して平均)

    サロゲートは `seed` で固定するので、同じスパイク列からは同じ値が出る。
    スパイクが無い窓では頻度 0、それ以外は `nan`。
    """
    num_neurons = int(num_neurons)
    labels = (np.zeros(num_neurons, dtype=np.int64) if groups is None
              else np.unique(np.asarray(groups), return_inverse=True)[1].ravel())
    if labels.size != num_neurons:
        raise ValueError(f"groups の長さ {labels.size} が num_neurons {num_neurons} と違います。")
    num_groups = int(labels.max()) + 1
    fired = firing_rate_series(times, ids, num_neurons, duration_ms, bin_ms) > 0
    num_bins = fired.shape[1]
    size = np.bincount(labels, minlength=num_groups).astype(np.float64)

    def group_activity(f):
        out = np.zeros((num_groups, num_bins))
        np.add.at(out, labels, f)
        return out / size[:, None]

    rng = np.random.default_rng(seed)
    rows = np.arange(num_neurons)[:, None]
    surrogate = np.concatenate([
        group_activity(fired[rows, (np.arange(num_bins)[None, :]
                                    - rng.integers(0, num_bins, num_neurons)[:, None]) % num_bins])
        for _ in range(int(num_surrogates))], axis=1)
    threshold = np.quantile(surrogate, quantile, axis=1)
    active = group_activity(fired) > threshold[:, None]

    edges = np.diff(np.concatenate([[0], active.any(axis=0).astype(np.int8), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    seconds = float(duration_ms) / 1000.0
    nan = float("nan")
    if starts.size == 0:
        return GlobalEvents(float(bin_ms), num_groups, float(threshold.mean()),
                            0.0, 0.0, 0.0, 0.0, nan, 0.0, nan, nan, nan,
                            0.0 if np.asarray(times).size else nan, nan, nan)

    spread = np.array([active[:, a:b].any(axis=1).sum() for a, b in zip(starts, stops)]) / num_groups
    part = np.array([fired[:, a:b].any(axis=1).mean() for a, b in zip(starts, stops)])
    width_ms = (stops - starts) * float(bin_ms)
    rates_hz = np.bincount(np.asarray(ids, dtype=np.int64), minlength=num_neurons) / seconds
    chance = 1.0 - np.exp(-np.outer(width_ms / 1000.0, rates_hz)).mean(axis=1)
    depth = (part - chance) / np.maximum(1.0 - chance, 1e-12)

    group_depth = np.empty(starts.size)
    for k, (a, b) in enumerate(zip(starts, stops)):
        on = np.flatnonzero(active[:, a:b].any(axis=1))
        fired_k = fired[:, a:b].any(axis=1)
        prob_k = 1.0 - np.exp(-rates_hz * width_ms[k] / 1000.0)
        part_g = np.bincount(labels, weights=fired_k, minlength=num_groups)[on] / size[on]
        chance_g = np.bincount(labels, weights=prob_k, minlength=num_groups)[on] / size[on]
        group_depth[k] = float(np.mean((part_g - chance_g) / np.maximum(1.0 - chance_g, 1e-12)))

    deep = depth >= DEEP_EVENT_DEPTH
    is_global = deep & (spread >= GLOBAL_EVENT_GROUP_FRACTION)
    local = (np.rint(spread * num_groups) == 1) & (num_groups > 1)
    in_global = np.zeros(num_bins, dtype=bool)
    for a, b in zip(starts[is_global], stops[is_global]):
        in_global[a:b] = True
    spike_bins = np.minimum((np.asarray(times, dtype=np.float64) // bin_ms).astype(np.int64),
                            num_bins - 1)
    return GlobalEvents(
        bin_ms=float(bin_ms), num_groups=num_groups, threshold_mean=float(threshold.mean()),
        event_rate_hz=starts.size / seconds,
        local_rate_hz=float(local.sum()) / seconds,
        deep_rate_hz=float(deep.sum()) / seconds,
        very_deep_rate_hz=float((depth >= VERY_DEEP_EVENT_DEPTH).sum()) / seconds,
        deep_group_fraction=float(spread[deep].mean()) if deep.any() else nan,
        global_rate_hz=float(is_global.sum()) / seconds,
        global_depth=float(depth[is_global].mean()) if is_global.any() else nan,
        global_participation=float(part[is_global].mean()) if is_global.any() else nan,
        global_width_ms=float(width_ms[is_global].mean()) if is_global.any() else nan,
        global_spike_fraction=float(in_global[spike_bins].mean()) if spike_bins.size else nan,
        global_group_depth=float(group_depth[is_global].mean()) if is_global.any() else nan,
        local_group_depth=float(group_depth[local].mean()) if local.any() else nan)


def global_event_columns(times, ids, num_neurons: int, duration_ms: float,
                         groups=None) -> dict:
    """`metrics.csv` に入れる列。`global_events()` の結果を `gevent_*` の名前で返す。"""
    ev = global_events(times, ids, num_neurons, duration_ms, groups)
    return {
        "gevent_bin_ms": ev.bin_ms,
        "gevent_num_groups": ev.num_groups,
        "gevent_threshold_mean": ev.threshold_mean,
        "gevent_event_rate_hz": ev.event_rate_hz,
        "gevent_local_rate_hz": ev.local_rate_hz,
        "gevent_deep_rate_hz": ev.deep_rate_hz,
        "gevent_very_deep_rate_hz": ev.very_deep_rate_hz,
        "gevent_deep_group_fraction": ev.deep_group_fraction,
        "gevent_global_rate_hz": ev.global_rate_hz,
        "gevent_global_depth": ev.global_depth,
        "gevent_global_participation": ev.global_participation,
        "gevent_global_width_ms": ev.global_width_ms,
        "gevent_global_spike_fraction": ev.global_spike_fraction,
        "gevent_global_group_depth": ev.global_group_depth,
        "gevent_local_group_depth": ev.local_group_depth,
    }
