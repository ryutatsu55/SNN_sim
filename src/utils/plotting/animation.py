"""スパイク列を時間発展のアニメーション (MP4) にする。

細胞体の位置に点を打ち、発火からの経過時間で明るさを減衰させる。座標を持たない run
(`no_space`) では自動で正方格子に並べる —— 空間が無いこと自体は異常ではないので、
ここだけは `MissingData` を投げずに落とす先を用意してある。

ffmpeg が要る。無ければ `RuntimeError`。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from src.utils.runview import optional

# 見た目と書き出し。**引数にしない** —— 変えたくなったらここを直す。
FPS = 30
DECAY_TAU_MS = 20.0
# 減衰が見えなくなる打ち切り。tau の 5 倍 (e^-5 ≈ 0.7%) で十分。
DECAY_CUTOFF_SCALE = 5.0
FIGSIZE = (6, 6)
MARKER_SIZE = 80
CMAP = "inferno"


def _validate_spike_arrays(spike_time: np.ndarray, neuron_id: np.ndarray):
    spike_time = np.asarray(spike_time, dtype=float).reshape(-1)
    neuron_id = np.asarray(neuron_id).reshape(-1)

    if spike_time.size == 0 or neuron_id.size == 0:
        raise ValueError("spike_time and neuron_id arrays must not be empty.")
    if spike_time.shape[0] != neuron_id.shape[0]:
        raise ValueError("spike_time and neuron_id arrays must have the same length.")
    if not np.all(np.isfinite(spike_time)):
        raise ValueError("spike_time must contain only finite values.")

    if not np.issubdtype(neuron_id.dtype, np.integer):
        if not np.all(np.equal(neuron_id, np.floor(neuron_id))):
            raise ValueError("neuron_id must contain integer values.")
        neuron_id = neuron_id.astype(int)
    else:
        neuron_id = neuron_id.astype(int, copy=False)

    return spike_time, neuron_id


def _grid_coords(neuron_ids: np.ndarray):
    unique_ids = np.asarray(np.unique(neuron_ids), dtype=int)
    cols = int(np.ceil(np.sqrt(unique_ids.size)))
    rows = int(np.ceil(unique_ids.size / cols))

    x = np.arange(unique_ids.size) % cols
    y = rows - 1 - (np.arange(unique_ids.size) // cols)
    coords = np.column_stack([x, y]).astype(float)

    return unique_ids, coords


def _resolve_animation_coords(neuron_id: np.ndarray, coords: np.ndarray | None):
    unique_ids = np.asarray(np.unique(neuron_id), dtype=int)

    if coords is None:
        return _grid_coords(unique_ids)

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError("coords must have shape (num_neurons, 2) or (num_neurons, 3).")

    if coords.shape[0] == 0:
        raise ValueError("coords must not be empty.")

    # no_space では NaN 座標が使われるため、自動グリッドへフォールバックする。
    if not np.all(np.isfinite(coords)):
        return _grid_coords(unique_ids)

    coords_2d = coords[:, :2]
    min_id = int(np.min(neuron_id))
    max_id = int(np.max(neuron_id))

    if min_id >= 0 and max_id < coords_2d.shape[0]:
        display_ids = np.arange(coords_2d.shape[0], dtype=int)
        return display_ids, coords_2d

    if coords_2d.shape[0] == unique_ids.size:
        return unique_ids, coords_2d

    raise ValueError(
        "coords length must either cover neuron_id values directly or match the number of unique neuron IDs."
    )


def _compute_decay_intensity(
    frame_time: float,
    sorted_time: np.ndarray,
    sorted_id: np.ndarray,
    display_index: dict[int, int],
    num_display_ids: int,
    decay_tau_ms: float,
    decay_cutoff_ms: float,
):
    if decay_tau_ms <= 0:
        raise ValueError("decay_tau_ms must be greater than 0.")
    if decay_cutoff_ms <= 0:
        raise ValueError("decay_cutoff_ms must be greater than 0.")

    left = np.searchsorted(sorted_time, frame_time - decay_cutoff_ms, side="left")
    right = np.searchsorted(sorted_time, frame_time, side="right")

    intensity = np.zeros(num_display_ids, dtype=float)
    if right <= left:
        return intensity

    recent_time = sorted_time[left:right]
    recent_id = sorted_id[left:right]
    elapsed_ms = frame_time - recent_time
    recent_intensity = np.exp(-elapsed_ms / decay_tau_ms)
    recent_index = np.fromiter(
        (display_index[int(neuron)] for neuron in recent_id),
        dtype=int,
        count=recent_id.size,
    )
    np.maximum.at(intensity, recent_index, recent_intensity)
    return intensity


def spike_animation(window, out_path) -> None:
    """記録窓 1 つのスパイク列を MP4 アニメーションにする。

    座標は**あれば**使い、無ければ正方格子へ並べる (`no_space` の run でも
    「どのニューロンがいつ光ったか」は読める)。長さは記録窓の幅。
    """
    spikes = window.spikes()
    spike_time, neuron_id = _validate_spike_arrays(spikes.times, spikes.ids)
    coords = optional(window.coords)
    fps = FPS
    decay_tau_ms = DECAY_TAU_MS
    decay_cutoff_ms = decay_tau_ms * DECAY_CUTOFF_SCALE

    duration_ms = float(window.record_window_ms)
    if duration_ms <= 0:
        duration_ms = float(np.max(spike_time))
    if duration_ms <= 0:
        raise ValueError("記録窓の長さが 0 です。アニメーションにできません。")

    display_ids, display_coords = _resolve_animation_coords(neuron_id, coords)
    display_index = {int(neuron): idx for idx, neuron in enumerate(display_ids)}

    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is required to save MP4 animations, but matplotlib cannot find it.")

    order = np.argsort(spike_time)
    sorted_time = spike_time[order]
    sorted_id = neuron_id[order]

    frame_count = max(2, int(np.ceil(duration_ms / 1000.0 * fps)) + 1)
    frame_times = np.linspace(0.0, duration_ms, frame_count)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    scatter = ax.scatter(
        display_coords[:, 0],
        display_coords[:, 1],
        s=MARKER_SIZE,
        c=np.zeros(display_ids.size),
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        edgecolors="0.25",
        linewidths=0.4,
    )

    ax.set_title(f"Spike animation {window.hour:g} h")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.08)

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.75,
            "edgecolor": "none",
        },
    )

    def update(frame_time):
        intensity = _compute_decay_intensity(
            frame_time=frame_time,
            sorted_time=sorted_time,
            sorted_id=sorted_id,
            display_index=display_index,
            num_display_ids=display_ids.size,
            decay_tau_ms=decay_tau_ms,
            decay_cutoff_ms=decay_cutoff_ms,
        )
        scatter.set_array(intensity)
        time_text.set_text(f"t = {frame_time:.1f} ms")
        return scatter, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frame_times,
        interval=1000 / fps,
        blit=True,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps)
    try:
        anim.save(str(out_path), writer=writer)
    finally:
        plt.close(fig)
