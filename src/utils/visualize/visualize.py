import numpy as np
import matplotlib.pyplot as plt


def export_spike_csv(spike_time: np.ndarray, neuron_id: np.ndarray, output_path="spikes.csv"):
    """発火時刻とニューロンIDをCSVとして保存する関数"""

    spike_time = np.asarray(spike_time, dtype=float).reshape(-1)
    neuron_id = np.asarray(neuron_id).reshape(-1)

    if spike_time.size == 0 or neuron_id.size == 0:
        raise ValueError("spike_time and neuron_id arrays must not be empty.")
    if spike_time.shape[0] != neuron_id.shape[0]:
        raise ValueError("spike_time and neuron_id arrays must have the same length.")

    spike_table = np.column_stack((spike_time, neuron_id.astype(int, copy=False)))
    np.savetxt(
        output_path,
        spike_table,
        delimiter=",",
        header="spike_time,neuron_id",
        comments="",
        fmt=["%.10f", "%d"],
    )


def raster(spike_time: np.ndarray, neuron_id: np.ndarray, title="Raster"):
    """発火時刻とニューロンIDからラスタープロットを作成・保存する関数"""

    fig, ax = plt.subplots(figsize=(8, 4))

    if spike_time.size >0:
        ax.scatter(spike_time, neuron_id, s=8, c='blue', marker='.', linewidths=0)
        ax.set_ylim(np.min(neuron_id) - 0.5, np.max(neuron_id) + 0.5)
        ax.set_xlim(0, np.max(spike_time))
    else:
        raise ValueError("spike_time and neuron_id arrays must not be empty.")
    
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Neuron ID')

    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.close(fig)


def PQN_test(V_data, I_in, config, title="PQN_V_test"):
    tmax = config.task.duration/1000
    time_axis = np.arange(len(V_data)) * config.simulation.dt / 1000.0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8, 4), sharex=True)
    
    ax1.plot(time_axis, V_data, color='tab:blue')
    ax1.set_ylabel('v')
    ax1.set_xlim(0, tmax)
    
    ax2.plot(time_axis, I_in, color='black')
    ax2.set_ylabel('I')
    ax2.set_xlabel('[s]')
    ax2.set_xlim(0, tmax)
    
    plt.tight_layout()
    plt.savefig(f"{title}.png")
