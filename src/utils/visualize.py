import numpy as np
import matplotlib.pyplot as plt

def raster(spike_data, title="Raster"):
    """ラスタープロット等の図を出力するダミー関数"""
    print(f" [Vis] Generating Plot: {title} (saving to file...)")
    # matplotlib 等の処理をここに記述

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