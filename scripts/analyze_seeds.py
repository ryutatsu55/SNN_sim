#!/usr/bin/env python3
"""seed別 akita_soc_delay の遷移時刻を集計する（実行中でも逐次OK）。

各 outputs/akita_soc_delay/*_seedN/ について:
  - weights_{h}h.npz の最大重み(max w)
  - spikes_{h}h.npz の平均発火率(Hz)
を時刻ごとに出し、遷移時刻 = 最初に max w >= 0.99 になった hour とみなす。
"""
import glob, os, re, sys
import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "outputs/akita_soc_delay"
N = 100
WINDOW_S = 600.0  # record_window_ms=600000 ms = 10 min

def seed_of(p):
    m = re.search(r"_seed(\d+)", p)
    return int(m.group(1)) if m else -1

def hour_of(p):
    m = re.search(r"_(\d+)h\.npz$", p)
    return int(m.group(1)) if m else None

dirs = sorted(glob.glob(os.path.join(BASE, "*_seed*/")), key=seed_of)
if not dirs:
    print(f"no seed dirs under {BASE}"); sys.exit(0)

print(f"{'seed':>4} | {'last_h':>6} | {'transition_h':>12} | rate@h(Hz) per hour")
print("-" * 80)
rows = []
for d in dirs:
    s = seed_of(d)
    # weights / spikes は run直下 or data/ サブdir のどちらかにある
    wfiles = glob.glob(os.path.join(d, "weights_*h.npz")) + glob.glob(os.path.join(d, "data", "weights_*h.npz"))
    sfiles = glob.glob(os.path.join(d, "spikes_*h.npz")) + glob.glob(os.path.join(d, "data", "spikes_*h.npz"))
    wbyh = {hour_of(f): f for f in wfiles if hour_of(f) is not None}
    sbyh = {hour_of(f): f for f in sfiles if hour_of(f) is not None}
    hours = sorted(set(wbyh) | set(sbyh))
    if not hours:
        print(f"{s:>4} | {'(none)':>6} |"); continue
    trans = None
    rates = []
    for h in hours:
        mx = None
        if h in wbyh:
            try: mx = float(np.load(wbyh[h])["weights"].max())
            except Exception: mx = None
        rt = None
        if h in sbyh:
            try:
                d_ = np.load(sbyh[h]); ids = d_[d_.files[0]] if "ids" not in d_.files else d_["ids"]
                rt = ids.size / (N * WINDOW_S)
            except Exception: rt = None
        rates.append(f"{h}h:{rt:.2f}" if rt is not None else f"{h}h:?")
        if trans is None and mx is not None and mx >= 0.99:
            trans = h
    last = hours[-1]
    rows.append((s, trans))
    print(f"{s:>4} | {last:>6} | {str(trans):>12} | " + " ".join(rates))

done = [t for _, t in rows if t is not None]
print("-" * 80)
if done:
    print(f"遷移検出済み {len(done)}/{len(rows)} seeds | 遷移hour: min={min(done)} max={max(done)} mean={np.mean(done):.1f}")
else:
    print("まだ遷移(max w>=0.99)を検出した seed はありません（実行初期）。")
