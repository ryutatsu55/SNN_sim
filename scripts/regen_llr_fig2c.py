#!/usr/bin/env python3
"""全 outputs run の metrics.csv の llr 列を論文準拠の新LLRで再計算し、
figure2c_reproduction.png を再生成する一括スクリプト。

各 run の metrics.csv があるディレクトリ(<run>/data または <run>)を対象に:
  1. 各 hour の spikes_*h.npz からアバランシェを再分割し、新LLRで llr 列を更新
  2. plot_figure2c でグラフ(figure2c_reproduction.png)を run ルートに再生成
"""
import sys, os, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from src.utils.akita_soc import split_avalanches, log_likelihood_ratio_power_vs_exponential
from src.utils.visualize.akita_soc_fig2c import plot_figure2c


def spikes_path(mdir, hour):
    for name in (f"spikes_{hour:g}h.npz", f"spikes_{int(round(hour))}h.npz"):
        p = os.path.join(mdir, name)
        if os.path.exists(p):
            return p
    return None


def update_llr(mdir):
    mc = os.path.join(mdir, "metrics.csv")
    rows = list(csv.DictReader(open(mc)))
    if not rows or "llr" not in rows[0] or "hour" not in rows[0]:
        return "no llr/hour col"
    updated = 0
    for r in rows:
        hour = float(r["hour"])
        sp = spikes_path(mdir, hour)
        if sp is None:
            continue
        d = np.load(sp)
        times = d["times"] if "times" in d.files else d[d.files[0]]
        local = np.sort(times) - hour * 3.6e6
        av = split_avalanches(local)
        r["llr"] = "%r" % float(log_likelihood_ratio_power_vs_exponential(av.sizes))
        updated += 1
    if updated == 0:
        return "no spikes"
    with open(mc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return f"llr updated ({updated}h)"


def main():
    metrics_files = sorted(glob.glob("outputs/**/metrics.csv", recursive=True))
    print(f"対象 metrics.csv: {len(metrics_files)} 件")
    ok = skip = fail = 0
    for mc in metrics_files:
        mdir = os.path.dirname(mc)
        run_root = os.path.dirname(mdir) if os.path.basename(mdir) == "data" else mdir
        # weights と config が無ければ figure2c は描けない
        if not glob.glob(os.path.join(mdir, "weights_*h.npz")) or not os.path.exists(os.path.join(mdir, "config.yaml")):
            print(f"  SKIP {mc} (weights/config無し)"); skip += 1; continue
        try:
            msg = update_llr(mdir)
            plot_figure2c(mdir, output_dir=run_root)
            plt.close("all")
            print(f"  OK   {run_root}  [{msg}]"); ok += 1
        except Exception as e:
            print(f"  FAIL {mc}: {e}"); fail += 1
    print(f"\n完了: OK={ok}, SKIP={skip}, FAIL={fail}")


if __name__ == "__main__":
    main()
