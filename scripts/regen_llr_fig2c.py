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
from src.core.config_manager import ConfigManager
from src.core.layout import NetworkLayout
from src.core.output_manager import AXES_NAME, CONFIG_NAME, DATA_SUBDIR, locate, require
from src.utils.analysis.avalanche import split_avalanches
from src.utils.analysis.powerlaw import log_likelihood_ratio_power_vs_exponential
from src.utils.experiments.akita_soc.fig2c import plot_figure2c
from src.utils.experiments.akita_soc.runio import (
    SPIKES,
    WEIGHTS,
    record_filename,
    record_glob,
)


def spikes_path(mdir, hour):
    """記録時刻からスパイクファイルを引く。命名は runio.record_filename が唯一の定義。"""
    p = os.path.join(mdir, record_filename(SPIKES, hour))
    return p if os.path.exists(p) else None


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
        run_root = os.path.dirname(mdir) if os.path.basename(mdir) == DATA_SUBDIR else mdir
        # weights と config が無ければ figure2c は描けない
        if not glob.glob(os.path.join(mdir, record_glob(WEIGHTS))) or locate(mdir, CONFIG_NAME) is None:
            print(f"  SKIP {mc} (weights/config無し)"); skip += 1; continue
        try:
            msg = update_llr(mdir)
            # 保存物から NetworkLayout を復元する (自動軸は config.yaml、外部軸は npz)。
            config = ConfigManager().load_resolved(require(mdir, CONFIG_NAME))
            layout = NetworkLayout.from_config(config)
            axes_path = locate(mdir, AXES_NAME)
            if axes_path is not None:
                layout.load_axes_file(axes_path)
            plot_figure2c(mdir, layout, output_dir=run_root)
            plt.close("all")
            print(f"  OK   {run_root}  [{msg}]"); ok += 1
        except Exception as e:
            print(f"  FAIL {mc}: {e}"); fail += 1
    print(f"\n完了: OK={ok}, SKIP={skip}, FAIL={fail}")


if __name__ == "__main__":
    main()
