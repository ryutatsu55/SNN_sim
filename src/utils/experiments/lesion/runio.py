"""損傷実験の出力ファイル規約。

**この規約を知っているのはここだけ** —— `akita_soc/runio.py` と同じ約束。

akita_soc の `{kind}_{hour}h.npz` を使わず `{kind}_p{index:03d}.npz` にしてある理由:

1. **時刻をファイル名に埋めると精度を失う。** `f"{hour:g}h"` は probe 間隔が細かいと
   `1.66667e-05h.npz` になり、float の往復で元の時刻に戻らない。
2. **切断前の probe は負の時刻を持つ。** 損傷実験の記録軸は「切断からの経過」であって
   絶対時刻ではない。
3. **akita_soc の解析 CLI に誤読されない。** あちらの glob は `spikes_*h.npz`、regex は
   末尾 `h.npz` 必須なので、`spikes_p003.npz` はどちらにも掛からない。
   `develop.py --replot-from` に損傷 run を渡しても「見つからない」で止まる。

時刻は `probes.csv` が持つ (index -> phase / 切断からの経過 ms / 窓幅)。
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROBE_PATTERN = re.compile(r"(?P<kind>[A-Za-z_]+)_p(?P<index>\d+)\.npz")

SPIKES = "spikes"
WEIGHTS = "weights"
ISI = "isi"

MANIFEST_NAME = "lesion.json"
CUT_NAME = "lesion_cut.npz"
CUT_PROFILE_NAME = "cut_profile.csv"
PROBES_NAME = "probes.csv"
METRICS_NAME = "metrics.csv"
STRUCTURE_NAME = "structure.csv"
# **`weights_p*` に見えない名前にすること。** probe の glob と衝突すると
# discover_probes がこのファイルを probe と誤認して落ちる。
PRE_WEIGHTS_NAME = "weights_at_cut.npz"

PHASE_BASELINE = "baseline"
PHASE_POST = "post"


@dataclass(frozen=True)
class ProbeFile:
    index: int
    path: Path


def probe_filename(kind: str, index: int) -> str:
    """記録ファイル名を組み立てる。読み取り側と同じ規約を使うための唯一の入口。"""
    return f"{kind}_p{int(index):03d}.npz"


def parse_probe_name(name: str) -> tuple[str, int]:
    """`spikes_p003.npz` -> ("spikes", 3)。規約に合わなければ ValueError。"""
    match = PROBE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"probe ファイル名の規約に合いません: {name}")
    return match["kind"], int(match["index"])


def discover_probes(directory: Path, kind: str) -> list[ProbeFile]:
    """`directory` 直下の probe を index 昇順で返す。"""
    # glob を数字始まりに限る。`weights_at_cut.npz` のような同じ接頭辞の記録を
    # probe と取り違えないため (規約外の名前は parse_probe_name が弾くが、そこまで
    # 行かせない方が事故が早く分かる)。
    files = [ProbeFile(index=parse_probe_name(path.name)[1], path=path)
             for path in Path(directory).glob(f"{kind}_p[0-9]*.npz")]
    return sorted(files, key=lambda item: item.index)


def save_manifest(run_dir: Path, manifest: dict) -> Path:
    """`lesion.json` を書く。損傷条件の記録は**これが正**。"""
    path = Path(run_dir) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_manifest(run_dir: Path) -> dict:
    path = Path(run_dir) / MANIFEST_NAME
    if not path.exists():
        path = Path(run_dir) / "data" / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def save_cut(run_dir: Path, per_synapse: dict) -> Path:
    """切断されたシナプスごとの素性を記録する。**何を失ったか**を後から数えられるように。

    `metrics.cut_profile()` が返す配列辞書をそのまま受ける。切断本数だけでは
    「ハブの出力を集中的に落とした」のか「弱い結合を薄く広く落とした」のかが
    区別できないので、重み・次数・participation・ハブ役割まで 1 本ごとに残す。
    """
    path = Path(run_dir) / CUT_NAME
    np.savez_compressed(path, **{name: np.asarray(values)
                                 for name, values in per_synapse.items()})
    return path


def write_rows_csv(rows: list[dict], out_path: Path) -> Path:
    """dict の並びを CSV に落とす。列は最初の行のキー順 + 後の行で増えた分を末尾に。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path
