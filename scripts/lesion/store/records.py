"""lesion 実験の記録ファイル名の規約。

**この規約を知っているのはここだけ。** ファイル名だけでなく、**npz の中の鍵の名前**も
書き出しと読み取りが対で持つ (`save_spikes` / `load_spikes` など)。片方だけ変えて
食い違うことがない。

返す値の形は `src/utils/runview.py` の契約 (`Spikes` / `Wiring`) に従う。
**この実験に固有なのはファイル名と npz の鍵名だけで、読めた値の形は全実験共通。**

## develop / akita_soc と規約が違う理由

あちらは `{kind}_{hour}h.npz`、こちらは `{kind}_p{index:03d}.npz`。

1. **時刻をファイル名に埋めると精度を失う。** `f"{hour:g}h"` は probe 間隔が細かいと
   `1.66667e-05h.npz` になり、float の往復で元の時刻に戻らない。
2. **切断前の probe は負の時刻を持つ。** 損傷実験の記録軸は「切断からの経過」であって
   絶対時刻ではない。`{-0.17}h.npz` のような名前は読みにくいうえ並べ替えも壊れる。
3. **develop の解析 CLI に誤読されない。** あちらの glob は `spikes_*h.npz` なので
   `spikes_p003.npz` は掛からない。develop の replot に損傷 run を渡しても
   「見つからない」で止まる。

**時刻の正は npz が持つ** (`record_start_ms`)。ファイル名の index は並べ替えと人間の
目印にしか使わない。これは develop と同じ約束で、違うのは名前の付け方だけ。

## 結合構造が 2 つある

損傷実験は**同じ run の中でシナプス本数が変わる**唯一の実験。Phase 1 (切断前) と
Phase 2 (切断後) で COO が違うので、`connectivity.npz` を 2 本持つ。

    connectivity_pre.npz   Phase 1 の結合 (全シナプス)
    connectivity.npz       Phase 2 の結合 (生き残ったシナプス)

どちらを使うかは probe の phase で決まる (`series.py` の `Window.wiring()`)。
2 つをまたいで重みを比べるときは `analysis/restore.py` の `align_subset_to_coo()` で
生存シナプスへ引き当てる —— **位置で対応づけてはいけない。**
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.runview import MissingData, Spikes, Wiring

# 記録ファイル名 `<種別>_p<番号>.npz`。
PROBE_PATTERN = re.compile(r"(?P<kind>[A-Za-z_]+)_p(?P<index>\d+)\.npz")

SPIKES = "spikes"
WEIGHTS = "weights"

# 結合構造。**2 つあるのが lesion の特徴**で、切断の前後で別ファイルに書く。
POST_CONNECTIVITY_NAME = "connectivity.npz"      # Phase 2 (切断後)
PRE_CONNECTIVITY_NAME = "connectivity_pre.npz"   # Phase 1 (切断前)

# soma の座標。結合構造と同じく run を通して不変なので 1 回だけ書く。
#
# **`no_space` の run は持たない。** config だけからは復元できない (空間コンポーネントが
# RNG を引く) ので、axes と同じ理由でここに残す。これがあると記録窓の view からも
# 空間の図が描ける —— 無ければ座標を要する図は再ビルドしないと出せない。
COORDS_NAME = "coords.npz"

MANIFEST_NAME = "lesion.json"
CUT_NAME = "lesion_cut.npz"
CUT_PROFILE_NAME = "cut_profile.csv"
METRICS_NAME = "metrics.csv"

PHASE_BASELINE = "baseline"
PHASE_POST = "post"

MS_PER_HOUR = 60.0 * 60.0 * 1000.0

# スパイク npz に埋める鍵。**ファイル名から復元しない**ためのもの。
RECORD_START_KEY = "record_start_ms"   # 切断時刻を 0 とする窓の原点 [ms]。baseline は負
RECORD_WINDOW_KEY = "record_window_ms"  # この窓の長さ [ms]。baseline だけ別幅にできる
PHASE_KEY = "phase"


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
    """`directory` 直下の probe を index 昇順で返す。

    glob は数字始まりに限る。同じ接頭辞を持つ probe 以外の記録を拾わないため。
    """
    files = [ProbeFile(index=parse_probe_name(path.name)[1], path=path)
             for path in Path(directory).glob(f"{kind}_p[0-9]*.npz")]
    return sorted(files, key=lambda item: item.index)


# ======================================================================================
# 結合構造
# ======================================================================================

def save_connectivity(path: Path, row, col, shape) -> None:
    """結合構造を書く。**鍵名 (`row` / `col` / `shape`) を知る場所を 1 つにするための対。**"""
    np.savez_compressed(path, row=np.asarray(row), col=np.asarray(col),
                        shape=np.asarray(shape))


def load_connectivity(path: Path) -> Wiring:
    """`connectivity*.npz` を読んで `Wiring` にする。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"結合情報 {path} が見つかりません。"
            " weights_p*.npz と対になる結合構造が必要です (再実行してください)。"
        )
    data = np.load(path, allow_pickle=True)
    shape = np.asarray(data["shape"], dtype=np.int64)
    wiring = Wiring(
        row=np.asarray(data["row"], dtype=np.int64),
        col=np.asarray(data["col"], dtype=np.int64),
        shape=(int(shape[0]), int(shape[1])),
    )
    _require_row_major(wiring, path)
    return wiring


def _require_row_major(wiring: Wiring, path: Path) -> None:
    """記録された COO が**行優先ソート済み**であることを確かめる。

    COO の並びはプロジェクト全体で 1 つ (CLAUDE.md の不変条件 4) だが、それを揃える前に
    作られた run は GeNN の格納順で書かれている。同じネットワークで本数も同じなので、
    **位置で対応づけると黙って別のシナプスに値が乗る。** 読んだ時点で弾く。

    狭義単調増加であることは「行優先かつ (pre, post) の重複なし」と同値。
    """
    if wiring.row.size < 2:
        return
    # **int64 で掛ける。** int32 のままだと N が大きいときに折り返り、別のペアが
    # 同じキーになって「ソート済み」に見えてしまう。
    key = wiring.row * int(wiring.shape[1]) + wiring.col
    bad = int(np.count_nonzero(np.diff(key) <= 0))
    if bad:
        raise ValueError(
            f"{path} の結合が行優先ソート済みではありません ({bad} 箇所)。\n"
            "  COO の並びを揃える前に作られた run です。位置で重みと対応づけると"
            " 別のシナプスに乗るので、読み込みを拒否しました。\n"
            "  再実行してください (docs/technical/compat_breaks.md 参照)。"
        )


def post_connectivity_path(directory: Path) -> Path:
    return Path(directory) / POST_CONNECTIVITY_NAME


def pre_connectivity_path(directory: Path) -> Path:
    return Path(directory) / PRE_CONNECTIVITY_NAME


# ======================================================================================
# probe 1 つぶん
# ======================================================================================

def save_coords(path: Path, coords) -> None:
    """soma の座標を書く。キー名を知っているのは読み書きのこの対だけ。

    `no_space` の run では呼ばないこと (座標が無いことと、`inf` で埋めた座標を
    保存したことを読む側から区別できなくなる)。
    """
    np.savez_compressed(path, data=np.asarray(coords, dtype=np.float64))


def load_coords(path: Path) -> np.ndarray:
    """`save_coords()` が書いた座標を読む。非有限値は「座標が無い」として扱う。"""
    with np.load(path) as data:
        coords = np.asarray(data["data"], dtype=np.float64)
    if not np.all(np.isfinite(coords)):
        raise MissingData("coords", "座標に有限でない値があります (no_space)")
    return coords


def save_weight_values(path: Path, values: np.ndarray) -> None:
    """重みを値ベクトルとして書く。キー名を知っているのは読み書きのこの対だけ。"""
    np.savez_compressed(path, data=np.asarray(values))


def load_weight_values(path: Path) -> np.ndarray:
    """重み npz から値ベクトル (1D) を読む。その probe の結合と index 整合。"""
    data = np.load(path, allow_pickle=True)
    if "data" not in data.files:
        raise KeyError(f"{path} にキー 'data' がありません。")
    return np.asarray(data["data"], dtype=np.float64).reshape(-1)


def save_spikes(path: Path, times, ids, *, record_start_ms: float,
                record_window_ms: float, phase: str) -> None:
    """スパイクと、**その窓が何であるか** (原点・窓幅・phase) を同じ npz に書く。

    窓幅を窓ごとに持つので、baseline と post で幅が違っても発火率が正しく出る。
    """
    np.savez_compressed(path, times=np.asarray(times), ids=np.asarray(ids),
                        **{RECORD_START_KEY: float(record_start_ms),
                           RECORD_WINDOW_KEY: float(record_window_ms),
                           PHASE_KEY: str(phase)})


def read_window_meta(path: Path) -> tuple[float, float, str]:
    """スパイク npz から `(原点 [ms], 窓幅 [ms], phase)` だけを読む。

    npz は鍵ごとに展開されるので、ここで配列は解かない。
    """
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in (RECORD_START_KEY, RECORD_WINDOW_KEY, PHASE_KEY)
                   if key not in data.files]
        if missing:
            raise KeyError(
                f"{path} に {missing} がありません。"
                " 窓の素性を持たない古い run です (再実行してください)。"
            )
        return (float(data[RECORD_START_KEY]), float(data[RECORD_WINDOW_KEY]),
                str(data[PHASE_KEY]))


def load_spikes(path: Path) -> Spikes:
    """スパイク npz を読み、**窓の原点を引いたローカル時刻**にして返す。

    絶対時刻は返さない (契約の `Spikes.times` はローカル)。絶対時刻を既定にすると、
    アバランチ分割は同じでも burstiness のビン割りが静かにずれる。
    """
    start_ms, _window_ms, _phase = read_window_meta(path)
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["times"], dtype=np.float64)
        ids = np.asarray(data["ids"])
    return Spikes(times=times - start_ms, ids=ids)


# ======================================================================================
# 損傷条件の記録
# ======================================================================================

def save_manifest(directory: Path, manifest: dict) -> Path:
    """`lesion.json` を書く。**切断で何が起きたかの記録はこれが正。**"""
    path = Path(directory) / MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_manifest(directory: Path) -> dict:
    return json.loads((Path(directory) / MANIFEST_NAME).read_text(encoding="utf-8"))


def save_cut(directory: Path, per_synapse: dict) -> Path:
    """切断されたシナプスごとの素性を記録する。**何を失ったか**を後から数えられるように。

    切断本数だけでは「ハブの出力を集中的に落とした」のか「弱い結合を薄く広く落とした」
    のかが区別できないので、重み・次数・participation・ハブ役割まで 1 本ごとに残す。
    """
    path = Path(directory) / CUT_NAME
    np.savez_compressed(path, **{name: np.asarray(values)
                                 for name, values in per_synapse.items()})
    return path


def load_cut(directory: Path) -> dict:
    with np.load(Path(directory) / CUT_NAME, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


# ======================================================================================
# 表
# ======================================================================================

def write_table(rows: list[dict], out_path: Path) -> Path:
    """dict の並びを CSV にする。**列は最初の行のキー順**、後の行で増えた分は末尾。

    `MetricsWriter` が「1 行ずつ追記する記録」なのに対し、こちらは**一度に確定する表**。
    """
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


class MetricsWriter:
    """`metrics.csv` を **1 行ずつ追記**する。途中で落ちた run もそこまでが読める。

    ヘッダは**最初の行で確定**する。以降の行でキーが変わると例外。
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def append(self, row: dict) -> None:
        keys = list(row.keys())
        if self._fieldnames is None:
            self._fieldnames = keys
            with open(self.path, "w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=keys).writeheader()
        elif keys != self._fieldnames:
            raise ValueError(
                "metrics の列が途中で変わりました。\n"
                f"  最初の行: {self._fieldnames}\n"
                f"  今回の行: {keys}"
            )
        with open(self.path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self._fieldnames).writerow(row)
