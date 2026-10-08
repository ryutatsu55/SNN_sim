"""基準ネットワークの構造を、モデル・重み・dtから独立に保存する。"""

import hashlib
import json
from pathlib import Path

import numpy as np

ARRAY_KEYS = (
    "ids",
    "polarity",
    "rows",
    "cols",
    "arrival_ms",
    "coords",
    "stimulus_orders",
    "autapse_exc_order",
    "autapse_inh_order",
)


def array_hash(arrays, keys=ARRAY_KEYS):
    digest = hashlib.sha256()
    for key in keys:
        a = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(a.dtype.str.encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def validate_snapshot(arrays, meta):
    n, ne = meta["n"], meta["n_exc"]
    if meta.get("version") != 1 or not 0 < ne < n:
        raise ValueError("未対応の配線形式または不正なE/I数です")
    if set(arrays) != set(ARRAY_KEYS):
        raise ValueError("配線配列の項目が一致しません")
    for key in (
        "ids",
        "polarity",
        "rows",
        "cols",
        "stimulus_orders",
        "autapse_exc_order",
        "autapse_inh_order",
    ):
        if arrays[key].dtype.kind not in "iu":
            raise ValueError(f"{key}は整数配列が必要です")
    if not np.array_equal(arrays["ids"], np.arange(n)):
        raise ValueError("細胞IDは0からの連番が必要です")
    expected_polarity = np.r_[np.zeros(ne), np.ones(n - ne)]
    if not np.array_equal(arrays["polarity"], expected_polarity):
        raise ValueError("E/I割当が不正です")
    rows, cols = arrays["rows"], arrays["cols"]
    if (
        rows.ndim != 1
        or cols.shape != rows.shape
        or arrays["arrival_ms"].shape != rows.shape
    ):
        raise ValueError("COO配列の形状が不正です")
    if (
        np.any(rows >= n)
        or np.any(cols >= n)
        or np.any(rows < 0)
        or np.any(cols < 0)
        or np.any(rows == cols)
    ):
        raise ValueError("範囲外の接続または自己結合があります")
    codes = rows.astype(np.int64) * n + cols
    if np.any(np.diff(codes) <= 0):
        raise ValueError("COOは重複なし・行優先の昇順で保存してください")
    if not np.all(np.isfinite(arrays["arrival_ms"])) or np.any(
        arrays["arrival_ms"] <= 0
    ):
        raise ValueError("実効到着遅延は有限の正値が必要です")
    coords = arrays["coords"]
    dimension = 0 if meta["space"]["kind"] == "none" else 2
    if coords.shape != (n, dimension) or not np.all(np.isfinite(coords)):
        raise ValueError("座標の次元・有限性が不正です")
    if dimension:
        if np.any(coords < 0) or np.any(coords > meta["space"]["extent"]):
            raise ValueError("座標が配置範囲外です")
    orders = arrays["stimulus_orders"]
    if orders.shape != (len(meta["stimulus_seeds"]), ne):
        raise ValueError("刺激対象順序の形状が不正です")
    for order in [*orders, arrays["autapse_exc_order"]]:
        if not np.array_equal(np.sort(order), np.arange(ne)):
            raise ValueError("興奮性対象IDの順列が不正です")
    if not np.array_equal(np.sort(arrays["autapse_inh_order"]), np.arange(ne, n)):
        raise ValueError("抑制性対象IDの順列が不正です")


def save_snapshot(path, arrays, meta):
    validate_snapshot(arrays, meta)
    meta = dict(meta)
    meta["array_hash"] = array_hash(arrays)
    payload = json.dumps(meta, sort_keys=True, ensure_ascii=False, allow_nan=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream, **arrays, metadata=np.array(payload), fingerprint=np.array(digest)
        )
    return digest


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        payload = str(data["metadata"].item())
        fingerprint = str(data["fingerprint"].item())
        arrays = {key: data[key].copy() for key in ARRAY_KEYS}
    if hashlib.sha256(payload.encode()).hexdigest() != fingerprint:
        raise ValueError("配線メタデータのハッシュが一致しません")
    meta = json.loads(payload)
    validate_snapshot(arrays, meta)
    if array_hash(arrays) != meta["array_hash"]:
        raise ValueError("配線配列のハッシュが一致しません")
    return arrays, meta, fingerprint


def arrival_to_dendritic_ms(arrival_ms, dt):
    """GeNNの伝達更新1dtを含め、要求する実効到着遅延を維持する。"""
    arrival = np.asarray(arrival_ms, dtype=float)
    if not np.isfinite(dt) or dt <= 0 or not np.all(np.isfinite(arrival)):
        raise ValueError("遅延とdtは有限値が必要です")
    steps = np.rint(arrival / dt).astype(np.int64)
    if (
        np.any(steps < 1)
        or np.any(steps > 256)
        or not np.allclose(steps * dt, arrival, rtol=0, atol=1e-8)
    ):
        raise ValueError("実効遅延はdtの整数倍かつ1〜256ステップが必要です")
    return (steps - 1) * dt
