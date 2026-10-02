"""結合・重み・遅延コンポーネントの `generate()` が返す COO の形と実現値を固定する。

- 結合: (rows, cols) が int32・行優先ソート済み・重複なし (`GlobalCOO` の不変条件)
- 重み / 遅延: 結合 1 本につき 1 値の float32
- 実現値: 同じ seed から同じ値が出て、生成後の rng 状態も同じであること

実現値の固定値 (`GOLDEN`) は、密な (N, N) で生成していた旧実装とビット単位で一致することを
確かめた実装から採ったもの。ここが変わると、同じ config.yaml から別のネットワークが
できる (`docs/technical/compat_breaks.md` に記録が要る)。
"""
import hashlib
from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.models.network import connectors as C
from src.models.network import delays as D
from src.models.network import weights as W

N = 157
N_EXC = 120


class _Layout:
    """`ids_by("polarity")` / `ids_by("module")` だけを持つ最小の layout。E/I は散在させる。"""

    def __init__(self, n, n_exc, modules=None):
        perm = np.random.RandomState(7).permutation(n)
        self._polarity = {"excitatory": np.sort(perm[:n_exc]), "inhibitory": np.sort(perm[n_exc:])}
        self._modules = modules

    def ids_by(self, axis):
        if axis == "polarity":
            return self._polarity
        if axis == "module":
            return {i: np.sort(m) for i, m in enumerate(self._modules)}
        raise KeyError(axis)

    def has_axis(self, axis):
        return axis == "module" and self._modules is not None


def _coords(n=N, seed=3, extent=500.0):
    return np.random.RandomState(seed).uniform(0, extent, (n, 2))


def _module_layout():
    perm = np.random.RandomState(9).permutation(N)
    return _Layout(N, N_EXC, [perm[:40], perm[40:100], perm[100:]])


_GAUSS = dict(sigma_ee=150.0, p0_ee=0.3, sigma_ei=100.0, p0_ei=0.2,
              sigma_ie=120.0, p0_ie=0.4, sigma_ii=80.0, p0_ii=0.1)
_BLOCK = dict(within_module_connection_prob=0.3, between_module_connection_prob=0.05)

# (名前, クラス, config, num_neurons, coords, layout)
CONNECTIONS = [
    ("constant_prob", C.ConstantProbabilityTopology, NS(p=0.1, allow_self_connections=False), N, None, None),
    ("constant_prob_self", C.ConstantProbabilityTopology, NS(p=0.1, allow_self_connections=True), N, None, None),
    ("constant_prob_multiblock", C.ConstantProbabilityTopology, NS(p=0.01, allow_self_connections=False), 4500, None, None),
    ("distance_based", C.DistanceBasedTopology, NS(max_prob=0.5, sigma=120.0), N, _coords(), None),
    ("gaussian_distance_type", C.GaussianDistanceTypeTopology, NS(**_GAUSS), N, _coords(), _Layout(N, N_EXC)),
    ("prob_based_block_1", C.BlockRandomTopology, NS(**_BLOCK, allow_self_connections=False, num_modules=1), N, None, None),
    ("prob_based_block_2", C.BlockRandomTopology, NS(**_BLOCK, allow_self_connections=True, num_modules=2), N, None, None),
    ("prob_based_block_5", C.BlockRandomTopology, NS(**_BLOCK, allow_self_connections=False, num_modules=5), N, None, None),
    ("prob_based_block_axis", C.BlockRandomTopology, NS(**_BLOCK, allow_self_connections=False, num_modules=99), N, None, _module_layout()),
    ("optional_connect", C.OptionalConnection, NS(src_ID=[5, 3, 5, 9], tgt_ID=[2, 7, 2, 0]), N, None, None),
    ("C.elegans", C.C_elegansConnection, NS(), 81, None, None),
]

_MASK_RNG = np.random.RandomState(5)
_ROWS, _COLS = np.nonzero(_MASK_RNG.random((N, N)) < 0.2)
_ROWS = _ROWS.astype(np.int32)
_COLS = _COLS.astype(np.int32)

# (名前, クラス, config, num_neurons, coords, layout)
VALUES = [
    ("w/constant", W.ConstantWeight, NS(base_weight=0.7), N, None, None),
    ("w/normal_broad", W.NormalRandomWeight, NS(mean=1.0, std=0.3), N, None, None),
    ("w/lognormal_broad", W.LogNormalRandomWeight, NS(mean=1.0, sigma_ln=0.9, w_min=0.1, w_max=3.0), N, None, None),
    ("w/offset_scaled_normal", W.OffsetScaledNormalWeight, NS(offset=0.2, g_scale=0.5), N, None, None),
    ("w/distance_dependent", W.DistanceDependentWeight, NS(max_weight=2.0, decay_length=150.0), N, _coords(), None),
    ("d/constant", D.ConstantDelay, NS(value=1.5), N, None, None),
    ("d/distance_based", D.DistanceBasedDelay, NS(velocity=500.0, min_delay=1.0, max_delay=None, synaptic_delay=0.0), N, _coords(), None),
    ("d/type_based", D.TypeBasedDelay, NS(d_ee=1.0, d_ei=2.0, d_ie=3.0, d_ii=4.0), N, None, _Layout(N, N_EXC)),
    ("d/random", D.RandomDelay, NS(mean=2.0, std=1.0, min=0.5, max=3.0), N, None, None),
]


def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _run_connection(cls, cfg, n, coords, layout, seed=0):
    rng = np.random.RandomState(seed)
    rows, cols = cls(cfg, n, coords, rng, layout=layout).generate()
    return rows, cols, rng.random_sample(4)


def _run_values(cls, cfg, n, coords, layout, seed=0):
    rng = np.random.RandomState(seed)
    values = cls(cfg, n, coords, rng=rng, layout=layout).generate(_ROWS, _COLS)
    return values, rng.random_sample(4)


GOLDEN = {
    "constant_prob": "a744c8f14ad04d09",
    "constant_prob_self": "573fd09ed16e1419",
    "constant_prob_multiblock": "1623ff8953cb9ea5",
    "distance_based": "247e4af37cfd59b2",
    "gaussian_distance_type": "cd368322b9451b19",
    "prob_based_block_1": "076650fbbc95049a",
    "prob_based_block_2": "6a924915818bf0f2",
    "prob_based_block_5": "5c10116c7b699f25",
    "prob_based_block_axis": "88a772bc35cf41f1",
    "optional_connect": "170ed8c71fcfc6e3",
    "C.elegans": "6c3dfe92a433905b",
    "w/constant": "cec76c58d52daf0a",
    "w/normal_broad": "6fdf1ddb83daf8d8",
    "w/lognormal_broad": "f6ca22f0d74ebe5e",
    "w/offset_scaled_normal": "a10f9dfb17a20e98",
    "w/distance_dependent": "36dd530aa98f837e",
    "d/constant": "d735af354b0a8e35",
    "d/distance_based": "da1d7d465649ccb3",
    "d/type_based": "d5e5184cd2f07ba1",
    "d/random": "33f37717135024a3",
}


@pytest.mark.parametrize("name,cls,cfg,n,coords,layout", CONNECTIONS, ids=[c[0] for c in CONNECTIONS])
def test_connection_is_row_major_unique_int32(name, cls, cfg, n, coords, layout):
    rows, cols, _ = _run_connection(cls, cfg, n, coords, layout)
    assert rows.dtype == np.int32 and cols.dtype == np.int32
    assert rows.size == cols.size > 0
    keys = rows.astype(np.int64) * n + cols
    assert np.all(np.diff(keys) > 0), "row 昇順・row 内 col 昇順・重複なし であること"


@pytest.mark.parametrize("name,cls,cfg,n,coords,layout", CONNECTIONS, ids=[c[0] for c in CONNECTIONS])
def test_connection_realization_is_fixed(name, cls, cfg, n, coords, layout):
    assert _digest(*_run_connection(cls, cfg, n, coords, layout)) == GOLDEN[name]


@pytest.mark.parametrize("name,cls,cfg,n,coords,layout", VALUES, ids=[v[0] for v in VALUES])
def test_values_are_one_float32_per_synapse(name, cls, cfg, n, coords, layout):
    values, _ = _run_values(cls, cfg, n, coords, layout)
    assert values.dtype == np.float32
    assert values.shape == _ROWS.shape


@pytest.mark.parametrize("name,cls,cfg,n,coords,layout", VALUES, ids=[v[0] for v in VALUES])
def test_values_realization_is_fixed(name, cls, cfg, n, coords, layout):
    assert _digest(*_run_values(cls, cfg, n, coords, layout)) == GOLDEN[name]


@pytest.mark.parametrize("name,cls,cfg,n,coords,layout", VALUES, ids=[v[0] for v in VALUES])
def test_values_accept_empty_coo(name, cls, cfg, n, coords, layout):
    empty = np.zeros(0, dtype=np.int32)
    values = cls(cfg, n, coords, rng=np.random.RandomState(0), layout=layout).generate(empty, empty)
    assert values.dtype == np.float32 and values.size == 0


def test_c_elegans_weight_keeps_synapses_without_source_weight():
    """マスクにあるが選択ソースに接続の無い要素は、重み 0 の結合として残る。"""
    rows = np.zeros(81, dtype=np.int32)          # ニューロン 0 から全ニューロンへ
    cols = np.arange(81, dtype=np.int32)
    cfg = NS(min_weight=1.0, max_weight=10.0, sources="both", priority="elec", normalize=True)
    values = W.C_elegansWeight(cfg, 81, None, rng=None).generate(rows, cols)
    assert values.shape == (81,)
    assert np.any(values == 0.0) and np.any(values > 0.0)
    assert np.all(values <= 1.0)                 # normalize=True で max_weight が 1.0 に写る


def test_type_based_delay_zero_for_unclassified_neurons():
    """E/I どちらにも属さないニューロンを端点に持つ結合の遅延は 0。"""
    layout = NS(ids_by=lambda axis: {"excitatory": np.array([0, 1]), "inhibitory": np.array([2])})
    cfg = NS(d_ee=1.0, d_ei=2.0, d_ie=3.0, d_ii=4.0)
    rows = np.array([0, 0, 2, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 0], dtype=np.int32)
    values = D.TypeBasedDelay(cfg, 4, None, layout=layout).generate(rows, cols)
    np.testing.assert_array_equal(values, np.array([1.0, 2.0, 3.0, 0.0], dtype=np.float32))


if __name__ == "__main__":
    # GOLDEN を作り直すときの出力 (実現値を意図して変えたときだけ使う)
    for name, cls, cfg, n, coords, layout in CONNECTIONS:
        print(f'    "{name}": "{_digest(*_run_connection(cls, cfg, n, coords, layout))}",')
    for name, cls, cfg, n, coords, layout in VALUES:
        print(f'    "{name}": "{_digest(*_run_values(cls, cfg, n, coords, layout))}",')
