"""seed → ネットワーク実現の固定値 (golden)。

同じ config と seed から、いまのコードが**ビット単位で同じ**ネットワーク (結合・重み・
遅延・座標・軸索幾何) を作ることを確かめる。落ちるのは、乱数の消費順か生成の数式が
変わったとき。

**意図した変更で落ちたら**、期待値 (`EXPECTED`) を失敗メッセージの値で更新し、
**同じコミットで `docs/technical/compat_breaks.md` に追記する** (既存 run の
`replot --verify` が通らなくなり、lesion は親 run を組み直せなくなる)。

コンポーネント単体の実現値は `test/models/network/test_coo_generation.py` が固定している。
ここが見るのは、それらを NetworkBuilder が組み上げたもの (layout の ID 割当、
area → space → connection → weight → delay の順で 1 本の乱数列を共有する順序)。
"""
import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.core.config_manager import ConfigManager
from src.core.NetworkBuilder import NetworkBuilder

# @register デコレータを走らせる
import src.models.network.area  # noqa: F401,E402
import src.models.network.connectors  # noqa: F401,E402
import src.models.network.delays  # noqa: F401,E402
import src.models.network.space  # noqa: F401,E402
import src.models.network.weights  # noqa: F401,E402
import src.models.neurons.akita_escape_lif  # noqa: F401,E402
import src.models.neurons.akita_escape_lif_physical  # noqa: F401,E402

# 名前 -> config。task はネットワーク生成に効かないので汎用の smoke。
CASES = {
    # develop 型: 軸索伸長。領域・配置・重み・遅延のすべてが乱数を引く。
    "axon_growth": root_path / "test" / "network_golden.yaml",
    # akita_soc 型: no_space の全結合。乱数を引くのは layout の ID 割当 (random) だけ。
    "no_space": root_path / "test" / "experiments" / "no_space_100.yaml",
}

EXPECTED = {
    "axon_growth": {
        "num_synapses": 2293, "wiring": "9c45bcd0496cf89a", "weights": "ddef57dbe0438171",
        "delays": "1b243bc24a75af7b", "coords": "61f6e3d3bec5f426",
        "geometry": "47743522dab72329",
    },
    "no_space": {
        "num_synapses": 9900, "wiring": "027efa4e777a8b86", "weights": "433daa6185c417d3",
        "delays": "989ea72ac3ec0eed", "coords": "3aea65f81c858dd6",
    },
}


def _sha(*arrays) -> str:
    """配列の並び・dtype・値をまとめたハッシュ。dtype はリトルエンディアンに固定する。"""
    digest = hashlib.sha256()
    for array in arrays:
        array = np.asarray(array)
        digest.update(np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<")).tobytes())
    return digest.hexdigest()[:16]


def fingerprint(builder) -> dict:
    """ビルド済みネットワークの指紋。部品ごとに分けて、どこが変わったか読めるようにする。"""
    coo = builder.global_coo()
    result = {
        "num_synapses": int(coo.num_synapses),
        "wiring": _sha(coo.row.astype(np.int64), coo.col.astype(np.int64)),
        "weights": _sha(coo.weights.astype(np.float64)),
        "delays": _sha(coo.delays.astype(np.float64)),
        "coords": _sha(np.asarray(builder.global_coords, dtype=np.float64)),
    }
    geometry = getattr(builder.connection, "axon_geometry", lambda: None)()
    if geometry is not None:
        result["geometry"] = _sha(geometry.seg_start, geometry.seg_end, geometry.seg_owner,
                                  geometry.offsets, geometry.pre, geometry.post,
                                  geometry.contact_seg, geometry.contact_t)
    return result


def build(name: str):
    """結合を生成しただけの builder (GeNN へは登録しない)。"""
    config = ConfigManager().resolve(str(CASES[name]), "smoke")
    config.simulation.backend = "cpu"
    builder = NetworkBuilder(config)
    builder._generate_global_matrices()
    return builder


class NetworkGoldenTest(unittest.TestCase):
    def test_same_seed_gives_the_same_network(self):
        for name in CASES:
            with self.subTest(case=name):
                actual = fingerprint(build(name))
                self.assertEqual(
                    actual, EXPECTED[name],
                    f"\n{name}: seed から出るネットワークが変わりました。\n"
                    f"  意図した変更なら EXPECTED[{name!r}] を次の値に更新し、"
                    f" docs/technical/compat_breaks.md に追記してください:\n  {actual}")


if __name__ == "__main__":
    for case in CASES:
        print(case, fingerprint(build(case)))
