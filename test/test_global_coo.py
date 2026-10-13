"""`NetworkBuilder.global_coo()` が返す `GlobalCOO` の不変条件。

ビルド以降 (GeNN 登録・`src/utils`・`scripts/`) はこの形しか見ないので、ここが崩れると
`_pair_coo()` が GeNN の期待する並びを作れなくなる。
"""
import os
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

# task プロファイルはネットワーク生成に効かないので、汎用の smoke で足りる。
_CONFIG = root_path / "configs" / "criticality_test2.yaml"
_NUM_EXC = 48
_NUM_INH = 12
# 元 config は N=3600 を半径 3000 um の円に置く。N だけ縮めると σ_ee (300 um) に対して
# 疎になりすぎて結合がほぼ消えるので、密度を保つよう半径も sqrt(N) 比で縮める
# (縮小版が元と別物になるのは既知の落とし穴)。
_ORIG_N = 3600
_ORIG_R = 3000.0


def _build():
    """結合を生成しただけの builder を返す (GeNN へは登録しない)。"""
    config = ConfigManager().resolve(str(_CONFIG), "smoke")
    # 単体テストとして回る規模へ縮める。
    names = list(config.neurons)
    config.neurons[names[0]].num = _NUM_EXC
    config.neurons[names[1]].num = _NUM_INH
    total = _NUM_EXC + _NUM_INH
    config.simulation.N = total
    config.network.space.r = _ORIG_R * (total / _ORIG_N) ** 0.5
    config.simulation.seed = 12345
    config.simulation.backend = "cpu"

    builder = NetworkBuilder(config)
    builder._generate_global_matrices()
    return builder


class GlobalCooTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = _build()

    def test_coo_is_row_major_sorted(self):
        """`_pair_coo()` が GeNN の期待する並びを得るために依存している不変条件。"""
        coo = self.builder.global_coo()
        # 数本しか繋がっていない実現では何も検証しない
        self.assertGreater(coo.num_synapses, 100, "結合が少なすぎて検証にならない")
        keys = coo.row.astype(np.int64) * coo.shape[1] + coo.col
        self.assertTrue(np.all(np.diff(keys) > 0),
                        "row 昇順・row 内 col 昇順・重複なし であること")

    def test_coo_dtypes_and_lengths(self):
        coo = self.builder.global_coo()
        self.assertEqual(coo.shape, (self.builder.total_neurons, self.builder.total_neurons))
        self.assertEqual(coo.row.dtype, np.int32)
        self.assertEqual(coo.col.dtype, np.int32)
        self.assertEqual(coo.weights.dtype, np.float64)
        self.assertEqual(coo.delays.dtype, np.float64)
        self.assertEqual({coo.row.size, coo.col.size, coo.weights.size, coo.delays.size},
                         {coo.num_synapses})

    def test_global_coo_before_generation_raises(self):
        config = ConfigManager().resolve(str(_CONFIG), "smoke")
        config.simulation.backend = "cpu"
        with self.assertRaises(RuntimeError):
            NetworkBuilder(config).global_coo()


if __name__ == "__main__":
    unittest.main()
