"""`NetworkBuilder.global_coo()` が疎/密の生成経路の違いを吸収していることの担保。

これがビルド以降を COO 一本にできる根拠そのもの。ここが崩れると、`src/utils` や
`scripts/` はどちらの経路で生成されたかによって別の結果を出すようになる。
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

# 疎と密の両方を実装しているコンポーネントの組み合わせ (connection: beggs_plenz)。
_CONFIG = root_path / "configs" / "criticality_test2.yaml"
_NUM_EXC = 48
_NUM_INH = 12
# 元 config は N=3600 を半径 3000 um の円に置く。N だけ縮めると σ_ee (300 um) に対して
# 疎になりすぎて結合がほぼ消えるので、密度を保つよう半径も sqrt(N) 比で縮める
# (縮小版が元と別物になるのは既知の落とし穴)。
_ORIG_N = 3600
_ORIG_R = 3000.0


def _build(sparse_mode: str):
    """同じ seed で 1 度ビルドし、正規化済みの COO を返す。"""
    config = ConfigManager().resolve(str(_CONFIG), "beggs_plenz_smoke")
    # 単体テストとして回る規模へ縮める。密経路が (N,N) を 3 本作るので N は小さく保つ。
    names = list(config.neurons)
    config.neurons[names[0]].num = _NUM_EXC
    config.neurons[names[1]].num = _NUM_INH
    total = _NUM_EXC + _NUM_INH
    config.simulation.N = total
    config.network.space.r = _ORIG_R * (total / _ORIG_N) ** 0.5
    config.simulation.seed = 12345
    config.simulation.backend = "cpu"
    config.network.sparse = sparse_mode

    builder = NetworkBuilder(config)
    builder._generate_global_matrices()
    return builder


class GlobalCooTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sparse = _build("on")
        cls.dense = _build("off")

    def test_both_paths_were_actually_taken(self):
        """前提の確認。両方とも疎で走ってしまうとこのテストは何も検証しない。"""
        self.assertTrue(self.sparse.is_sparse)
        self.assertFalse(self.dense.is_sparse)

    def test_sparse_and_dense_agree(self):
        """同じ seed なら、生成経路が違っても COO は値まで完全に一致する。"""
        a, b = self.sparse.global_coo(), self.dense.global_coo()

        self.assertEqual(a.shape, b.shape)
        # 数本しか繋がっていない実現では「たまたま一致した」と区別できない。
        self.assertGreater(a.num_synapses, 100, "結合が少なすぎて比較にならない")
        np.testing.assert_array_equal(a.row, b.row)
        np.testing.assert_array_equal(a.col, b.col)
        np.testing.assert_allclose(a.weights, b.weights)
        np.testing.assert_allclose(a.delays, b.delays)

    def test_coo_is_row_major_sorted(self):
        """`_pair_coo()` が GeNN の期待する並びを得るために依存している不変条件。"""
        for name, builder in (("sparse", self.sparse), ("dense", self.dense)):
            with self.subTest(path=name):
                coo = builder.global_coo()
                keys = coo.row.astype(np.int64) * coo.shape[1] + coo.col
                self.assertTrue(np.all(np.diff(keys) > 0),
                                "row 昇順・row 内 col 昇順・重複なし であること")

    def test_dense_path_keeps_zero_weight_synapses(self):
        """密経路の「結合がある」判定は重みではなくマスク。重み 0 でも結合は結合。"""
        mask_nnz = int(np.count_nonzero(np.asarray(self.dense._global_mask)))
        self.assertEqual(self.dense.global_coo().num_synapses, mask_nnz)

    def test_global_coo_before_generation_raises(self):
        config = ConfigManager().resolve(str(_CONFIG), "beggs_plenz_smoke")
        config.simulation.backend = "cpu"
        with self.assertRaises(RuntimeError):
            NetworkBuilder(config).global_coo()


if __name__ == "__main__":
    unittest.main()
