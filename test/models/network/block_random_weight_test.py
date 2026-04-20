import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Add project root to sys.path for pytest execution.
root_path = Path(__file__).resolve().parents[3]
sys.path.append(str(root_path))

from src.models.network.weights import BlockRandomWeight


class ConfigDict(dict[str, Any]):
    # Provide attribute-style access while preserving dict typing.
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def make_config(offset=1.0, g_scale=0.5) -> ConfigDict:
    return ConfigDict(offset=offset, g_scale=g_scale)


def make_mask() -> np.ndarray:
    return np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.int8,
    )


def test_generate_raises_when_offset_is_not_number():
    weight = BlockRandomWeight(
        config=make_config(offset="1.0"),  # type: ignore
        num_neurons=3,
        coords=None,
        mask=make_mask(),
        rng=np.random.RandomState(0),
    )

    with pytest.raises(ValueError, match="offset must be a real number"):
        weight.generate()


def test_generate_raises_when_g_scale_is_not_number():
    weight = BlockRandomWeight(
        config=make_config(g_scale="0.5"),  # type: ignore
        num_neurons=3,
        coords=None,
        mask=make_mask(),
        rng=np.random.RandomState(0),
    )

    with pytest.raises(ValueError, match="g_scale must be a real number"):
        weight.generate()


def test_generate_raises_when_g_scale_is_negative():
    weight = BlockRandomWeight(
        config=make_config(g_scale=-0.1),
        num_neurons=3,
        coords=None,
        mask=make_mask(),
        rng=np.random.RandomState(0),
    )

    with pytest.raises(ValueError, match="g_scale must be greater than or equal to 0.0"):
        weight.generate()


def test_generate_accepts_zero_g_scale():
    weight = BlockRandomWeight(
        config=make_config(offset=1.2, g_scale=0.0),
        num_neurons=3,
        coords=None,
        mask=make_mask(),
        rng=np.random.RandomState(0),
    )

    weights = weight.generate()

    expected = np.array(
        [
            [0.0, 1.2, 0.0],
            [1.2, 0.0, 1.2],
            [0.0, 1.2, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(weights, expected)
