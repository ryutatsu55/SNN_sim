"""`RandomDelay` の引数の検証とクリップ。

数でない・負の `std` と、`min` > `max` を弾く。`std` = 0 では遅延を指定範囲にクリップする。
"""
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Add project root to sys.path for pytest execution.
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.network.delays import RandomDelay


class ConfigDict(dict[str, Any]):
    # Provide attribute-style access while preserving dict typing.
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def make_coo() -> tuple[np.ndarray, np.ndarray]:
    """3 ニューロンで 0-1, 1-0, 1-2, 2-1 を結ぶ COO (行優先)。"""
    return np.array([0, 1, 1, 2], dtype=np.int32), np.array([1, 0, 2, 1], dtype=np.int32)


def test_random_delay_raises_when_std_is_not_number():
    delay = RandomDelay(
        config=ConfigDict(mean=7.0, std="1.0", min=4.0, max=10.0),  # type: ignore
        num_neurons=3,
        coords=None,
        rng=np.random.RandomState(0),
    )

    with pytest.raises(ValueError, match="std must be a real number"):
        delay.generate(*make_coo())


def test_random_delay_raises_when_std_is_negative():
    delay = RandomDelay(
        config=ConfigDict(mean=7.0, std=-1.0, min=4.0, max=10.0),
        num_neurons=3,
        coords=None,
        rng=np.random.RandomState(0),
    )

    with pytest.raises(ValueError, match="std must be greater than or equal to 0.0"):
        delay.generate(*make_coo())


def test_random_delay_raises_when_min_exceeds_max():
    delay = RandomDelay(
        config=ConfigDict(mean=7.0, std=1.0, min=11.0, max=10.0),
        num_neurons=3,
        coords=None,
        rng=np.random.RandomState(0),
    )

    with pytest.raises(ValueError, match="min must be less than or equal to max"):
        delay.generate(*make_coo())


def test_random_delay_accepts_zero_std_and_applies_clipping_bounds():
    delay = RandomDelay(
        config=ConfigDict(mean=7.0, std=0.0, min=4.0, max=10.0),
        num_neurons=3,
        coords=None,
        rng=np.random.RandomState(0),
    )

    delays = delay.generate(*make_coo())

    expected = np.full(4, 7.0, dtype=np.float32)
    np.testing.assert_array_equal(delays, expected)
