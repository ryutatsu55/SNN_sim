import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Add project root to sys.path for pytest execution.
root_path = Path(__file__).resolve().parents[3]
sys.path.append(str(root_path))

from src.models.network.connectors import BlockRandomTopology


class ConfigDict(dict[str, Any]):
    # Provide attribute-style access while preserving dict typing.
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def make_config(
    num_modules,
    within_module_connection_prob=1.0,
    between_module_connection_prob=0.0,
    allow_self_connections=False,
) -> ConfigDict:
    return ConfigDict(
        num_modules=num_modules,
        within_module_connection_prob=within_module_connection_prob,
        between_module_connection_prob=between_module_connection_prob,
        allow_self_connections=allow_self_connections,
    )


def test_generate_raises_when_num_modules_exceeds_num_neurons():
    topology = BlockRandomTopology(
        config=make_config(num_modules=5),
        num_neurons=4,
        coords=None,
        rng=np.random.RandomState(0),
    )

    with pytest.raises(ValueError, match="must not exceed num_neurons"):
        topology.generate()


def test_generate_raises_when_num_modules_is_not_integer():
    topology = BlockRandomTopology(
        config=make_config(num_modules=2.5),
        num_neurons=4,
        coords=None,
        rng=np.random.RandomState(0),
    )

    with pytest.raises(ValueError, match="must be an integer"):
        topology.generate()


def test_generate_raises_when_within_module_connection_prob_is_not_number():
    topology = BlockRandomTopology(
        config=make_config(num_modules=2, within_module_connection_prob="0.1"), # type: ignore
        num_neurons=4,
        coords=None,
        rng=np.random.RandomState(0),
    )

    with pytest.raises(ValueError, match="within_module_connection_prob must be a real number"):
        topology.generate()


def test_generate_raises_when_between_module_connection_prob_is_out_of_range():
    topology = BlockRandomTopology(
        config=make_config(num_modules=2, between_module_connection_prob=1.5),
        num_neurons=4,
        coords=None,
        rng=np.random.RandomState(0),
    )

    with pytest.raises(ValueError, match="between_module_connection_prob must be between 0.0 and 1.0"):
        topology.generate()


def test_generate_raises_when_allow_self_connections_is_not_bool():
    topology = BlockRandomTopology(
        config=make_config(num_modules=2, allow_self_connections=1), # type: ignore
        num_neurons=4,
        coords=None,
        rng=np.random.RandomState(0),
    )

    with pytest.raises(ValueError, match="allow_self_connections must be a boolean"):
        topology.generate()


def test_generate_accepts_num_modules_equal_to_num_neurons():
    topology = BlockRandomTopology(
        config=make_config(num_modules=4),
        num_neurons=4,
        coords=None,
        rng=np.random.RandomState(0),
    )

    mask = topology.generate()

    assert mask.shape == (4, 4)
    assert np.all(np.diag(mask) == 0)
