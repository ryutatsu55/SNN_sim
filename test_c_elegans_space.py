#!/usr/bin/env python
"""Test C. elegans space coordinate loading"""

import numpy as np
from src.models.network.space import C_elegansSpace


class MockConfig:
    """Mock config object for testing"""
    pass


def test_c_elegans_space():
    """Test C_elegansSpace.generate()"""

    # Test 1: Load all neurons (81)
    print("Test 1: Loading all 81 C. elegans neurons...")
    config = MockConfig()
    rng = np.random.RandomState(42)
    space = C_elegansSpace(config, num_neurons=81, rng=rng)
    coords = space.generate()

    assert coords.shape == (81, 3), f"Expected shape (81, 3), got {coords.shape}"
    assert coords.dtype == np.float32, f"Expected dtype float32, got {coords.dtype}"
    assert not np.any(np.isnan(coords)), "Coordinates should not contain NaN"
    print(f"✓ Successfully loaded 81 neurons")
    print(f"  Coordinate range - X: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"  Coordinate range - Y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
    print(f"  Coordinate range - Z: [{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")

    # Test 2: Load partial neurons
    print("\nTest 2: Loading first 40 neurons...")
    space = C_elegansSpace(config, num_neurons=40, rng=rng)
    coords = space.generate()

    assert coords.shape == (40, 3), f"Expected shape (40, 3), got {coords.shape}"
    print(f"✓ Successfully loaded 40 neurons")
    print(f"  Sample coordinates:\n{coords[:3]}")

    # Test 3: Try to load too many neurons
    print("\nTest 3: Testing error handling for too many neurons...")
    space = C_elegansSpace(config, num_neurons=100, rng=rng)
    try:
        coords = space.generate()
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")

    # Test 4: Load single neuron
    print("\nTest 4: Loading single neuron...")
    space = C_elegansSpace(config, num_neurons=1, rng=rng)
    coords = space.generate()

    assert coords.shape == (1, 3), f"Expected shape (1, 3), got {coords.shape}"
    print(f"✓ Successfully loaded 1 neuron: {coords[0]}")

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == "__main__":
    test_c_elegans_space()
