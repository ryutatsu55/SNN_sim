import numpy as np
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.utils.visualize import export_spike_csv  # noqa: E402


def test_export_spike_csv_saves_expected_rows(tmp_path):
    output_path = tmp_path / "spikes.csv"

    export_spike_csv(
        spike_time=np.array([1.5, 2.0], dtype=float),
        neuron_id=np.array([3, 7], dtype=int),
        output_path=str(output_path),
    )

    lines = output_path.read_text().splitlines()
    assert lines[0] == "spike_time,neuron_id"
    assert lines[1] == "1.5000000000,3"
    assert lines[2] == "2.0000000000,7"


def test_export_spike_csv_raises_when_lengths_do_not_match(tmp_path):
    output_path = tmp_path / "spikes.csv"

    try:
        export_spike_csv(
            spike_time=np.array([1.0, 2.0], dtype=float),
            neuron_id=np.array([1], dtype=int),
            output_path=str(output_path),
        )
    except ValueError as exc:
        assert str(exc) == "spike_time and neuron_id arrays must have the same length."
        return

    raise AssertionError("ValueError was not raised for mismatched array lengths.")
