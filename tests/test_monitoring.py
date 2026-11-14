import json

import pytest

from monitoring import FEATURE_PSI, DriftMonitor, bin_fractions, psi


def test_psi_is_zero_for_identical_distributions():
    assert psi([0.25, 0.25, 0.5], [0.25, 0.25, 0.5]) == pytest.approx(0)


def test_psi_grows_with_shift():
    expected = [0.25, 0.25, 0.25, 0.25]
    assert psi(expected, [0.1, 0.1, 0.1, 0.7]) > psi(expected, [0.2, 0.3, 0.25, 0.25]) > 0


def test_bin_fractions():
    assert bin_fractions([1, 5, 10, 20], [5, 10]) == [0.5, 0.25, 0.25]


def test_drift_monitor_detects_shift(tmp_path):
    reference = {
        "area": {
            "mean": 50.0,
            "std": 10.0,
            "edges": [40.0, 50.0, 60.0],
            "fractions": [0.25, 0.25, 0.25, 0.25],
        }
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")

    monitor = DriftMonitor(reference_path=path, window_size=40)
    for value in [35, 45, 55, 65] * 10:
        monitor.observe({"area": value})
    stable = FEATURE_PSI.labels(feature="area")._value.get()

    for _ in range(40):
        monitor.observe({"area": 150})
    shifted = FEATURE_PSI.labels(feature="area")._value.get()

    assert stable < 0.1
    assert shifted > 0.25
