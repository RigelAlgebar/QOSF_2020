import numpy as np

from Task_1.utils import (
    _split_angle_vector,
    _flatten_angles,
    _reshape_optimized_angles,
    make_random_phi,
    simulation,
)


def test_angle_roundtrip_shapes():
    angles = np.arange(16, dtype=float)
    odd, even, layers = _split_angle_vector(angles)
    assert layers == 2
    assert len(odd) == len(even) == 2

    flat = _flatten_angles(1, odd, even)
    reshaped_odd, reshaped_even = _reshape_optimized_angles(flat)
    assert len(reshaped_odd) == len(reshaped_even) == 2


def test_random_phi_reproducible_and_normalized():
    a = make_random_phi(seed=101)
    b = make_random_phi(seed=101)
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)


def test_all_cases_build():
    odd = [np.zeros(4), np.zeros(4)]
    even = [np.zeros(4), np.zeros(4)]
    for case in range(1, 10):
        qc = simulation(2, odd, even, case).build_case(case)
        assert qc.num_qubits == 4
