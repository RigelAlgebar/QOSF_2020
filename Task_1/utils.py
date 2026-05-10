from dataclasses import dataclass
from math import pi

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector


@dataclass(frozen=True)
class SimulationConfig:
    """Runtime configuration for objective/optimization evaluation."""

    phi: np.ndarray


def make_random_phi(seed=101, n_qubits=4):
    """Create a reproducible normalized random state vector."""
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    vec = 2 * pi * rng.random(dim) + 2 * pi * rng.random(dim) * 1j
    return vec / np.linalg.norm(vec)


# Backward-compatible default target state
DEFAULT_CONFIG = SimulationConfig(phi=make_random_phi(seed=101, n_qubits=4))


_ROTATION_GATES = {
    "x": QuantumCircuit.rx,
    "y": QuantumCircuit.ry,
    "z": QuantumCircuit.rz,
}

_CONTROLLED_GATES = {
    "x": QuantumCircuit.cx,
    "y": QuantumCircuit.cy,
    "z": QuantumCircuit.cz,
}

_PAIR_INDICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


# Backward-compatible public constant
phi = DEFAULT_CONFIG.phi


def _validate_flat_angles(angles):
    if len(angles) == 0 or len(angles) % 8 != 0:
        raise ValueError("`angles` length must be a non-zero multiple of 8.")


def _validate_layer_inputs(layer, odd_block_angles, even_block_angles):
    if layer < 0:
        raise ValueError("`layer` must be >= 0.")
    req = layer + 1
    if len(odd_block_angles) < req or len(even_block_angles) < req:
        raise ValueError("Not enough odd/even angle blocks for the requested layer.")


def _split_angle_vector(angles):
    """Split a flat angle vector into odd/even blocks and infer layer count."""
    _validate_flat_angles(angles)
    odd_angles = [angles[i:i + 4] for i in range(0, len(angles), 8)]
    even_angles = [angles[i:i + 4] for i in range(4, len(angles), 8)]
    layers = len(angles) // 8
    return odd_angles, even_angles, layers


def _flatten_angles(layer, odd_block_angles, even_block_angles):
    """Pack odd/even angles into the vector format expected by scipy.minimize."""
    _validate_layer_inputs(layer, odd_block_angles, even_block_angles)
    return np.hstack([odd_block_angles[:layer + 1], even_block_angles[:layer + 1]]).flatten()


def _reshape_optimized_angles(opt_vector):
    """Restore minimize output into [layers][4 angles] for odd/even blocks."""
    _validate_flat_angles(opt_vector)
    new_odd_angles = [opt_vector[i:i + 4] for i in range(0, len(opt_vector), 8)]
    new_even_angles = [opt_vector[i:i + 4] for i in range(4, len(opt_vector), 8)]
    return [new_odd_angles, new_even_angles]


def _run_objective(angles, case_num, config=None):
    cfg = DEFAULT_CONFIG if config is None else config
    odd_angles, even_angles, layers = _split_angle_vector(angles)
    sim = simulation(layers, odd_angles, even_angles, case_num)
    trial_circuit = sim.build_case(case_num)
    state_trial = Statevector.from_instruction(trial_circuit).data
    return np.linalg.norm(state_trial - cfg.phi)


def _optimize_case(layer, odd_block_angles, even_block_angles, case_num, config=None):
    angles = _flatten_angles(layer, odd_block_angles, even_block_angles)
    bounds = tuple((0, 2.0 * pi) for _ in range(len(angles)))
    result = minimize(
        lambda x: _run_objective(x, case_num, config=config),
        angles,
        method='L-BFGS-B',
        bounds=bounds,
    )
    return _reshape_optimized_angles(result.x)


class oddBlock:
    """Adds a series of four single-qubit rotations on the selected axis."""

    @staticmethod
    def _add_block(axis, q, qc, vector_angles):
        if len(vector_angles) != 4:
            raise ValueError("Each odd block must contain exactly 4 angles.")
        rot_gate = _ROTATION_GATES[axis]
        for qubit, angle in enumerate(vector_angles):
            rot_gate(qc, angle, q[qubit])
        return qc

    @staticmethod
    def addBlock_xaxis(q, qc, vector_angles):
        return oddBlock._add_block("x", q, qc, vector_angles)

    @staticmethod
    def addBlock_yaxis(q, qc, vector_angles):
        return oddBlock._add_block("y", q, qc, vector_angles)

    @staticmethod
    def addBlock_zaxis(q, qc, vector_angles):
        return oddBlock._add_block("z", q, qc, vector_angles)


class evenBlock:
    """Adds four rotations followed by all pair-wise controlled gates."""

    @staticmethod
    def _add_block(axis, q, qc, vector_angles):
        if len(vector_angles) != 4:
            raise ValueError("Each even block must contain exactly 4 angles.")
        rot_gate = _ROTATION_GATES[axis]
        ctrl_gate = _CONTROLLED_GATES[axis]

        for qubit, angle in enumerate(vector_angles):
            rot_gate(qc, angle, q[qubit])

        for control, target in _PAIR_INDICES:
            ctrl_gate(qc, q[control], q[target])

        return qc

    @staticmethod
    def addBlock_xaxis(q, qc, vector_angles):
        return evenBlock._add_block("x", q, qc, vector_angles)

    @staticmethod
    def addBlock_yaxis(q, qc, vector_angles):
        return evenBlock._add_block("y", q, qc, vector_angles)

    @staticmethod
    def addBlock_zaxis(q, qc, vector_angles):
        return evenBlock._add_block("z", q, qc, vector_angles)


class simulation:
    """Defines and builds parameterized circuits for all odd/even axis cases."""

    _CASE_AXES = {
        1: ("x", "x"), 2: ("x", "y"), 3: ("x", "z"),
        4: ("y", "x"), 5: ("y", "y"), 6: ("y", "z"),
        7: ("z", "x"), 8: ("z", "y"), 9: ("z", "z"),
    }

    def __init__(self, *args, **kwargs):
        """Support both legacy and simplified constructor signatures.

        Legacy: simulation(q, qc, layers, odd_angles, even_angles, case_num)
        New:    simulation(layers, odd_angles, even_angles, case_num=1)
        """
        if len(args) >= 6:
            _, _, layers, vector_odd_angles, vector_even_angles, case_num = args[:6]
        elif len(args) >= 3:
            layers, vector_odd_angles, vector_even_angles = args[:3]
            case_num = args[3] if len(args) >= 4 else kwargs.get("case_num", 1)
        else:
            raise TypeError("Invalid simulation constructor signature.")

        self.q = QuantumRegister(4)
        self.qc = QuantumCircuit(self.q)
        self.layers = layers
        self.angles_odd = vector_odd_angles
        self.angles_even = vector_even_angles
        self.case_num = case_num

        if layers <= 0:
            raise ValueError("`layers` must be > 0.")
        if len(vector_odd_angles) < layers or len(vector_even_angles) < layers:
            raise ValueError("Not enough odd/even angle blocks for `layers`.")

    def _build(self, odd_axis, even_axis):
        odd_builder = getattr(oddBlock, f"addBlock_{odd_axis}axis")
        even_builder = getattr(evenBlock, f"addBlock_{even_axis}axis")

        for layer_idx in range(self.layers):
            self.qc = odd_builder(self.q, self.qc, self.angles_odd[layer_idx])
            self.qc = even_builder(self.q, self.qc, self.angles_even[layer_idx])
        return self.qc

    def build_case(self, case_num):
        if case_num not in self._CASE_AXES:
            raise ValueError(f"Invalid case number: {case_num}. Must be 1..9.")
        odd_axis, even_axis = self._CASE_AXES[case_num]
        return self._build(odd_axis, even_axis)

    def build_case1(self): return self.build_case(1)
    def build_case2(self): return self.build_case(2)
    def build_case3(self): return self.build_case(3)
    def build_case4(self): return self.build_case(4)
    def build_case5(self): return self.build_case(5)
    def build_case6(self): return self.build_case(6)
    def build_case7(self): return self.build_case(7)
    def build_case8(self): return self.build_case(8)
    def build_case9(self): return self.build_case(9)


def objective_case1(angles, config=None):
    return _run_objective(angles, 1, config=config)


def objective_case2(angles, config=None):
    return _run_objective(angles, 2, config=config)


def objective_case3(angles, config=None):
    return _run_objective(angles, 3, config=config)


def optimization_case1(layer, odd_block_angles, even_block_angles, config=None):
    return _optimize_case(layer, odd_block_angles, even_block_angles, 1, config=config)


def optimization_case2(layer, odd_block_angles, even_block_angles, config=None):
    return _optimize_case(layer, odd_block_angles, even_block_angles, 2, config=config)


def optimization_case3(layer, odd_block_angles, even_block_angles, config=None):
    return _optimize_case(layer, odd_block_angles, even_block_angles, 3, config=config)
