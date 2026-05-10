# Mathematical imports
from scipy.optimize import minimize
import numpy as np
from math import pi

# Quiskit imports
from qiskit import QuantumCircuit, QuantumRegister, execute
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer

# Import this to see the progress of the simulations
from tqdm import trange


# Define the backend
backend = BasicAer.get_backend('statevector_simulator')

# Here we set the seed for the random number generator (for reproducibility)
np.random.seed(101)

# This is the random "PHI" state we optimize against with
phi = 2 * pi * np.random.random(16) + 2 * pi * np.random.random(16) * 1j
phi = phi / np.linalg.norm(phi)


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


def _split_angle_vector(angles):
    """Split a flat angle vector into odd/even blocks and infer layer count."""
    odd_angles = [angles[i:i + 4] for i in range(0, len(angles), 8)]
    even_angles = [angles[i:i + 4] for i in range(4, len(angles), 8)]
    layers = len(angles) // 8
    return odd_angles, even_angles, layers


def _flatten_angles(layer, odd_block_angles, even_block_angles):
    """Pack odd/even angles into the vector format expected by scipy.minimize."""
    return np.hstack([odd_block_angles[:layer + 1], even_block_angles[:layer + 1]]).flatten()


def _reshape_optimized_angles(opt_vector):
    """Restore minimize output into [layers][4 angles] for odd/even blocks."""
    new_odd_angles = [opt_vector[i:i + 4] for i in range(0, len(opt_vector), 8)]
    new_even_angles = [opt_vector[i:i + 4] for i in range(4, len(opt_vector), 8)]
    return [new_odd_angles, new_even_angles]


def _run_objective(angles, case_num):
    q_trial = QuantumRegister(4)
    qc_trial = QuantumCircuit(q_trial)

    odd_angles, even_angles, layers = _split_angle_vector(angles)
    sim = simulation(q_trial, qc_trial, layers, odd_angles, even_angles, case_num)
    trial_circuit = sim.build_case(case_num)
    state_trial = execute(trial_circuit, backend).result().get_statevector()
    return np.linalg.norm(state_trial - phi)


def _optimize_case(layer, odd_block_angles, even_block_angles, case_num):
    angles = _flatten_angles(layer, odd_block_angles, even_block_angles)
    bounds = tuple((0, 2.0 * pi) for _ in range(len(angles)))
    result = minimize(
        lambda x: _run_objective(x, case_num),
        angles,
        method='L-BFGS-B',
        bounds=bounds,
    )
    return _reshape_optimized_angles(result.x)


class oddBlock:
    """Adds a series of four single-qubit rotations on the selected axis."""

    @staticmethod
    def _add_block(axis, q, qc, vector_angles):
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

    def __init__(self, q, qc, layers, vector_oddAngles, vector_evenAngles, caseNum):
        self.q = QuantumRegister(4)
        self.qc = QuantumCircuit(self.q)
        self.layers = layers
        self.angles_odd = vector_oddAngles
        self.angles_even = vector_evenAngles
        self.caseNum = caseNum

    def _build(self, odd_axis, even_axis):
        odd_builder = getattr(oddBlock, f"addBlock_{odd_axis}axis")
        even_builder = getattr(evenBlock, f"addBlock_{even_axis}axis")

        for layer_idx in range(self.layers):
            self.qc = odd_builder(self.q, self.qc, self.angles_odd[layer_idx])
            self.qc = even_builder(self.q, self.qc, self.angles_even[layer_idx])
        return self.qc

    def build_case(self, case_num):
        odd_axis, even_axis = self._CASE_AXES[case_num]
        return self._build(odd_axis, even_axis)

    def build_case1(self):
        return self.build_case(1)

    def build_case2(self):
        return self.build_case(2)

    def build_case3(self):
        return self.build_case(3)

    def build_case4(self):
        return self.build_case(4)

    def build_case5(self):
        return self.build_case(5)

    def build_case6(self):
        return self.build_case(6)

    def build_case7(self):
        return self.build_case(7)

    def build_case8(self):
        return self.build_case(8)

    def build_case9(self):
        return self.build_case(9)


def objective_case1(angles):
    return _run_objective(angles, 1)


def objective_case2(angles):
    return _run_objective(angles, 2)


def objective_case3(angles):
    return _run_objective(angles, 3)


def optimization_case1(layer, odd_block_angles, even_block_angles):
    return _optimize_case(layer, odd_block_angles, even_block_angles, 1)


def optimization_case2(layer, odd_block_angles, even_block_angles):
    return _optimize_case(layer, odd_block_angles, even_block_angles, 2)


def optimization_case3(layer, odd_block_angles, even_block_angles):
    return _optimize_case(layer, odd_block_angles, even_block_angles, 3)
