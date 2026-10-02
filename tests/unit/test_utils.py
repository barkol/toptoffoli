"""Unit tests for toffoli_optimizer.utils.circuit_utils (+ utils package API).

Every numeric claim is checked against an independent computation (plain
Python / qiskit), never against the module under test. Confirmed bugs are
marked ``xfail(strict=True)`` with a minimal reproducer.
"""

import itertools
import random

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Qubit
from qiskit.circuit.library import XGate, HGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap

from toffoli_optimizer.utils import circuit_utils as cu


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """No test may write into the repository: run each test inside tmp_path."""
    monkeypatch.chdir(tmp_path)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _edges(cmap):
    return {tuple(e) for e in cmap}


def _global_2q_pairs(circ):
    """Global (circuit-wide) qubit indices of every multi-qubit instruction."""
    out = []
    for inst in circ.data:
        if len(inst.qubits) >= 2 and inst.operation.name != "barrier":
            out.append(tuple(circ.find_bit(q).index for q in inst.qubits))
    return out


def _sample_circuit():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.ccx(0, 1, 3)
    qc.cx(0, 3)
    qc.t(2)
    qc.ccx(3, 2, 0)
    return qc


# --------------------------------------------------------------------------- #
# calculate_gate_fidelities
# --------------------------------------------------------------------------- #

class TestCalculateGateFidelities:
    def test_default_rates_match_independent_product(self):
        counts = {"cx": 3, "ccx": 2, "h": 5, "rz": 4, "u3": 7}
        rates = {"cx": 0.01, "ccx": 0.03, "h": 0.001, "rz": 0.0005}
        expected = 1.0
        for g, n in counts.items():
            expected *= (1 - rates.get(g, 0.001)) ** n
        assert cu.calculate_gate_fidelities(counts) == pytest.approx(expected, rel=1e-12)

    def test_empty_counts_is_one(self):
        assert cu.calculate_gate_fidelities({}) == 1.0

    def test_custom_rates_with_default(self):
        f = cu.calculate_gate_fidelities({"cx": 2, "foo": 1}, {"cx": 0.1, "default": 0.5})
        assert f == pytest.approx(0.9 ** 2 * 0.5)

    @pytest.mark.xfail(strict=True, reason="BUG: custom error_rates without a 'default' key "
                       "raise KeyError for any gate not listed (circuit_utils.py:49)")
    def test_custom_rates_without_default_key(self):
        assert cu.calculate_gate_fidelities({"h": 1}, {"cx": 0.1}) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# estimate_fidelity
# --------------------------------------------------------------------------- #

def _ref_fidelity(nq, nops, cx, t, depth):
    other = nops - cx - t
    f = 0.99 ** cx * 0.998 ** t * 0.999 ** other
    if depth is not None:
        f *= max(0.8, 1.0 - 0.001 * nq * np.log(1 + depth))
    return max(0.0, min(1.0, f))


class TestEstimateFidelity:
    def test_from_circuit_matches_model(self):
        qc = _sample_circuit()
        qc.tdg(1)
        ops = qc.count_ops()
        ref = _ref_fidelity(qc.num_qubits, sum(ops.values()), ops.get("cx", 0),
                            ops.get("t", 0) + ops.get("tdg", 0), qc.depth())
        assert cu.estimate_fidelity(qc) == pytest.approx(ref, rel=1e-12)

    def test_keyword_form(self):
        ref = _ref_fidelity(5, 100, int(100 * 0.2), int(100 * 0.1), None)
        assert cu.estimate_fidelity(num_qubits=5, num_operations=100) == pytest.approx(ref)

    def test_explicit_counts(self):
        ref = _ref_fidelity(3, 20, 5, 4, 10)
        got = cu.estimate_fidelity(num_qubits=3, num_operations=20, cx_count=5, t_count=4, depth=10)
        assert got == pytest.approx(ref)

    def test_monotone_in_cx(self):
        a = cu.estimate_fidelity(num_qubits=3, num_operations=50, cx_count=5, t_count=0)
        b = cu.estimate_fidelity(num_qubits=3, num_operations=50, cx_count=25, t_count=0)
        assert 0.0 <= b < a <= 1.0

    def test_empty_circuit_is_one(self):
        assert cu.estimate_fidelity(QuantumCircuit(2)) == pytest.approx(1.0)

    def test_non_circuit_object_returns_default(self):
        assert cu.estimate_fidelity(object()) == 0.9

    @pytest.mark.xfail(strict=True, reason="BUG: documented positional form "
                       "estimate_fidelity(num_qubits, num_operations) binds num_qubits to "
                       "`circuit` and silently returns the 0.9 placeholder (circuit_utils.py:80-82)")
    def test_documented_positional_form(self):
        assert cu.estimate_fidelity(5, 100) == pytest.approx(
            cu.estimate_fidelity(num_qubits=5, num_operations=100))


# --------------------------------------------------------------------------- #
# coupling maps
# --------------------------------------------------------------------------- #

class TestDefaultCouplingMap:
    @pytest.mark.parametrize("n", [1, 2, 5, 8])
    def test_linear(self, n):
        cmap = cu.get_default_coupling_map("linear", n)
        expected = {(i, i + 1) for i in range(n - 1)} | {(i + 1, i) for i in range(n - 1)}
        assert _edges(cmap) == expected
        assert len(cmap) == len(expected)

    @pytest.mark.parametrize("rows", [2, 3, 4])
    def test_grid_square_matches_qiskit(self, rows):
        n = rows * rows
        ref = {tuple(e) for e in CouplingMap.from_grid(rows, rows, bidirectional=True).get_edges()}
        assert _edges(cu.get_default_coupling_map("grid", n)) == ref

    def test_grid_ragged(self):
        # 5 qubits on a 3-wide grid: rows [0,1,2] / [3,4]
        und = {(0, 1), (1, 2), (3, 4), (0, 3), (1, 4)}
        expected = und | {(b, a) for a, b in und}
        assert _edges(cu.get_default_coupling_map("grid", 5)) == expected

    @pytest.mark.parametrize("n", [4, 7, 12])
    def test_falcon_properties(self, n):
        cmap = cu.get_default_coupling_map("falcon", n)
        e = _edges(cmap)
        assert len(e) == len(cmap), "duplicate edges"
        assert all(a != b and 0 <= a < n and 0 <= b < n for a, b in e)
        assert all((b, a) in e for a, b in e), "not symmetric"
        assert CouplingMap(cmap).is_connected()

    def test_unknown_topology_falls_back_to_linear(self, capsys):
        assert cu.get_default_coupling_map("heavyhex??", 4) == cu.get_default_coupling_map("linear", 4)
        assert "Unknown topology" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# physical mapping
# --------------------------------------------------------------------------- #

class TestPhysicalMapping:
    CM = cu.get_default_coupling_map("linear", 4)

    def _check_coupling(self, mapped):
        allowed = _edges(self.CM)
        for pair in _global_2q_pairs(mapped):
            assert len(pair) == 2 and pair in allowed, pair

    def test_naive_mapping_respects_coupling_and_is_equivalent(self):
        qc = _sample_circuit()
        mapped = cu.create_naive_physical_mapping(qc, self.CM)
        self._check_coupling(mapped)
        assert set(mapped.count_ops()) <= {"id", "rz", "sx", "x", "cx"}
        assert Operator.from_circuit(mapped).equiv(Operator(qc))

    def test_does_not_mutate_input(self):
        qc = _sample_circuit()
        before = qc.copy()
        cu.create_optimized_physical_mapping(qc, self.CM, optimization_level=1)
        assert qc == before

    def test_custom_basis(self):
        mapped = cu.create_optimized_physical_mapping(_sample_circuit(), self.CM,
                                                      basis_gates=["u", "cx"], optimization_level=1)
        assert set(mapped.count_ops()) <= {"u", "cx"}

    def test_none_circuit(self):
        assert cu.create_optimized_physical_mapping(None, self.CM) is None

    def test_level3_respects_coupling(self):
        mapped = cu.create_optimized_physical_mapping(_sample_circuit(), self.CM, optimization_level=3)
        self._check_coupling(mapped)

    @pytest.mark.xfail(strict=True, reason="BUG: second transpile pass (circuit_utils.py:210-217) "
                       "re-lays-out an already routed circuit; the first-pass layout/permutation is "
                       "dropped, so the returned circuit's layout metadata no longer describes the "
                       "input (Operator.from_circuit(result) != Operator(input))")
    def test_level3_equivalent_under_its_layout(self):
        qc = _sample_circuit()
        mapped = cu.create_optimized_physical_mapping(qc, self.CM, optimization_level=3)
        assert Operator.from_circuit(mapped).equiv(Operator(qc))

    def test_fallback_returns_original_when_impossible(self):
        # 5-qubit circuit cannot fit a 3-qubit device -> all transpiles fail.
        qc = QuantumCircuit(5)
        qc.cx(0, 4)
        out = cu.create_optimized_physical_mapping(qc, cu.get_default_coupling_map("linear", 3))
        assert out is qc


# --------------------------------------------------------------------------- #
# validate_physical_circuit / get_qubit_index
# --------------------------------------------------------------------------- #

class TestValidate:
    CM = [[0, 1], [1, 2], [2, 3]]

    def test_valid_circuit(self):
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(2, 1)
        qc.h(3)
        assert cu.validate_physical_circuit(qc, self.CM) == (True, [])

    def test_cx_violation(self):
        qc = QuantumCircuit(4)
        qc.cx(0, 2)
        ok, v = cu.validate_physical_circuit(qc, self.CM)
        assert not ok and len(v) == 1 and "0 and 2" in v[0]

    def test_ccx_pairwise_violations(self):
        qc = QuantumCircuit(4)
        qc.ccx(0, 1, 2)  # pairs (0,1) ok, (0,2) bad, (1,2) ok
        ok, v = cu.validate_physical_circuit(qc, self.CM)
        assert not ok and len(v) == 1

    def test_coupling_map_object(self):
        qc = QuantumCircuit(4)
        qc.cx(3, 2)
        assert cu.validate_physical_circuit(qc, CouplingMap(self.CM))[0]
        qc.cx(0, 3)
        assert not cu.validate_physical_circuit(qc, CouplingMap(self.CM))[0]

    def test_none_inputs(self):
        assert cu.validate_physical_circuit(None, self.CM)[0] is False
        assert cu.validate_physical_circuit(QuantumCircuit(1), None)[0] is False

    def test_invalid_coupling_format(self):
        ok, v = cu.validate_physical_circuit(QuantumCircuit(2), "not-a-map")
        assert not ok and v == ["Invalid coupling map format"]

    def test_transpiled_circuit_valid(self):
        mapped = transpile(_sample_circuit(), coupling_map=self.CM, seed_transpiler=1,
                           basis_gates=["cx", "rz", "sx", "x"])
        assert cu.validate_physical_circuit(mapped, self.CM)[0]

    @pytest.mark.xfail(strict=True, reason="BUG: uses Qubit._index (index inside its register, "
                       "not inside the circuit); for multi-register circuits a cx between global "
                       "qubits 0 and 3 is checked as (0,1) and passes (circuit_utils.py:315-317)")
    def test_multi_register_violation_detected(self):
        a, b = QuantumRegister(2, "a"), QuantumRegister(2, "b")
        qc = QuantumCircuit(a, b)
        qc.cx(a[0], b[1])  # global (0, 3): not coupled
        assert not cu.validate_physical_circuit(qc, self.CM)[0]


class TestGetQubitIndex:
    def test_single_register(self):
        qc = QuantumCircuit(5)
        assert [cu.get_qubit_index(q) for q in qc.qubits] == list(range(5))

    @pytest.mark.xfail(strict=True, reason="BUG: returns the register-local index "
                       "(Qubit._index) instead of the circuit index (circuit_utils.py:254-256)")
    def test_multi_register(self):
        a, b = QuantumRegister(2, "a"), QuantumRegister(2, "b")
        qc = QuantumCircuit(a, b)
        assert [cu.get_qubit_index(q) for q in qc.qubits] == [0, 1, 2, 3]

    @pytest.mark.xfail(strict=True, reason="BUG: register-less Qubit() has _index None; "
                       "function returns None instead of an int or ValueError")
    def test_bare_qubit(self):
        res = None
        try:
            res = cu.get_qubit_index(Qubit())
        except ValueError:
            return
        assert isinstance(res, int)


# --------------------------------------------------------------------------- #
# CircuitGateProcessor
# --------------------------------------------------------------------------- #

class TestCircuitGateProcessor:
    def test_controlled_x_is_toffoli(self):
        qc = QuantumCircuit(3)
        assert cu.CircuitGateProcessor().create_controlled_gate_safely(XGate(), [0, 1], [2], qc)
        ref = QuantumCircuit(3)
        ref.ccx(0, 1, 2)
        assert Operator(qc).equiv(Operator(ref))

    def test_out_of_range(self):
        qc = QuantumCircuit(3)
        assert not cu.CircuitGateProcessor(debug_mode=True).create_controlled_gate_safely(
            XGate(), [0, 5], [1], qc)
        assert len(qc.data) == 0

    def test_no_controls(self):
        qc = QuantumCircuit(2)
        assert not cu.CircuitGateProcessor().create_controlled_gate_safely(HGate(), [], [1], qc)

    def test_object_without_control(self):
        class NoCtl:
            name = "noctl"
        assert not cu.CircuitGateProcessor(debug_mode=True).create_controlled_gate_safely(
            NoCtl(), [0], [1], QuantumCircuit(2))

    def test_too_many_controls_are_truncated(self):
        # Documented "try with fewer": 3 controls on a 3-qubit circuit -> 2 controls.
        qc = QuantumCircuit(3)
        assert cu.CircuitGateProcessor().create_controlled_gate_safely(XGate(), [0, 1, 2], [2], qc)
        assert qc.data[0].operation.num_ctrl_qubits == 2

    def test_process_preserves_single_register_circuit(self):
        qc = QuantumCircuit(4, 2)
        qc.h(0)
        qc.ccx(0, 1, 3)
        qc.cx(3, 2)
        qc.rz(0.3, 1)
        qc.swap(1, 2)
        qc.measure(0, 1)
        out = cu.CircuitGateProcessor().process_circuit_safely(qc)
        assert out == qc

    def test_process_does_not_mutate_input(self):
        qc = _sample_circuit()
        before = qc.copy()
        cu.CircuitGateProcessor().process_circuit_safely(qc, [[0, 1], [1, 2], [2, 3]])
        assert qc == before

    def test_process_unitary_equivalence(self):
        qc = _sample_circuit()
        out = cu.CircuitGateProcessor().process_circuit_safely(qc)
        assert Operator(out).equiv(Operator(qc))

    @pytest.mark.xfail(strict=True, reason="BUG: register-local Qubit._index used to rebuild the "
                       "circuit; gates on register b are silently moved onto register a "
                       "(circuit_utils.py:545, 648) -> different unitary")
    def test_process_multi_register(self):
        a, b = QuantumRegister(2, "a"), QuantumRegister(2, "b")
        qc = QuantumCircuit(a, b)
        qc.cx(a[0], b[1])
        qc.h(b[0])
        out = cu.CircuitGateProcessor().process_circuit_safely(qc)
        assert Operator(out).equiv(Operator(qc))

    @pytest.mark.xfail(strict=True, reason="BUG: gates whose name lacks the letter 'c' and violate "
                       "the coupling map are silently DROPPED (circuit_utils.py:597-645), returning "
                       "a non-equivalent circuit without any error")
    def test_process_does_not_silently_drop_gates(self):
        qc = QuantumCircuit(3)
        qc.swap(0, 2)
        out = cu.CircuitGateProcessor().process_circuit_safely(qc, [[0, 1], [1, 2]])
        assert Operator(out).equiv(Operator(qc))

    @pytest.mark.xfail(strict=True, reason="BUG: any gate whose name contains 'c' (cx, cz, ecr...) "
                       "takes the control-count branch and the coupling check in the `elif` is never "
                       "reached; a cx on uncoupled qubits passes through unflagged/unfixed "
                       "(circuit_utils.py:575-597)")
    def test_process_cx_coupling_enforced(self):
        qc = QuantumCircuit(3)
        qc.cx(0, 2)
        out = cu.CircuitGateProcessor().process_circuit_safely(qc, [[0, 1], [1, 2]])
        assert cu.validate_physical_circuit(out, [[0, 1], [1, 2]])[0]


# --------------------------------------------------------------------------- #
# random network / benchmark generators
# --------------------------------------------------------------------------- #

class TestGenerators:
    @pytest.mark.parametrize("seed", [0, 7, 123])
    def test_random_network_well_formed_and_deterministic(self, seed):
        a = cu.generate_random_toffoli_network(num_gates=12, target_logical_depth=3, num_qubits=8, seed=seed)
        b = cu.generate_random_toffoli_network(num_gates=12, target_logical_depth=3, num_qubits=8, seed=seed)
        assert a == b
        gates, outs, ins, n = a
        assert n == 8 and len(gates) == 12
        for controls, target in gates:
            assert len(controls) == 2 and len(set(controls)) == 2
            assert target not in controls
            assert all(0 <= q < 8 for q in list(controls) + [target])
        assert ins == sorted(set(ins)) and outs == sorted(set(outs))
        assert len(ins) >= 3 and len(outs) >= 1

    @pytest.mark.parametrize("conn", ["linear", "grid", "all"])
    def test_benchmark_circuit(self, conn):
        c1 = cu.generate_benchmark_circuit(num_qubits=6, depth=6, connectivity=conn, seed=11)
        c2 = cu.generate_benchmark_circuit(num_qubits=6, depth=6, connectivity=conn, seed=11)
        assert c1 == c2
        assert c1.num_qubits == 6
        assert c1.count_ops().get("measure", 0) == 6
        if conn == "linear":
            for inst in c1.data:
                if inst.operation.name == "cx":
                    i, j = (c1.find_bit(q).index for q in inst.qubits)
                    assert abs(i - j) == 1

    def test_analyze_empty(self):
        r = cu.analyze_toffoli_network_structure([])
        assert r["num_gates"] == 0 and r["depth"] == 0

    def test_analyze_parallel_gates(self):
        r = cu.analyze_toffoli_network_structure([([0, 1], 2), ([3, 4], 5)])
        assert r["num_gates"] == 2
        assert r["depth"] == 1 and r["max_parallelism"] == 2
        assert r["input_size"] == 4

    @pytest.mark.xfail(strict=True, reason="BUG: every control qubit of every gate is pre-marked "
                       "available, and layers are never closed while gates remain executable, so a "
                       "strictly sequential chain reports depth 1 (circuit_utils.py:929-956)")
    def test_analyze_chain_depth(self):
        chain = [([0, 1], 2), ([2, 3], 4), ([4, 0], 5)]  # each gate needs previous target
        assert cu.analyze_toffoli_network_structure(chain)["depth"] == 3


# --------------------------------------------------------------------------- #
# package-level API
# --------------------------------------------------------------------------- #

class TestPublicAPI:
    def test_utils_all_resolvable(self):
        import toffoli_optimizer.utils as u
        missing = [n for n in u.__all__ if not hasattr(u, n)]
        assert missing == []
        assert u.QISKIT_AVAILABLE is True

    def test_top_level_exports(self):
        import toffoli_optimizer as t
        for name in ["ToffoliCompiler", "ToffoliType", "ToffoliDepthOptimizer", "OptimizationStrategy",
                     "estimate_fidelity", "create_optimized_physical_mapping", "get_default_coupling_map",
                     "save_circuit_to_qasm", "load_circuit_from_qasm", "Optimize", "BaseOptimizer"]:
            assert hasattr(t, name), name
        assert t.estimate_fidelity is cu.estimate_fidelity

    def test_version_matches_pyproject(self):
        import pathlib
        import re
        import toffoli_optimizer as t
        txt = (pathlib.Path(t.__file__).resolve().parent.parent / "pyproject.toml").read_text()
        assert re.search(r'^version\s*=\s*"([^"]+)"', txt, re.M).group(1) == t.__version__

    def test_scripts_package(self):
        import toffoli_optimizer.scripts as s
        assert callable(s.main) and set(s.__all__) == {"Optimize", "BaseOptimizer", "main"}
