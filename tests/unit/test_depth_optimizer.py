"""Correctness tests for ToffoliDepthOptimizer, CircuitGateProcessor, the
optimizer facade and OptimizationStrategy.

Main property: every transformation that claims to preserve the circuit returns a
circuit unitarily equivalent (up to global phase) to its input; ground truth is
qiskit.quantum_info.Operator on <= 7 qubits.  Confirmed bugs are pinned with
``xfail(strict=True)``.
"""

import copy
import time

import numpy as np
import pytest
import qiskit
from hypothesis import HealthCheck, given, settings, strategies as st
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import HGate, XGate
from qiskit.transpiler import CouplingMap

import toffoli_optimizer.core as core
from toffoli_optimizer.core import optimizer as facade
from toffoli_optimizer.core.circuit_gate_processor import CircuitGateProcessor
from toffoli_optimizer.core.compiler import ToffoliType
from toffoli_optimizer.core.optimization_strategy import OptimizationStrategy as S
from toffoli_optimizer.core.toffoli_depth_optimizer import (TimeoutException,
                                                            ToffoliDepthOptimizer,
                                                            time_limit)

from ._qhelpers import (BASIS, clean_ancilla_equiv, equiv, independent_counts,
                        independent_depth, random_circuit)

LINE4 = [[0, 1], [1, 2], [2, 3]]


@pytest.fixture
def opt(tmp_path):
    return ToffoliDepthOptimizer(output_dir=str(tmp_path), max_passes=1)


def run_opt(o, circ, strategy=None, cmap=None):
    if strategy is not None:
        o.strategy = strategy
    return o._optimize_circuit(circ.copy(), [], [], circ.num_qubits, cmap, BASIS, 0.95)


# ================================================================ facade / enum

def test_facade_and_enum():
    assert facade.ToffoliDepthOptimizer is ToffoliDepthOptimizer
    assert core.OptimizationStrategy is S
    assert len(S) == 12 and len({s.value for s in S}) == 12
    assert {'TRANSPILER', 'HYBRID', 'TOFFOLI_COUNT_REDUCTION'} <= set(S.__members__)
    for name in facade.__all__:
        assert hasattr(facade, name)


@pytest.mark.parametrize('arg,expected', [
    (None, S.HYBRID), (S.GATE_REDUCTION, S.GATE_REDUCTION),
    ('DEPTH_REDUCTION', S.DEPTH_REDUCTION), ('nope', S.HYBRID), (3, S.HYBRID)])
def test_strategy_parsing(tmp_path, arg, expected):
    assert ToffoliDepthOptimizer(strategy=arg, output_dir=str(tmp_path)).strategy is expected


def test_verify_equivalence_default_and_override(tmp_path, monkeypatch):
    assert ToffoliDepthOptimizer(output_dir=str(tmp_path)).verify_equivalence is False
    monkeypatch.setattr(ToffoliDepthOptimizer, '_verify_equivalence_default', True)
    assert ToffoliDepthOptimizer(output_dir=str(tmp_path)).verify_equivalence is True
    assert ToffoliDepthOptimizer(output_dir=str(tmp_path),
                                 verify_equivalence=0).verify_equivalence is False
    assert ToffoliDepthOptimizer().output_dir == 'toffoli_optimizer_results'


def test_time_limit_context_manager():
    t0 = time.time()
    with pytest.raises(TimeoutException):
        with time_limit(1):
            while time.time() - t0 < 5:
                pass
    assert time.time() - t0 < 3
    with time_limit(0):  # 0 disables the alarm
        pass


# ================================================================ _is_better_circuit

@pytest.mark.parametrize('strategy,new,cur,final,expected', [
    (S.DEPTH_REDUCTION, (9, 99, 0.9), (10, 1, 0.9), False, True),
    (S.DEPTH_REDUCTION, (10, 1, 0.9), (10, 99, 0.9), False, False),
    (S.GATE_REDUCTION, (99, 9, 0.9), (1, 10, 0.9), False, True),
    (S.FIDELITY, (99, 99, 0.95), (1, 1, 0.9), False, True),
    (S.FIDELITY, (1, 1, 0.5), (9, 9, 0.4), False, False),  # below min_fidelity=0.8
    (S.HYBRID, (5, 10, 0.9), (10, 10, 0.9), False, True),
    (S.HYBRID, (10, 10, 0.9), (10, 10, 0.9), False, False),
    (S.STANDARD, (10, 9, 0.9), (10, 10, 0.9), False, True),
    (S.STANDARD, (10, 10, 0.95), (10, 10, 0.9), False, True),
    (S.STANDARD, (11, 7, 0.9), (10, 10, 0.9), False, False),
    (S.STANDARD, (11, 7, 0.9), (10, 10, 0.9), True, True),
    (S.TRANSPILER_L2, (10, 9, 0.9), (10, 10, 0.9), False, True),
    (S.TRANSPILER, (11, 1, 0.99), (10, 10, 0.9), False, False),
    (S.ULTRA_DEPTH_REDUCTION, (6, 99, 0.6), (10, 10, 0.9), False, True),
    (S.ULTRA_DEPTH_REDUCTION, (10, 99, 0.85), (10, 10, 0.9), False, False),
    (S.DEPTH_FIDELITY_BALANCE, (5, 10, 0.9), (10, 10, 0.9), False, True),
    (S.DEPTH_FIDELITY_BALANCE, (10, 10, 0.9), (10, 10, 0.9), False, False),
])
def test_is_better_circuit(opt, strategy, new, cur, final, expected):
    opt.strategy = strategy
    assert opt._is_better_circuit(*new, *cur, final_pass=final) is expected


# ================================================================ fidelity estimates

def test_estimate_physical_fidelity(opt):
    assert opt.estimate_physical_fidelity(None) == 0.0
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    f = opt.estimate_physical_fidelity(qc)
    assert 0.0 < f <= 1.0
    assert f == opt.compiler.get_circuit_metrics(qc)['fidelity']
    big = QuantumCircuit(3)
    for _ in range(50):
        big.cx(0, 1)
    assert opt.estimate_physical_fidelity(big) < f


def test_calculate_logical_fidelity_basic(opt):
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    assert opt.calculate_logical_fidelity(None, qc) == 0.0
    assert opt.calculate_logical_fidelity(qc, qc) == pytest.approx(0.9)


@pytest.mark.xfail(strict=True, reason=(
    "BUG toffoli_depth_optimizer.py:325-378: 'logical fidelity (function preservation)' "
    "is a gate/depth-count heuristic; an EMPTY circuit scores 1.0 against a CCX "
    "network, i.e. the reported metric rewards deleting the computation."))
def test_logical_fidelity_penalises_wrong_function(opt):
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.cx(2, 0)
    assert opt.calculate_logical_fidelity(qc, QuantumCircuit(3)) < 0.5


# ================================================================ validate_physical_circuit

def test_validate_physical_circuit(opt):
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.ccx(1, 2, 3)
    qc.cx(3, 0)
    ok, viol = opt.validate_physical_circuit(qc, LINE4)
    assert not ok and viol == [(1, 3), (3, 0)]
    ok2, viol2 = opt.validate_physical_circuit(qc, CouplingMap.from_line(4))
    assert (ok2, viol2) == (ok, viol)
    assert opt.validate_physical_circuit(qc, None) == (True, [])
    assert opt.validate_physical_circuit(None, LINE4) == (True, [])
    assert opt.validate_physical_circuit(qc, object()) == (True, [])  # un-introspectable
    good = QuantumCircuit(4)
    good.cx(2, 1)
    assert opt.validate_physical_circuit(good, LINE4) == (True, [])


@pytest.mark.xfail(strict=True, reason=(
    "BUG toffoli_depth_optimizer.py:202-206 (same in circuit_gate_processor.py:112-116): "
    "coupling maps given as a list of TUPLES are silently ignored (isinstance(pair, list)),"
    " so every 2-qubit gate is reported as a violation."))
def test_validate_accepts_tuple_pairs(opt):
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    assert opt.validate_physical_circuit(qc, [(0, 1)]) == (True, [])


@pytest.mark.xfail(strict=True, reason=(
    "BUG toffoli_depth_optimizer.py:230: qubit index taken as Qubit._index (index inside "
    "its REGISTER), wrong for multi-register circuits: cx(a[0], b[1]) is checked as (0,1)."))
def test_validate_multi_register(opt):
    a, b = QuantumRegister(2, 'a'), QuantumRegister(2, 'b')
    qc = QuantumCircuit(a, b)
    qc.cx(a[0], b[1])  # circuit qubits (0, 3)
    assert opt.validate_physical_circuit(qc, [[0, 3]]) == (True, [])


# ================================================================ CircuitGateProcessor

@settings(max_examples=25, deadline=None, database=None, derandomize=True,
          suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 10 ** 6), n=st.integers(1, 5), m=st.integers(0, 15))
def test_gate_processor_no_cmap_preserves_unitary(seed, n, m):
    qc = random_circuit(n, m, np.random.default_rng(seed))
    snapshot = qc.copy()
    out = CircuitGateProcessor().process_circuit_safely(qc)
    assert qc == snapshot  # input not mutated
    assert out.num_qubits == n and equiv(qc, out)


def test_gate_processor_keeps_clbits_and_measurements():
    qc = QuantumCircuit(3, 2)
    qc.h(0)
    qc.ccx(0, 1, 2)
    qc.barrier()
    qc.measure([0, 2], [1, 0])
    out = CircuitGateProcessor().process_circuit_safely(qc, LINE4[:2])
    assert out.num_clbits == 2
    meas = [(out.find_bit(i.qubits[0]).index, out.find_bit(i.clbits[0]).index)
            for i in out.data if i.operation.name == 'measure']
    assert meas == [(0, 1), (2, 0)]


@pytest.mark.xfail(strict=True, reason=(
    "BUG circuit_gate_processor.py:180-197,239-241: with a coupling map, 2-qubit gates "
    "whose name contains no 'c' (swap, rzz, ...) on non-adjacent qubits are silently "
    "DROPPED ('skipped_gates'). Gates with a 'c' in the name are never checked (elif)."))
def test_gate_processor_with_cmap_preserves_unitary():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.swap(0, 3)
    qc.rzz(0.3, 1, 3)
    out = CircuitGateProcessor().process_circuit_safely(qc, LINE4)
    assert equiv(qc, out)


@pytest.mark.xfail(strict=True, reason=(
    "BUG circuit_gate_processor.py:136: Qubit._index is the index inside the qubit's "
    "register; on a 2-register circuit cx(a[0], b[1]) is rebuilt as cx(0, 1)."))
def test_gate_processor_multi_register():
    a, b = QuantumRegister(2, 'a'), QuantumRegister(2, 'b')
    qc = QuantumCircuit(a, b)
    qc.h(a[0])
    qc.cx(a[0], b[1])
    out = CircuitGateProcessor().process_circuit_safely(qc)
    assert equiv(qc, out)


def test_gate_processor_cnot_reversal_path_is_equivalent():
    """The H-sandwich CNOT reversal is mathematically right (exercised directly)."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.cx(1, 0)
    qc.h(0)
    qc.h(1)
    ref = QuantumCircuit(2)
    ref.cx(0, 1)
    assert equiv(qc, ref)


def test_create_controlled_gate_safely():
    gp = CircuitGateProcessor()
    qc = QuantumCircuit(4)
    assert gp.create_controlled_gate_safely(XGate(), [0, 1], [3], qc) is True
    ref = QuantumCircuit(4)
    ref.ccx(0, 1, 3)
    assert equiv(qc, ref)
    qc2 = QuantumCircuit(2)
    assert gp.create_controlled_gate_safely(HGate(), [1], [0], qc2) is True
    ref2 = QuantumCircuit(2)
    ref2.ch(1, 0)
    assert equiv(qc2, ref2)
    assert gp.create_controlled_gate_safely(object(), [0], [1], QuantumCircuit(2)) is False
    assert gp.create_controlled_gate_safely(XGate(), [], [1], QuantumCircuit(2)) is False
    qc3 = QuantumCircuit(2)
    assert gp.create_controlled_gate_safely(XGate(), [0], [5], qc3) is False
    assert qc3.size() == 0


# ================================================================ _optimize_circuit

@pytest.mark.parametrize('strategy', list(S))
@pytest.mark.parametrize('seed', range(2))
def test_optimize_circuit_preserves_unitary_all_strategies(opt, strategy, seed):
    rng = np.random.default_rng(1000 * seed + strategy.value)
    qc = random_circuit(int(rng.integers(3, 6)), int(rng.integers(1, 12)), rng)
    snapshot = qc.copy()
    out = run_opt(opt, qc, strategy)
    assert qc == snapshot
    assert equiv(qc, out)


@pytest.mark.parametrize('strategy', [S.HYBRID, S.DEPTH_REDUCTION, S.TRANSPILER_L2])
def test_optimize_circuit_with_cmap_cx_family_is_equivalent(opt, strategy):
    rng = np.random.default_rng(7)
    qc = random_circuit(4, 10, rng, gates=('cx', 'ccx', 'h', 't', 'x'))
    out = run_opt(opt, qc, strategy, LINE4)
    assert equiv(qc, out)


@pytest.mark.xfail(strict=True, reason=(
    "BUG toffoli_depth_optimizer.py:1260 -> circuit_gate_processor.py: with a coupling "
    "map, _optimize_circuit drops non-adjacent SWAP gates before transpiling; output "
    "is not equivalent to the input."))
def test_optimize_circuit_with_cmap_keeps_swaps(opt):
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.swap(0, 2)
    assert equiv(qc, run_opt(opt, qc, S.HYBRID, [[0, 1], [1, 2]]))


def test_optimize_circuit_edge_cases(opt):
    out = run_opt(opt, QuantumCircuit(3), S.HYBRID)
    assert out.size() == 0 and out.num_qubits == 3
    one = QuantumCircuit(1)
    one.h(0)
    one.t(0)
    one.tdg(0)
    one.h(0)
    assert equiv(one, run_opt(opt, one, S.GATE_REDUCTION))
    hi = QuantumCircuit(7)
    hi.ccx(6, 5, 4)
    hi.h(6)
    hi.mcx([6, 0, 3], 1)
    hi.rz(0.3, 6)
    assert equiv(hi, run_opt(opt, hi, S.DEPTH_REDUCTION))
    meas = QuantumCircuit(2, 2)
    meas.h(0)
    meas.cx(0, 1)
    meas.measure([0, 1], [0, 1])
    out = run_opt(opt, meas, S.HYBRID)
    assert out.count_ops()['measure'] == 2 and out.num_clbits == 2
    unitary_part = out.remove_final_measurements(inplace=False)
    assert equiv(meas.remove_final_measurements(inplace=False), unitary_part)


def test_optimize_circuit_deterministic(opt):
    qc = random_circuit(4, 10, np.random.default_rng(3), gates=('cx', 'ccx', 'h', 't'))
    outs = set()
    for s in range(3):
        np.random.seed(s)  # the layout trial uses the global numpy RNG
        outs.add(str(run_opt(opt, qc, S.DEPTH_REDUCTION, LINE4).data))
    assert len(outs) == 1


def test_optimize_circuit_with_memory_management(opt):
    rng = np.random.default_rng(11)
    for strategy in (S.HYBRID, S.STANDARD):
        opt.strategy = strategy
        qc = random_circuit(4, 10, rng)
        out = opt.optimize_circuit_with_memory_management(
            qc.copy(), [], [], [], 4, None, BASIS, 0.95)
        assert equiv(qc, out)
    bad = opt.optimize_circuit_with_memory_management(None, [], [], [], 4, None, BASIS, 0.95)
    assert bad is None  # returns the input on error


# ================================================================ private helpers

def test_decompose_ccx_gates_no_ccx_returns_input(opt):
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    assert opt._decompose_ccx_gates(qc, ToffoliType.STANDARD) is qc


@pytest.mark.xfail(strict=True, reason=(
    "BUG toffoli_depth_optimizer.py:1904-1918: _decompose_ccx_gates pops each CCX and "
    "APPENDS its decomposition at the END of the circuit, reordering it "
    "(ccx; h(2) becomes h(2); decomposition)."))
def test_decompose_ccx_gates_keeps_order(opt):
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.h(2)
    out = opt._decompose_ccx_gates(qc, ToffoliType.STANDARD, use_ancilla=False)
    expected = QuantumCircuit(3)
    opt.compiler._create_standard_toffoli(expected, 0, 1, 2)  # same (buggy) block
    expected.h(2)
    assert equiv(out, expected)


def test_decompose_ccx_gates_adds_ancillas(opt):
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    out = opt._decompose_ccx_gates(qc, ToffoliType.OPTIMIZED_2, use_ancilla=True)
    assert out.num_qubits == 5 and out.ancilla_indices == [3, 4]
    assert 'ccx' not in out.count_ops()
    assert qc.count_ops() == {'ccx': 1}  # input untouched


@pytest.mark.xfail(strict=True, reason=(
    "BUG toffoli_depth_optimizer.py:1697: 'from qiskit import decompose' always raises "
    "ImportError, the except returns the input unchanged: _process_logical_circuit is a "
    "silent no-op (MCX gates are never expanded)."))
def test_process_logical_circuit_expands_mcx(opt):
    qc = QuantumCircuit(4)
    qc.mcx([0, 1, 2], 3)
    out = opt._process_logical_circuit(qc, use_ancilla=False)
    assert 'mcx' not in out.count_ops()


@pytest.mark.xfail(strict=True, reason=(
    "BUG (latent, behind the ImportError) toffoli_depth_optimizer.py:1782-1786: the "
    "no-ancilla 3-control MCX rewrite ccx(c0,c1,t); cx(c2,t); ccx(c0,c1,t) equals CX(c2,t),"
    " not C3X."))
def test_process_logical_circuit_no_ancilla_rewrite_is_equivalent(opt, monkeypatch):
    monkeypatch.setattr(qiskit, 'decompose', None, raising=False)  # get past the import
    qc = QuantumCircuit(4)
    qc.mcx([0, 1, 2], 3)
    out = opt._process_logical_circuit(qc, use_ancilla=False)
    assert 'mcx' not in out.count_ops()
    assert equiv(qc, out)


def test_process_logical_circuit_ancilla_rewrite_is_equivalent(opt, monkeypatch):
    """The 1-ancilla V-chain branch is correct on a clean ancilla (latent code)."""
    monkeypatch.setattr(qiskit, 'decompose', None, raising=False)
    qc = QuantumCircuit(4)
    qc.mcx([0, 1, 2], 3)
    out = opt._process_logical_circuit(qc, use_ancilla=True)
    assert out.num_qubits == 5 and out.count_ops() == {'ccx': 3}
    ref = QuantumCircuit(5)
    ref.mcx([0, 1, 2], 3)
    assert clean_ancilla_equiv(out, ref, ancillas=[4])


@pytest.mark.xfail(strict=True, reason=(
    "BUG toffoli_depth_optimizer.py:1465: uses Qubit.index, which does not exist in "
    "qiskit 2.x; the AttributeError is swallowed and the method always returns set()."))
def test_identify_depth_critical_gates(opt):
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.cx(0, 1)
    qc.h(0)
    assert 0 in opt._identify_depth_critical_gates(qc)


# ================================================================ equivalence gate

def test_equivalence_gate(opt):
    a = QuantumCircuit(3)
    a.ccx(0, 1, 2)
    b = QuantumCircuit(3)
    b.h(2)
    b.ccz(0, 1, 2)
    b.h(2)
    c = QuantumCircuit(3)
    c.cx(0, 2)
    assert opt._equivalence_gate(a, c) == 'ok'  # gate disabled by default
    opt.verify_equivalence = True
    assert opt._equivalence_gate(a, b) == 'ok'
    assert opt._equivalence_gate(a, c) == 'reject'
    assert opt._equivalence_gate(None, c) == 'skip'

    class Boom:
        def verify(self, *a, **k):
            raise RuntimeError('boom')

    opt._equivalence_verifier = Boom()
    assert opt._equivalence_gate(a, b) == 'skip'


# ================================================================ optimize_toffoli_network

NET = [([0, 1], 2), ([1, 2], 3), ([0, 3], 1), ([2], 0), ([0, 1, 3], 2)]


def _ref_network(n, gates):
    ref = QuantumCircuit(n)
    for ctrls, t in gates:
        ref.mcx(list(ctrls), t)
    return ref


def _check_metrics(entry):
    circ = entry['circuit']
    ind = independent_counts(circ)
    assert entry['depth'] == independent_depth(circ)
    assert entry['gate_count'] == ind['size']
    assert entry['cx_count'] == ind['cx']
    assert entry['t_gates'] == ind['t']


@pytest.mark.parametrize('strategy,topology', [
    (S.HYBRID, None), (S.DEPTH_REDUCTION, 'linear'), (S.TRANSPILER_L2, 'linear'),
    (S.ULTRA_DEPTH_REDUCTION, None)])
def test_network_pipeline_preserves_logical_circuit(tmp_path, strategy, topology):
    o = ToffoliDepthOptimizer(output_dir=str(tmp_path), max_passes=1, strategy=strategy)
    gates = copy.deepcopy(NET)
    r = o.optimize_toffoli_network(gates, num_qubits=4, topology=topology,
                                   toffoli_type=ToffoliType.STANDARD)
    assert gates == NET  # input not mutated
    assert 'error' not in r
    L, O, M = r['logical']['circuit'], r['optimized']['circuit'], r['mapped']['circuit']
    assert equiv(L, O)
    assert equiv(L, M)
    assert set(M.count_ops()) <= set(BASIS) | {'barrier'}
    for key in ('logical', 'naive_physical', 'optimized', 'final', 'mapped'):
        _check_metrics(r[key])
    if topology is not None:
        assert r['mapped']['respects_coupling_map'] is True
    exp = (r['logical']['depth'] - r['optimized']['depth']) / r['logical']['depth'] * 100
    assert r['depth_reduction'] == pytest.approx(exp)
    assert r['mapped']['fidelity'] == r['mapped']['physical_fidelity']
    assert 0 <= r['optimized']['best_pass'] <= 1


@pytest.mark.xfail(strict=True, reason=(
    "BUG compiler.py:111-117 propagated: with toffoli_type=STANDARD the 'logical' "
    "circuit of optimize_toffoli_network does not implement the requested CCX network."))
def test_network_pipeline_logical_matches_network(tmp_path):
    o = ToffoliDepthOptimizer(output_dir=str(tmp_path), max_passes=1)
    r = o.optimize_toffoli_network(list(NET), num_qubits=4, toffoli_type=ToffoliType.STANDARD)
    assert equiv(r['logical']['circuit'], _ref_network(4, NET))


@pytest.mark.xfail(strict=True, reason=(
    "BUG toffoli_depth_optimizer.py:613-614 + compiler.py:119-127: the DEFAULT "
    "toffoli_type (RELATIVE_PHASE_1) yields a logical circuit that is not the CCX "
    "network even on a clean ancilla."))
def test_network_pipeline_default_type_matches_network(tmp_path):
    o = ToffoliDepthOptimizer(output_dir=str(tmp_path), max_passes=1)
    gates = [([0, 1], 2)]
    r = o.optimize_toffoli_network(gates, num_qubits=4)
    assert clean_ancilla_equiv(r['logical']['circuit'], _ref_network(4, gates), ancillas=[3])


def test_network_pipeline_cnot_only_matches_network(tmp_path):
    o = ToffoliDepthOptimizer(output_dir=str(tmp_path), max_passes=1)
    gates = [([0], 1), ([2], 0), ([1], 2)]
    r = o.optimize_toffoli_network(gates, num_qubits=3, topology='linear')
    ref = _ref_network(3, gates)
    assert equiv(r['logical']['circuit'], ref)
    assert equiv(r['optimized']['circuit'], ref)
    assert equiv(r['mapped']['circuit'], ref)


@pytest.mark.xfail(strict=True, reason=(
    "BUG toffoli_depth_optimizer.py:778-789: with pass_timeout_seconds=0 the call passes "
    "an extra positional argument (toffoli_gates.copy()) to _optimize_circuit -> "
    "TypeError, swallowed into {'error': ...}."))
def test_network_pipeline_without_timeout(tmp_path):
    o = ToffoliDepthOptimizer(output_dir=str(tmp_path), max_passes=1, pass_timeout_seconds=0)
    r = o.optimize_toffoli_network([([0], 1)], num_qubits=2)
    assert 'error' not in r


@pytest.mark.xfail(strict=True, reason=(
    "BUG toffoli_depth_optimizer.py:600-610: num_qubits inference only handles list "
    "controls; tuple controls ((0,1),2) raise TypeError outside the try block."))
def test_network_pipeline_infers_qubits_with_tuple_controls(tmp_path):
    o = ToffoliDepthOptimizer(output_dir=str(tmp_path), max_passes=1)
    r = o.optimize_toffoli_network([((0,), 1)])
    assert 'error' not in r


def test_network_pipeline_infers_qubits(tmp_path):
    o = ToffoliDepthOptimizer(output_dir=str(tmp_path), max_passes=1)
    r = o.optimize_toffoli_network([(0, 2, 1), ([3], 0)], toffoli_type=ToffoliType.STANDARD)
    assert r['logical']['circuit'].num_qubits == 4


@pytest.mark.xfail(strict=True, reason=(
    "BUG (end-to-end) toffoli_depth_optimizer.py:1260: a user logical_circuit with a "
    "SWAP on non-adjacent qubits + topology='linear' loses the SWAP in "
    "results['optimized'] and results['mapped'] (verify_equivalence off by default)."))
def test_network_pipeline_user_circuit_with_swap(tmp_path):
    L = QuantumCircuit(4)
    L.h(0)
    L.swap(0, 3)
    L.ccx(0, 1, 2)
    o = ToffoliDepthOptimizer(output_dir=str(tmp_path), max_passes=1)
    r = o.optimize_toffoli_network([], num_qubits=4, topology='linear', logical_circuit=L.copy())
    assert equiv(L, r['optimized']['circuit'])


def test_network_pipeline_equivalence_gate_catches_swap_drop(tmp_path):
    L = QuantumCircuit(4)
    L.h(0)
    L.swap(0, 3)
    L.ccx(0, 1, 2)
    snapshot = L.copy()
    o = ToffoliDepthOptimizer(output_dir=str(tmp_path), max_passes=1, verify_equivalence=True)
    r = o.optimize_toffoli_network([], num_qubits=4, topology='linear', logical_circuit=L)
    assert L == snapshot
    assert equiv(L, r['optimized']['circuit'])
    assert equiv(L, r['mapped']['circuit'])
    assert r['optimized']['best_pass'] == 0


@pytest.mark.parametrize('strategy', [S.ULTRA_DEPTH_REDUCTION, S.DEPTH_FIDELITY_BALANCE,
                                      S.DEPTH_REDUCTION])
def test_debug_mode_does_not_change_result(tmp_path, strategy):
    rng = np.random.default_rng(42)
    qc = random_circuit(4, 10, rng, gates=('cx', 'ccx', 'h', 't', 'x'))
    quiet = ToffoliDepthOptimizer(output_dir=str(tmp_path), strategy=strategy)
    loud = ToffoliDepthOptimizer(output_dir=str(tmp_path), strategy=strategy, debug_mode=True)
    a = run_opt(quiet, qc, cmap=LINE4)
    b = run_opt(loud, qc, cmap=LINE4)
    assert equiv(qc, b)
    assert str(a.data) == str(b.data)


def test_compiler_debug_mode_does_not_change_network(tmp_path):
    from toffoli_optimizer.core.compiler import ToffoliCompiler
    gates = [([0, 1], 2), ([0, 1, 2], 3), (1, 0), ([0, 9], 1), ([2, 2], 1)]
    for tt in (ToffoliType.STANDARD, ToffoliType.RELATIVE_PHASE_1, ToffoliType.OPTIMIZED_2):
        for strat in ('shared', 'dedicated'):
            a, x = ToffoliCompiler().create_toffoli_network(
                gates, 6, toffoli_type=tt, allocate_ancilla_strategy=strat)
            b, y = ToffoliCompiler(debug_mode=True).create_toffoli_network(
                gates, 6, toffoli_type=tt, allocate_ancilla_strategy=strat)
            assert a == b and x == y


@pytest.mark.xfail(strict=True, reason=(
    "BUG end-to-end via circuit_gate_processor.py:136 (called at "
    "toffoli_depth_optimizer.py:1260): _optimize_circuit on a 2-register circuit returns "
    "a non-equivalent circuit even WITHOUT a coupling map (cx(a[0], b[1]) -> cx(0, 1))."))
def test_optimize_circuit_multi_register(opt):
    a, b = QuantumRegister(2, 'a'), QuantumRegister(2, 'b')
    qc = QuantumCircuit(a, b)
    qc.h(a[0])
    qc.cx(a[0], b[1])
    assert equiv(qc, run_opt(opt, qc, S.HYBRID))


@pytest.mark.xfail(strict=True, reason=(
    "BUG (cosmetic/dead code) toffoli_depth_optimizer.py:1232,1359: the 'bookend SK "
    "approximation' of ULTRA_DEPTH_REDUCTION / DEPTH_FIDELITY_BALANCE calls methods "
    "that do not exist; the AttributeError is swallowed, so both steps never run."))
def test_sk_bookend_methods_exist():
    assert hasattr(ToffoliDepthOptimizer, '_create_early_sk_approximation')
    assert hasattr(ToffoliDepthOptimizer, '_create_approximate_toffoli_circuit')


@pytest.mark.xfail(strict=True, reason=(
    "BUG (reproducibility) toffoli_depth_optimizer.py:1187-1192 (and 1273, 1313): "
    "transpile(..., coupling_map=...) is called without seed_transpiler, so Sabre "
    "layout/routing is stochastic: TRANSPILER_L1 on the same 4-qubit circuit returned "
    "depth 42 or 43 in repeated runs; reported metrics are not reproducible."))
def test_transpile_calls_are_seeded(opt, monkeypatch):
    calls = []
    real = qiskit.transpile

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(qiskit, 'transpile', spy)
    qc = random_circuit(4, 10, np.random.default_rng(42), gates=('cx', 'ccx', 'h', 't', 'x'))
    out = run_opt(opt, qc, S.TRANSPILER_L1, LINE4)
    assert equiv(qc, out)
    routed = [kw for kw in calls if kw.get('coupling_map') is not None]
    assert routed and all(kw.get('seed_transpiler') is not None for kw in routed)
