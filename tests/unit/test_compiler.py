"""Correctness tests for toffoli_optimizer.core.compiler (ToffoliCompiler).

Ground truth: qiskit.quantum_info.Operator.  Confirmed bugs are pinned with
``xfail(strict=True)`` so that a fix turns them into XPASS -> failure, forcing the
marker to be removed.
"""

import copy

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap

from toffoli_optimizer.core.compiler import ToffoliCompiler, ToffoliType

from ._qhelpers import (BASIS, clean_ancilla_equiv, equiv, independent_counts,
                        independent_depth, random_circuit)

N_ANC = {
    ToffoliType.RELATIVE_PHASE_1: 1,
    ToffoliType.OPTIMIZED_2: 2,
    ToffoliType.OPTIMIZED_3: 3,
    ToffoliType.OPTIMIZED_4: 4,
    ToffoliType.OPTIMIZED_7: 7,
}


@pytest.fixture(scope='module')
def comp():
    return ToffoliCompiler()


def _ccx(n, c1, c2, t):
    qc = QuantumCircuit(n)
    qc.ccx(c1, c2, t)
    return qc


# --------------------------------------------------------------------------- init

def test_init_defaults_and_custom():
    c = ToffoliCompiler()
    assert c.default_basis_gates == ['id', 'rz', 'sx', 'x', 'cx']
    assert isinstance(c.coupling_map, CouplingMap)
    assert c.coupling_map.size() == 20
    assert sorted(c.coupling_map.get_edges())[:2] == [(0, 1), (1, 0)]
    assert set(c.implementations) == set(ToffoliType)
    cm = [[0, 1]]
    c2 = ToffoliCompiler(default_basis_gates=['u', 'cx'], coupling_map=cm, optimization_level=0)
    assert c2.default_basis_gates == ['u', 'cx'] and c2.coupling_map is cm
    assert c2.optimization_level == 0


def test_default_coupling_map_is_line(comp):
    cm = comp._create_default_coupling_map(5)
    edges = {tuple(sorted(e)) for e in cm.get_edges()}
    assert edges == {(0, 1), (1, 2), (2, 3), (3, 4)}


# ------------------------------------------------------ standard Toffoli (no ancilla)

@pytest.mark.parametrize('order', [(0, 1, 2), (2, 0, 1), (1, 2, 0)])
def test_standard_toffoli_classical_action_is_ccx(comp, order):
    """Moduli match CCX: the permutation (classical truth table) is right."""
    qc = QuantumCircuit(3)
    comp._create_standard_toffoli(qc, *order)
    assert clean_ancilla_equiv(qc, _ccx(3, *order), ancillas=[], rel_phase=True)
    cnt = qc.count_ops()
    assert cnt['cx'] == 6 and cnt['h'] == 2


@pytest.mark.xfail(strict=True, reason=(
    "BUG compiler.py:111-117: _create_standard_toffoli is CCX followed by a diagonal "
    "phase diag(1, i, e^{i pi/4}, e^{-i pi/4}) on (control2, control1): extra "
    "t(control2) and swapped t/tdg in the control tail. Not equivalent to CCX."))
@pytest.mark.parametrize('order', [(0, 1, 2), (2, 0, 1)])
def test_standard_toffoli_is_exact_ccx(comp, order):
    qc = QuantumCircuit(3)
    comp._create_standard_toffoli(qc, *order)
    cnt = qc.count_ops()
    assert cnt['t'] + cnt['tdg'] == 7  # textbook T-count; currently 8 (extra t(control2))
    assert equiv(qc, _ccx(3, *order))


def test_standard_toffoli_phase_error_is_observable(comp):
    """Documents why the relative phase matters: sandwiching with H on a control."""
    qc = QuantumCircuit(3)
    qc.h(0)
    comp._create_standard_toffoli(qc, 0, 1, 2)
    qc.h(0)
    ref = QuantumCircuit(3)
    ref.h(0)
    ref.ccx(0, 1, 2)
    ref.h(0)
    # The moduli now differ as well -> classical outcome statistics differ.
    assert not clean_ancilla_equiv(qc, ref, ancillas=[], rel_phase=True)


# -------------------------------------------------- ancilla Toffoli implementations

@pytest.mark.xfail(strict=True, reason=(
    "BUG compiler.py:119-187: RELATIVE_PHASE_1 / OPTIMIZED_2/3/4/7 are not Toffoli "
    "gates, not even up to relative phase, on clean (|0>) ancillas. OPTIMIZED_4/7 "
    "contain the placeholder comment 'Middle operations would go here'."))
@pytest.mark.parametrize('tt', list(N_ANC))
def test_ancilla_toffoli_relative_phase_contract(comp, tt):
    n = 3 + N_ANC[tt]
    qc = QuantumCircuit(n)
    comp.create_toffoli(qc, 0, 1, 2, tt, list(range(3, n)))
    assert clean_ancilla_equiv(qc, _ccx(n, 0, 1, 2), ancillas=list(range(3, n)),
                               rel_phase=True)


@pytest.mark.parametrize('tt', list(N_ANC))
def test_create_toffoli_requires_enough_ancillas(comp, tt):
    qc = QuantumCircuit(12)
    with pytest.raises(ValueError):
        comp.create_toffoli(qc, 0, 1, 2, tt, list(range(3, 3 + N_ANC[tt] - 1)))
    with pytest.raises(ValueError):
        comp.create_toffoli(qc, 0, 1, 2, tt, None)
    assert qc.size() == 0  # nothing appended before the error


def test_create_toffoli_uses_only_requested_qubits(comp):
    for tt, k in N_ANC.items():
        n = 3 + k + 2
        qc = QuantumCircuit(n)
        comp.create_toffoli(qc, 0, 1, 2, tt, list(range(3, n)))  # extra ancillas offered
        used = {qc.find_bit(q).index for inst in qc.data for q in inst.qubits}
        assert used <= set(range(3 + k)), tt


def test_create_toffoli_standard_matches_private(comp):
    a, b = QuantumCircuit(3), QuantumCircuit(3)
    comp.create_toffoli(a, 2, 0, 1)
    comp._create_standard_toffoli(b, 2, 0, 1)
    assert a == b


# ---------------------------------------------------------- approximate Toffoli

def test_approximate_toffoli_branches(comp):
    exact, std = QuantumCircuit(3), QuantumCircuit(3)
    comp._create_approximate_toffoli(exact, 0, 1, 2, fidelity=0.999)
    comp._create_standard_toffoli(std, 0, 1, 2)
    assert exact == std  # >0.99 delegates to the "standard" construction

    hi = QuantumCircuit(3)
    comp._create_approximate_toffoli(hi, 0, 1, 2, fidelity=0.97)
    assert hi.count_ops()['cx'] == 4 and hi.count_ops()['h'] == 2
    # documented approximation: correct moduli, wrong (relative) phases
    assert clean_ancilla_equiv(hi, _ccx(3, 0, 1, 2), [], rel_phase=True)

    mid = QuantumCircuit(3)
    comp._create_approximate_toffoli(mid, 0, 1, 2, fidelity=0.92)
    assert mid.count_ops()['cx'] == 3

    low = QuantumCircuit(3)
    comp._create_approximate_toffoli(low, 0, 1, 2, fidelity=0.5)
    ref = QuantumCircuit(3)
    ref.cx(1, 2)  # cx(c1,t) twice cancels: the "approximation" is just CX(c2,t)
    assert equiv(low, ref)


# ------------------------------------------------------------ create_toffoli_network

def _net(comp, gates, n, **kw):
    return comp.create_toffoli_network(gates, n, **kw)


@pytest.mark.parametrize('seed', range(6))
def test_network_cnot_only_is_exact(comp, seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 6))
    gates, ref = [], QuantumCircuit(n)
    for _ in range(int(rng.integers(1, 8))):
        c, t = (int(x) for x in rng.permutation(n)[:2])
        gates.append(([c], t) if rng.random() < 0.5 else (c, t))
        ref.cx(c, t)
    circ, anc = _net(comp, gates, n)
    assert anc is None or isinstance(anc, list)
    assert equiv(circ, ref)


def test_network_mcx_standard_is_exact(comp):
    circ, anc = _net(comp, [([0, 1, 2], 3), ([3, 1, 0], 2)], 4, toffoli_type=ToffoliType.STANDARD)
    ref = QuantumCircuit(4)
    ref.mcx([0, 1, 2], 3)
    ref.mcx([3, 1, 0], 2)
    assert anc is None
    assert equiv(circ, ref)


def test_network_mcx_vchain_exact_on_clean_ancilla(comp):
    circ, anc = _net(comp, [([0, 1, 2], 3)], 5, toffoli_type=ToffoliType.RELATIVE_PHASE_1)
    assert anc == [4]
    ref = QuantumCircuit(5)
    ref.mcx([0, 1, 2], 3)
    assert clean_ancilla_equiv(circ, ref, ancillas=anc)


def test_network_ancilla_selection(comp):
    # computational qubits {0,1,2}; free = {3,4,5}
    _, anc = _net(comp, [(0, 1, 2)], 6, toffoli_type=ToffoliType.OPTIMIZED_2)
    assert anc == [3, 4]
    _, anc = _net(comp, [(0, 1, 2)], 6, toffoli_type=ToffoliType.OPTIMIZED_4)
    assert anc is None  # not enough free qubits -> fall back to STANDARD
    _, anc = _net(comp, [(0, 1, 2)], 6, toffoli_type=ToffoliType.RELATIVE_PHASE_1,
                  use_ancilla=False)
    assert anc is None
    _, anc = _net(comp, [(0, 1, 2)], 3)
    assert anc is None


def test_network_fallback_to_standard_construction(comp):
    """No free qubit -> the 2-control gate is the 'standard' 16-gate construction."""
    circ, anc = _net(comp, [(0, 1, 2)], 3, toffoli_type=ToffoliType.OPTIMIZED_7)
    std = QuantumCircuit(3)
    comp._create_standard_toffoli(std, 0, 1, 2)
    assert anc is None and circ == std


def test_network_old_and_new_style_identical(comp):
    a, _ = _net(comp, [(0, 1, 2), (2, 1, 0)], 3)
    b, _ = _net(comp, [([0, 1], 2), ([2, 1], 0)], 3)
    c, _ = _net(comp, [((0, 1), 2), ((2, 1), 0)], 3)
    assert a == b == c


@pytest.mark.parametrize('strategy', [
    'shared',
    pytest.param('dedicated', marks=pytest.mark.xfail(strict=True, reason=(
        "BUG compiler.py:496-527: 'dedicated' allocation gives gate i the free qubits "
        "available_ancilla[i*k:(i+1)*k] (here 5,6 for the 2nd gate) but the returned "
        "ancilla_indices only lists the first k ([3,4]); callers are not told which "
        "qubits were used as ancillas."))),
    'bogus'])
def test_network_ancilla_strategies_only_touch_declared_qubits(comp, strategy):
    gates = [(0, 1, 2), (1, 2, 0)]
    circ, anc = _net(comp, gates, 7, toffoli_type=ToffoliType.OPTIMIZED_2,
                     allocate_ancilla_strategy=strategy)
    assert anc == [3, 4]
    used = {circ.find_bit(q).index for inst in circ.data for q in inst.qubits}
    assert used <= {0, 1, 2, 3, 4}


@pytest.mark.xfail(strict=True, reason=(
    "BUG compiler.py:279 + 119-127: the DEFAULT toffoli_type=RELATIVE_PHASE_1 builds a "
    "non-Toffoli whenever a free qubit exists, so create_toffoli_network([(0,1,2)], 4) "
    "does not implement CCX even on a clean ancilla."))
def test_network_default_type_implements_ccx(comp):
    circ, anc = _net(comp, [(0, 1, 2)], 4)
    assert anc == [3]
    assert clean_ancilla_equiv(circ, _ccx(4, 0, 1, 2), ancillas=anc)


@pytest.mark.xfail(strict=True, reason=(
    "BUG compiler.py:111-117 (via create_toffoli_network STANDARD path): 2-control "
    "Toffoli networks are off by a diagonal phase on the controls."))
def test_network_standard_two_control_is_exact(comp):
    circ, _ = _net(comp, [(0, 1, 2), (2, 0, 1)], 3, toffoli_type=ToffoliType.STANDARD)
    ref = QuantumCircuit(3)
    ref.ccx(0, 1, 2)
    ref.ccx(2, 0, 1)
    assert equiv(circ, ref)


def test_network_invalid_entries_are_skipped(comp):
    circ, anc = _net(comp, [[0, 1, 2], (0, 1, 2, 3), 'x', (0, 9), ([0, 1], 7)], 3)
    assert circ.size() == 0
    # (cosmetic) with no valid gate every qubit is 'free', so anc == [0] is reported
    assert anc in (None, [0])


@pytest.mark.xfail(strict=True, reason=(
    "BUG compiler.py:385-437: a gate with an out-of-range control, e.g. ([0,5],2) on 3 "
    "qubits, is silently degraded to CX(0,2) instead of being skipped (docstring: "
    "'Gates with invalid control or target qubits are skipped')."))
def test_network_out_of_range_control_is_skipped(comp):
    circ, _ = _net(comp, [([0, 5], 2)], 3)
    assert circ.size() == 0


@pytest.mark.xfail(strict=True, reason=(
    "BUG compiler.py:549 + 574-577: duplicate controls ([0,0],2) crash halfway through "
    "_create_standard_toffoli; the exception is swallowed and 12 orphan gates of the "
    "partial decomposition stay in the circuit."))
def test_network_duplicate_controls_leave_no_partial_gates(comp):
    circ, _ = _net(comp, [([0, 0], 2)], 3, toffoli_type=ToffoliType.STANDARD)
    assert circ.size() == 0


def test_network_does_not_mutate_input(comp):
    gates = [([0, 1], 2), (1, 2, 0), ([0], 1)]
    snapshot = copy.deepcopy(gates)
    _net(comp, gates, 4)
    assert gates == snapshot


def test_network_is_deterministic(comp):
    gates = [([0, 1], 2), ([2, 3, 0], 1), (0, 3)]
    a, _ = _net(comp, gates, 6)
    b, _ = _net(comp, gates, 6)
    assert a == b


# ------------------------------------------------------------ decompose_to_basis_gates

@pytest.mark.parametrize('seed', range(4))
def test_decompose_to_basis_is_equivalent_and_in_basis(comp, seed):
    rng = np.random.default_rng(100 + seed)
    qc = random_circuit(4, 10, rng)
    out = comp.decompose_to_basis_gates(qc, optimization_level=seed % 4)
    assert set(out.count_ops()) <= set(BASIS)
    assert equiv(qc, out)


def test_decompose_custom_basis_and_none(comp):
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.h(0)
    out = comp.decompose_to_basis_gates(qc, basis_gates=['u', 'cx'])
    assert set(out.count_ops()) <= {'u', 'cx'} and equiv(qc, out)
    assert comp.decompose_to_basis_gates(None) is None
    assert comp.decompose_to_basis_gates(qc, basis_gates=['nonexistent_gate']) is None


# ------------------------------------------------------------------- metrics

@pytest.mark.parametrize('seed', range(6))
def test_metrics_match_independent_counts(comp, seed):
    rng = np.random.default_rng(200 + seed)
    qc = random_circuit(int(rng.integers(1, 6)), int(rng.integers(0, 25)), rng,
                        gates=('x', 'cx', 'h', 't', 's', 'rz', 'swap', 'ccx'))
    if rng.random() < 0.5:
        qc = comp.decompose_to_basis_gates(qc, basis_gates=['h', 't', 'tdg', 'cx', 's', 'x', 'rz'],
                                           optimization_level=0)
    m = comp.get_circuit_metrics(qc)
    ind = independent_counts(qc)
    assert m['depth'] == independent_depth(qc)
    assert m['width'] == qc.num_qubits
    assert m['size'] == ind['size']
    assert m['cx_count'] == ind['cx']
    assert m['t_gates'] == ind['t']
    assert sum(m['gate_counts'].values()) == ind['size']
    assert 0.0 <= m['fidelity'] <= 1.0


def test_metrics_with_measure_and_barrier(comp):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    m = comp.get_circuit_metrics(qc)
    assert m['depth'] == independent_depth(qc) == 3
    assert m['gate_counts']['measure'] == 2 and m['cx_count'] == 1


def test_metrics_none_and_empty(comp):
    m = comp.get_circuit_metrics(None)
    assert m['depth'] == 0 and m['size'] == 0 and m['fidelity'] == 0
    m = comp.get_circuit_metrics(QuantumCircuit(3))
    assert m['depth'] == 0 and m['width'] == 3 and m['fidelity'] == 1.0


@pytest.mark.xfail(strict=True, reason=(
    "BUG (cosmetic) compiler.py:720-724: get_circuit_metrics knows t_gates but does not "
    "pass it to estimate_fidelity, which then guesses t_count=10% of size; a pure "
    "10xT circuit is scored as 1 T + 9 'other' gates."))
def test_metrics_fidelity_uses_actual_t_count(comp):
    qc = QuantumCircuit(1)
    for _ in range(10):
        qc.t(0)
    m = comp.get_circuit_metrics(qc)
    assert m['fidelity'] == pytest.approx(comp.estimate_fidelity(1, 10, cx_count=0, t_count=10))


def test_estimate_fidelity_model(comp):
    f = comp.estimate_fidelity(3, 10, cx_count=2, t_count=3)
    assert f == pytest.approx(0.99 ** 2 * 0.998 ** 3 * 0.999 ** 5)
    assert comp.estimate_fidelity(3, 0, 0, 0) == 1.0
    # monotone in CX count
    fs = [comp.estimate_fidelity(3, 50, cx_count=k, t_count=0) for k in range(0, 50, 10)]
    assert all(a > b for a, b in zip(fs, fs[1:]))
    # decoherence factor is floored at 0.8
    base = comp.estimate_fidelity(1000, 0, 0, 0)
    assert comp.estimate_fidelity(1000, 0, 0, 0, depth=10 ** 6) == pytest.approx(0.8 * base)
    assert 0.0 <= comp.estimate_fidelity(5, 10 ** 5) <= 1.0


def test_print_comparison_reports_true_numbers(comp, capsys):
    a = QuantumCircuit(3)
    a.ccx(0, 1, 2)
    b = comp.decompose_to_basis_gates(a)
    c = comp.decompose_to_basis_gates(a, basis_gates=['h', 't', 'tdg', 'cx'], optimization_level=0)
    comp.print_comparison(a, b, c)
    out = capsys.readouterr().out
    assert '=== Circuit Comparison ===' in out
    line = next(ln for ln in out.splitlines() if ln.startswith('CNOT Count'))
    assert line.split()[2:] == [str(0), str(b.count_ops().get('cx', 0)),
                                str(c.count_ops().get('cx', 0))]
    comp.print_comparison(None, b, c)
    assert 'Cannot compare' in capsys.readouterr().out


def test_create_toffoli_network_controls_as_int_path(comp):
    circ, _ = comp.create_toffoli_network([(1, 0)], 2)
    ref = QuantumCircuit(2)
    ref.cx(1, 0)
    assert equiv(circ, ref)
    # an X gate object is not part of the API: make sure nothing crashes on it
    circ, _ = comp.create_toffoli_network([XGate()], 2)
    assert circ.size() == 0
