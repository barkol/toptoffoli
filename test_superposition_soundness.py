#!/usr/bin/env python3
"""Regression tests: admissibility must hold on the reachable SUBSPACE, not per basis state.

Run with:  python test_superposition_soundness.py

Background (editor's counterexample, Quantum, 2026-10-01): with both controls at 1,
the target at 0 and a Hadamard on the target before and after the Toffoli, Qiskit's
RCCX agrees with CCX up to a phase on EACH reachable basis state |110>, |111>, but the
two phases differ, and the final Hadamard turns the difference into a different target
outcome. A basis-state-wise "agrees up to phase" check accepts RCCX; the subspace
check (one phase for the whole reachable subspace) rejects it.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from toffoli_optimizer.core.subspace_check import (
    check_on_subspace, certify_on_input_subspace, project_support)
from toffoli_optimizer.core.decomposition_selector import (
    ErrorBudgetSelector, append_relative_phase_ccx)
from toffoli_optimizer.core.reachable_subspace import reachable_overapprox


def _u(build, n=3):
    qc = QuantumCircuit(n)
    build(qc)
    return Operator(qc).data


U_CCX = _u(lambda q: q.ccx(0, 1, 2))
U_RCCX = _u(lambda q: q.rccx(0, 1, 2))


def test_editor_counterexample_basiswise_accepts_subspace_rejects():
    reach = [0b011, 0b111]  # q0=1, q1=1, t in {0,1} (little-endian: bit j = qubit j)
    # Basis-state-wise: each column equal up to its own phase -> would be accepted.
    for x in reach:
        a, b = U_RCCX[:, x], U_CCX[:, x]
        assert abs(abs(np.vdot(b, a)) - 1) < 1e-9
    # Subspace (one phase): rejected.
    ok, dev, _ = check_on_subspace(U_RCCX, U_CCX, reach, epsilon=0.0)
    assert not ok and dev > 0.5, dev
    # And the substitution really changes the outcome.
    def run(gate):
        c = QuantumCircuit(3); c.x(0); c.x(1); c.h(2); gate(c); c.h(2)
        return Statevector(c).probabilities([2])
    p_ccx, p_rccx = run(lambda c: c.ccx(0, 1, 2)), run(lambda c: c.rccx(0, 1, 2))
    assert not np.allclose(p_ccx, p_rccx)
    print("[1] PASS: editor's RCCX counterexample: basis-wise check accepts, subspace check rejects "
          f"(deviation {dev:.3f}); outcomes differ {np.round(p_ccx,3)} vs {np.round(p_rccx,3)}")


def test_editor_circuit_through_pipeline_is_exact_and_certified():
    c = QuantumCircuit(3); c.x(0); c.x(1); c.h(2); c.ccx(0, 1, 2); c.h(2)
    res = ErrorBudgetSelector().select(c)
    rep = res["report"]
    assert not rep["phase_aware_admitted"], rep["phase_aware_admitted"]
    assert rep["verified"] is True
    print("[2] PASS: pipeline keeps the Toffoli exact on the editor's circuit "
          f"(certification={rep['verify_info'].get('certification')})")


def test_fig1b_control_interference_rejected_and_forced_gadget_detected():
    # Fig. 1(b) of the revised paper: H on control q0 after g.
    c = QuantumCircuit(3); c.ccx(0, 1, 2); c.h(0)
    res = ErrorBudgetSelector(semantics="program").select(c)
    rep = res["report"]
    assert not rep["phase_aware_admitted"]
    # Forcing the gadget is observably wrong on a superposed data input ...
    forced = QuantumCircuit(3); append_relative_phase_ccx(forced, 0, 1, 2); forced.h(0)
    exact = ErrorBudgetSelector().decompose_exact_only(c)
    inputs = [x for x in range(8) if not (x >> 2) & 1]  # data q0,q1 free, ancilla a=0
    ok, dev, _ = certify_on_input_subspace(
        Operator(exact).data, Operator(forced).data, inputs, mode="observational")
    assert not ok and dev > 0.5, dev
    # ... while every BASIS input alone would not reveal it.
    for x in inputs:
        sv_e = Statevector.from_int(x, 8).evolve(exact).probabilities()
        sv_f = Statevector.from_int(x, 8).evolve(forced).probabilities()
        assert np.allclose(sv_e, sv_f)
    print(f"[3] PASS: Fig. 1(b) rejected; forced gadget caught by the subspace certificate "
          f"(deviation {dev:.3f}) though invisible on basis inputs")


def test_control_drop_rejected_when_control_in_superposition():
    c = QuantumCircuit(3); c.h(0); c.x(1); c.ccx(0, 1, 2)
    res = ErrorBudgetSelector(epsilon=0.0).select(c, input_space=[0])
    rep = res["report"]
    # q1 is provably 1, so dropping q1 (keep q0 -> CX(q0, t)) is exact on the
    # reachable subspace even though q0 is in superposition; dropping q0 is not.
    kept = [d["keep"] for d in rep["approx_admitted"]]
    assert kept == [[0]], kept
    assert rep["verified"] is True
    print("[4] PASS: control-drop admitted only for the provably-set control; "
          "superposed control kept; output certified on the input subspace")


def test_default_subroutine_semantics_never_uses_U():
    # A lone, measured Toffoli: (U) would admit the gadget in "program" semantics,
    # but the default "subroutine" semantics must keep it exact (unitary changes).
    c = QuantumCircuit(3); c.ccx(0, 1, 2)
    rep = ErrorBudgetSelector().select(c)["report"]
    assert not rep["phase_aware_admitted"] and rep["semantics"] == "subroutine"
    assert rep["verified"] is True
    rep_p = ErrorBudgetSelector(semantics="program").select(c)["report"]
    assert rep_p["phase_aware_admitted"] and rep_p["semantics"] == "observational"
    assert rep_p["verified"] is True
    print("[6] PASS: default subroutine semantics keeps a lone Toffoli exact; "
          "program semantics admits it under (U), certified observationally")


def test_pairs_remain_unitary_exact():
    c = QuantumCircuit(4); c.ccx(0, 1, 2); c.cx(2, 3); c.ccx(0, 1, 2)
    res = ErrorBudgetSelector().select(c)
    rep = res["report"]
    assert rep["sites_applied"] == 1 and rep["verified"] is True
    assert rep["semantics"] == "subroutine"
    print("[5] PASS: compute/uncompute pair admitted under (C), unitary-exact, subroutine semantics")


if __name__ == "__main__":
    test_editor_counterexample_basiswise_accepts_subspace_rejects()
    test_editor_circuit_through_pipeline_is_exact_and_certified()
    test_fig1b_control_interference_rejected_and_forced_gadget_detected()
    test_control_drop_rejected_when_control_in_superposition()
    test_default_subroutine_semantics_never_uses_U()
    test_pairs_remain_unitary_exact()
    print("\nAll superposition-soundness tests passed.")
