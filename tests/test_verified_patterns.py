"""Test that the Toffoli pattern library is self-verifying (Feature C).

Run:  python test_verified_patterns.py

Asserts that:
  (a) the library self-verifies at build time without crashing,
  (b) every simplification it will actually APPLY independently passes the
      ExactEquivalenceVerifier (no non-equivalent rewrite can ever be applied),
  (c) the verified/quarantined counts are reported and in the expected ballpark
      ~42 verified / ~66 quarantined out of 108.

Note on counts: the original audit (verify_patterns_demo.py) reported
216 = 84 verified / 132 quarantined, but that demo calls build_patterns() a
SECOND time after construction, which (in the old code) appended duplicate
patterns and doubled every figure. The genuine, deduplicated library has
36 patterns x 3 methods = 108 simplifications; exactly half of the audit's
numbers (42 verified / 66 quarantined). build_patterns() is now idempotent.
"""

import pytest
from qiskit import QuantumCircuit

from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier
from toffoli_optimizer.core.pattern_library_module import ToffoliPatternLibrary


@pytest.fixture(scope="module")
def lib():
    return ToffoliPatternLibrary()


def test_build_self_verifies(lib):
    """(a) Build the library; it must self-verify without crashing."""
    assert lib.patterns, "no patterns generated"
    assert lib.simplified_circuits, "no simplifications generated"
    assert lib.verified, "library did not record verification results"

    report = lib.verification_report()
    assert report["total"] == report["verified"] + report["quarantined"]
    n_results = sum(len(methods) for methods in lib.verified.values())
    assert n_results == report["total"], "verification counts inconsistent with self.verified"


def test_applied_simplifications_are_equivalent(lib):
    """(b) Every simplification the library would APPLY passes the verifier."""
    verifier = ExactEquivalenceVerifier()
    checked = 0
    for pattern_id, pattern in lib.patterns.items():
        metrics = lib.simplification_metrics[pattern_id]
        best_method = metrics["best_method"]
        if best_method is None:
            # No safe method -> library applies nothing for this pattern.
            continue
        # require_verified is on by default, so best_method must be verified.
        assert lib.is_verified(pattern_id, best_method), (
            f"{pattern_id}/{best_method} is selected for application but not verified"
        )
        simplified = lib.simplified_circuits[pattern_id][best_method]
        ok, _perm, info = verifier.verify(pattern["circuit"], simplified,
                                          allow_permutation=False)
        assert ok, (
            f"applied simplification {pattern_id}/{best_method} is NOT equivalent: "
            f"{info.get('reason')}"
        )
        checked += 1
    assert checked > 0, "no applicable simplifications were checked"
    print(f"(b) {checked} applicable simplifications all pass ExactEquivalenceVerifier.")


def test_no_quarantined_method_is_ever_applied(lib):
    """A quarantined (non-equivalent) method must never be chosen as best_method."""
    for pattern_id in lib.patterns:
        best_method = lib.simplification_metrics[pattern_id]["best_method"]
        if best_method is not None:
            assert lib.is_verified(pattern_id, best_method), (
                f"quarantined method {pattern_id}/{best_method} selected for application"
            )


def test_optimize_circuit_preserves_function():
    """End-to-end: optimizing a CCX-containing circuit stays equivalent."""
    verifier = ExactEquivalenceVerifier()
    lib = ToffoliPatternLibrary()
    circ = QuantumCircuit(3)
    circ.ccx(0, 1, 2)
    circ.ccx(0, 1, 2)
    optimized = lib.optimize_circuit(circ, threshold=0.0)
    ok, _perm, info = verifier.verify(circ, optimized, allow_permutation=False)
    assert ok, f"optimize_circuit corrupted the circuit: {info.get('reason')}"
    print("(end-to-end) optimize_circuit preserves function.")


def test_report_counts(lib):
    """(c) Report verified/quarantined counts (expect 42 verified / 66 quarantined).

    These are half the audit's 84/132 figures because the audit double-built the
    library; see the module docstring.
    """
    report = lib.print_verification_report()
    assert report["total"] == 108, f"expected 108 simplifications, got {report['total']}"
    # The relative-phase one_ancilla/multi_ancilla variants are quarantined.
    assert report["verified"] == 42, f"expected 42 verified, got {report['verified']}"
    assert report["quarantined"] == 66, f"expected 66 quarantined, got {report['quarantined']}"


if __name__ == "__main__":
    lib = test_build_self_verifies()
    test_applied_simplifications_are_equivalent(lib)
    test_no_quarantined_method_is_ever_applied(lib)
    test_optimize_circuit_preserves_function()
    test_report_counts(lib)
    print("\nAll verified-pattern tests passed.")
