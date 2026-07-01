#!/usr/bin/env python3
"""Tests for Feature A: the exact-equivalence correctness gate wired into the
ToffoliDepthOptimizer.

Run with:  python test_verify_gate.py

Covers:
  (a) A small CCX network optimized with verify_equivalence=True produces an
      output that the ExactEquivalenceVerifier confirms is equivalent to the
      pre-optimization (logical) input.
  (b) The gate REJECTS a known-wrong rewrite: when a pass is forced to emit a
      candidate that replaces a CCX with a CX, the optimizer falls back to the
      last verified-good circuit rather than emitting the broken candidate.
"""

import sys

from qiskit import QuantumCircuit

from toffoli_optimizer.core.toffoli_depth_optimizer import ToffoliDepthOptimizer
from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier


def _build_optimizer(**kw):
    # pass_timeout_seconds > 0 exercises the (correct-arity) timeout path of the
    # internal _optimize_circuit call.
    params = dict(max_passes=1, pass_timeout_seconds=30, debug_mode=False)
    params.update(kw)
    return ToffoliDepthOptimizer(**params)


def test_gate_accepts_equivalent_output():
    """(a) Optimize a small CCX network with the gate ON; the produced optimized
    logical circuit must verify equivalent to the logical (input) circuit."""
    opt = _build_optimizer(verify_equivalence=True)
    assert opt.verify_equivalence is True, "gate should be enabled"

    res = opt.optimize_toffoli_network([(0, 1, 2)], num_qubits=3, topology="linear")
    assert "error" not in res or res.get("error") is None, f"optimize errored: {res.get('error')}"
    assert "logical" in res and "optimized" in res, "missing result circuits"

    logical = res["logical"]["circuit"]
    optimized = res["optimized"]["circuit"]

    # Independently confirm the optimized output is equivalent to the input.
    verifier = ExactEquivalenceVerifier()
    is_equiv, _perm, info = verifier.verify(logical, optimized, allow_permutation=False)
    assert is_equiv, f"optimized output is NOT equivalent to input: {info}"
    print(f"[a] PASS: optimized output verified equivalent to input (method={info.get('method')})")


def test_gate_rejects_bad_rewrite():
    """(b) Force a pass to emit a known-wrong candidate (CCX -> CX) and assert the
    optimizer rejects it and falls back instead of emitting the broken circuit."""
    opt = _build_optimizer(verify_equivalence=True)

    # A deliberately wrong "optimized" circuit: a single CX where the function
    # demands a CCX. This is shallower/cheaper, so _is_better_circuit would
    # happily accept it -- only the equivalence gate should stop it.
    bad = QuantumCircuit(3)
    bad.cx(0, 2)

    # Monkeypatch the per-pass optimizer to return the broken candidate.
    opt._optimize_circuit = lambda *a, **k: bad.copy()

    res = opt.optimize_toffoli_network([(0, 1, 2)], num_qubits=3, topology="linear")
    assert "error" not in res or res.get("error") is None, f"optimize errored: {res.get('error')}"

    logical = res["logical"]["circuit"]
    optimized = res["optimized"]["circuit"]

    verifier = ExactEquivalenceVerifier()

    # Sanity: the bad candidate really IS non-equivalent (so the test is meaningful).
    bad_equiv, _p, bad_info = verifier.verify(logical, bad, allow_permutation=False)
    assert not bad_equiv, f"test setup broken: forced candidate was actually equivalent ({bad_info})"

    # The emitted optimized circuit must NOT be the broken candidate, and must be
    # equivalent to the input (the gate fell back to the verified-good logical).
    is_equiv, _perm, info = verifier.verify(logical, optimized, allow_permutation=False)
    assert is_equiv, f"gate failed to reject bad rewrite; output is non-equivalent: {info}"

    # Confirm it isn't the single-CX broken circuit (it should be the full network).
    op_counts = optimized.count_ops()
    assert not (len(optimized.data) == 1 and op_counts.get("cx", 0) == 1), (
        f"optimizer emitted the broken candidate instead of falling back: {dict(op_counts)}"
    )
    print(f"[b] PASS: bad rewrite (CCX->CX) rejected; fell back to a verified-good "
          f"equivalent circuit (method={info.get('method')})")


def test_gate_off_unchanged():
    """Sanity: with the gate OFF (default), the optimizer still runs and the
    verify flag is False -- default behavior is unchanged."""
    opt = _build_optimizer()  # verify_equivalence defaults to off
    assert opt.verify_equivalence is False, "gate must default OFF"
    res = opt.optimize_toffoli_network([(0, 1, 2)], num_qubits=3, topology="linear")
    assert "optimized" in res, "optimize did not produce a result with gate off"
    print("[off] PASS: default (gate off) behavior preserved")


def main():
    test_gate_off_unchanged()
    test_gate_accepts_equivalent_output()
    test_gate_rejects_bad_rewrite()
    print("\nAll equivalence-gate tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
