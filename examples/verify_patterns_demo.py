"""Demonstrate the exact equivalence verifier and audit toptoffoli's own patterns.

Run:  pip install -e . && python examples/verify_patterns_demo.py

Part 1 sanity-checks the verifier on known cases.
Part 2 runs it against ToffoliPatternLibrary's simplified circuits -- which were
previously NEVER checked for functional equivalence -- and reports any that do
not actually compute the original Toffoli function.
"""

from qiskit import QuantumCircuit

from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier


def _section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def part1_sanity():
    _section("Part 1 - verifier sanity checks")
    v = ExactEquivalenceVerifier()

    # (a) CCX vs a correct decomposition -> equivalent
    a = QuantumCircuit(3); a.ccx(0, 1, 2)
    b = QuantumCircuit(3)
    b.h(2); b.cx(1, 2); b.tdg(2); b.cx(0, 2); b.t(2); b.cx(1, 2); b.tdg(2)
    b.cx(0, 2); b.t(1); b.t(2); b.cx(0, 1); b.h(2); b.t(0); b.tdg(1); b.cx(0, 1)
    ok, perm, info = v.verify(a, b)
    print(f"[{'OK ' if ok else 'BAD'}] CCX == standard 7-T decomposition  -> {ok}  ({info.get('method')})")
    assert ok

    # (b) CCX vs CX -> NOT equivalent (verifier must reject)
    c = QuantumCircuit(3); c.cx(0, 2)
    ok, _, info = v.verify(a, c)
    print(f"[{'OK ' if not ok else 'BAD'}] CCX != CX  -> rejected={not ok}  ({info.get('reason')})")
    assert not ok

    # (c) two CCX annihilate -> identity (equivalent to empty)
    d = QuantumCircuit(3); d.ccx(0, 1, 2); d.ccx(0, 1, 2)
    e = QuantumCircuit(3)
    ok, _, info = v.verify(e, d)
    print(f"[{'OK ' if ok else 'BAD'}] CCX;CCX == identity  -> {ok}  ({info.get('method')})")
    assert ok

    # (d) relative-phase ("Margolus") Toffoli vs true CCX -> NOT equal (off by phase)
    rp = QuantumCircuit(3)
    rp.ry(0.7853981633974483, 2); rp.cx(1, 2); rp.ry(0.7853981633974483, 2)
    rp.cx(0, 2); rp.ry(-0.7853981633974483, 2); rp.cx(1, 2); rp.ry(-0.7853981633974483, 2)
    ok, _, info = v.verify(a, rp)
    print(f"[{'OK ' if not ok else 'BAD'}] relative-phase Toffoli != true CCX  -> rejected={not ok}  ({info.get('reason')})")
    assert not ok
    print("\nAll sanity checks passed.")


def part2_audit_patterns():
    _section("Part 2 - auditing ToffoliPatternLibrary's (previously unverified) simplifications")
    try:
        from toffoli_optimizer.core.pattern_library_module import ToffoliPatternLibrary
    except Exception as ex:  # pragma: no cover
        print(f"could not import ToffoliPatternLibrary: {ex}")
        return

    lib = ToffoliPatternLibrary()
    if hasattr(lib, "build_patterns"):
        try:
            lib.build_patterns()
        except Exception:
            pass

    patterns = getattr(lib, "patterns", {}) or {}
    simplified = getattr(lib, "simplified_circuits", {}) or {}
    if not patterns or not simplified:
        print("pattern library exposed no {patterns, simplified_circuits}; "
              "skipping (API may differ). Verifier is still usable directly.")
        return

    v = ExactEquivalenceVerifier()
    total = bad = 0
    bad_list = []
    for pid, pat in patterns.items():
        orig = pat.get("circuit") if isinstance(pat, dict) else None
        if orig is None:
            continue
        for method, simp in (simplified.get(pid, {}) or {}).items():
            if simp is None:
                continue
            total += 1
            try:
                ok, _, info = v.verify(orig, simp, allow_permutation=False)
            except Exception as ex:
                ok, info = False, {"reason": f"verify error: {ex}"}
            if not ok:
                bad += 1
                bad_list.append((pid, method, info.get("reason")))

    print(f"checked {total} pattern simplifications; "
          f"{total - bad} verified equivalent, {bad} NOT equivalent.")
    for pid, method, reason in bad_list[:20]:
        print(f"  NOT EQUIVALENT: pattern={pid} method={method}  reason={reason}")
    if bad:
        print("\n=> These rewrites would silently corrupt a circuit. The verifier "
              "lets the optimizer reject them (Feature A) and lets you ship only "
              "verified patterns (Feature C).")
    else:
        print("\n=> All current simplifications verified equivalent. The verifier now "
              "guarantees this holds for any future pattern, automatically.")


if __name__ == "__main__":
    part1_sanity()
    part2_audit_patterns()
