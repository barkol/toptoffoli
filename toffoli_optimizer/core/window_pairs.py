"""Window-certified relative-phase pairs (condition (W) of the paper).

The structural compute/uncompute detector (``context_analysis``) admits a pair
(g, g') of Toffolis on the same controls and target only if no gate in between
writes a control or the target. Then the diagonal phase D of the gadget commutes
with the segment between them and cancels (condition (C)).

Many reversible-arithmetic circuits violate that syntactic rule but still cancel
the phase: in a Cuccaro adder the MAJ and UMA blocks are mirror images, the CX gates
between a Toffoli and its mirror image are undone symmetrically, and the phases of
nested gadget pairs cancel jointly.

Condition (W): the pair is admissible if the SEGMENT from g to g' (inclusive), with
both Toffolis replaced by gadgets (and any already-admitted gadgets inside), is equal
to the exact segment up to one global phase, as an operator on the qubits the segment
touches. Equality of the segment operator implies equality of the whole circuit up to
that phase, i.e. subroutine equivalence; (C) is the special case in which D commutes
with the window.

The check is exact (dense unitary of the segment) and its cost depends only on the
number of qubits the segment touches, not on the width of the circuit. Candidates are
processed from the shortest window to the longest, so inner pairs are admitted
before the outer pairs whose cancellation relies on them.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

_CCX = {"ccx", "mcx", "mcx_gray"}
_NONUNITARY = {"measure", "reset", "barrier"}


def _ccx_info(circuit, inst):
    name = inst.operation.name.lower()
    qb = [circuit.find_bit(q).index for q in inst.qubits]
    if name in _CCX and len(qb) == 3:
        return frozenset(qb[:2]), qb[2]
    return None


def mirror_pair_candidates(circuit, exclude=()) -> List[Tuple[int, int]]:
    """Pairs (i, j), i < j, of 3-qubit Toffolis with the same controls and target,
    j the nearest later such Toffoli. Sorted by window length (shortest first)."""
    excl = set(exclude)
    info = {}
    for i, inst in enumerate(circuit.data):
        ci = _ccx_info(circuit, inst)
        if ci is not None:
            info[i] = ci
    pairs, used = [], set()
    for i in sorted(info):
        if i in excl or i in used:
            continue
        for j in sorted(k for k in info if k > i):
            if info[j] == info[i]:
                if j not in excl and j not in used:
                    pairs.append((i, j))
                break
    pairs.sort(key=lambda p: p[1] - p[0])
    return pairs


def window_certified(circuit, i: int, j: int, gadget_idx, append_gadget: Callable,
                     max_window_qubits: int = 12, atol: float = 1e-9) -> Tuple[bool, dict]:
    """Condition (W) for the segment [i, j] with gadgets at every index in
    ``gadget_idx`` (which must contain i and j)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    ops = circuit.data[i:j + 1]
    qubits = sorted({circuit.find_bit(q).index for inst in ops for q in inst.qubits})
    if len(qubits) > max_window_qubits:
        return False, {"reason": f"window touches {len(qubits)} qubits > {max_window_qubits}"}
    if any(inst.operation.name.lower() in _NONUNITARY for inst in ops):
        return False, {"reason": "non-unitary operation inside the window"}
    pos = {q: k for k, q in enumerate(qubits)}
    ref = QuantumCircuit(len(qubits))
    cand = QuantumCircuit(len(qubits))
    for k, inst in enumerate(ops, start=i):
        qb = [pos[circuit.find_bit(q).index] for q in inst.qubits]
        ref.append(inst.operation, [ref.qubits[x] for x in qb])
        if k in gadget_idx:
            append_gadget(cand, *qb)
        else:
            cand.append(inst.operation, [cand.qubits[x] for x in qb])
    A = Operator(cand).data
    B = Operator(ref).data
    tr = np.trace(B.conj().T @ A)
    if abs(tr) < 1e-12:
        return False, {"reason": "segment not equal (zero overlap)", "window_qubits": len(qubits)}
    phase = tr / abs(tr)
    ok = bool(np.allclose(A, phase * B, atol=atol))
    return ok, {"window_qubits": len(qubits), "window_gates": len(ops)}


def _crosses(a, b):
    (i, j), (k, l) = a, b
    return (i < k < j < l) or (k < i < l < j)


def admit_window_pairs(circuit, current_gadgets, append_gadget: Callable,
                       max_window_qubits: int = 12, exclude=(), accepted_windows=()) -> Tuple[set, List[dict]]:
    """Greedily admit mirror pairs that pass condition (W), innermost first.

    Admitted windows (including ``accepted_windows`` from the structural pass) must
    form a laminar family: a new window may contain or be disjoint from an accepted
    one, never cross it. Then the certificate of every window stays valid when later,
    larger windows are added, and the whole circuit equals the exact one up to a
    global phase. Returns the enlarged gadget index set and a log of admitted pairs."""
    gadgets = set(current_gadgets)
    windows = list(accepted_windows)
    log = []
    for i, j in mirror_pair_candidates(circuit, exclude=set(exclude) | gadgets):
        if any(_crosses((i, j), w) for w in windows):
            continue
        trial = gadgets | {i, j}
        ok, info = window_certified(circuit, i, j, trial, append_gadget, max_window_qubits)
        if ok:
            gadgets = trial
            windows.append((i, j))
            log.append({"pair": (i, j), "condition": "W", **info})
    return gadgets, log
