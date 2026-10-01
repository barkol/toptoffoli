"""Operator-level admissibility checks on the reachable SUBSPACE.

Why this module exists
----------------------
A relative-phase decomposition differs from CCX by a diagonal phase D. On any single
computational-basis state D is just a global scalar, so a check of the form

    "for every reachable basis state x:  U_d|x>  =  e^{i a_x} U_CCX|x>"

is blind to it: it passes even when the phases a_x differ between reachable basis
states. The state that actually enters the gate can be a SUPERPOSITION of reachable
basis states (data in superposition, or a Hadamard earlier in the circuit), and then
the differing a_x become a relative phase that a later interfering gate can measure.
(Counterexample: controls = 1, target = 0, H on the target before and after the
Toffoli, CCX replaced by Qiskit's RCCX.)

The sound condition is an OPERATOR identity on the span of the reachable basis
states, with ONE phase for the whole subspace:

    (R)        U_d P  =  e^{i theta} U_CCX P,      P = projector onto span(S(g)),

and its bounded-approximate variant  min_theta || (U_d - e^{i theta} U_CCX) P || <= eps
(operator norm). This module checks both for a gate acting on k qubits.

Projection to the gate's qubits
-------------------------------
S(g) is a set of n-qubit basis strings. Let L be its projection onto the gate's
qubits. Since S(g) is contained in L x {0,1}^(n-k), span(S(g)) is contained in
span(L) (x) C^(2^(n-k)), and U_d, U_CCX act as (.) (x) I. Checking the condition on
span(L) is therefore a check on a SUPERSET of the reachable subspace, which is sound
(it can only reject more).

Whole-circuit certification
---------------------------
``certify_on_input_subspace`` checks the end-to-end statements of the soundness
theorem for circuits small enough to build their unitaries:

* mode="subroutine":  U_sel[:, I] = e^{i theta} U_ex[:, I]   (one phase), and
* mode="observational":  there is a diagonal unitary Phi with
  U_sel[:, I] = Phi U_ex[:, I]  (one phase PER OUTPUT ROW).  This is exactly
  equality of the computational-basis output distributions for every input state
  supported on span{|x> : x in I}, superpositions included.

With a tolerance, the reported deviation is || U_sel[:, I] - Phi U_ex[:, I] ||_op
with the best scalar (subroutine) or best diagonal row phases (observational); it
upper-bounds the total-variation distance of the output distributions for every
input state on the input subspace.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence, Tuple

import numpy as np


def project_support(states: Iterable[int], qubits: Sequence[int]) -> list:
    """Project global basis indices onto ``qubits`` (local bit j == qubits[j])."""
    local = set()
    for s in states:
        v = 0
        for j, q in enumerate(qubits):
            if (s >> q) & 1:
                v |= 1 << j
        local.add(v)
    return sorted(local)


def gadget_unitary(append_fn: Callable, k: int = 3) -> np.ndarray:
    """Unitary of a k-qubit gadget built by ``append_fn(qc, 0, 1, ..., k-1)``."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(k)
    append_fn(qc, *range(k))
    return Operator(qc).data


def _min_scalar_phase_opnorm(A: np.ndarray, B: np.ndarray) -> Tuple[float, float]:
    """min over theta of ||A - e^{i theta} B||_2 (spectral norm), and the argmin.

    The function of theta is continuous and 2pi-periodic; we start from the
    trace-optimal phase and refine with a dense grid plus golden-section search.
    """
    def f(th):
        return np.linalg.norm(A - np.exp(1j * th) * B, 2)

    tr = np.trace(B.conj().T @ A)
    th0 = float(np.angle(tr)) if abs(tr) > 1e-15 else 0.0
    grid = th0 + np.linspace(-np.pi, np.pi, 73)
    vals = [f(t) for t in grid]
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(60):
        if f(c) < f(d):
            hi = d
        else:
            lo = c
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    th = 0.5 * (lo + hi)
    best = min(f(th), vals[i], f(th0))
    return float(best), th


def check_on_subspace(
    U_d: np.ndarray, U_ref: np.ndarray, local_states: Sequence[int],
    epsilon: float = 0.0, atol: float = 1e-9,
) -> Tuple[bool, float, dict]:
    """Condition (R) (epsilon == 0) or its bounded-approximate variant.

    Returns (admissible, deviation, info) with
    deviation = min_theta || (U_d - e^{i theta} U_ref) P_L ||_op.
    """
    cols = list(local_states)
    if not cols:
        return True, 0.0, {"method": "subspace_R", "reachable_local": 0}
    A = U_d[:, cols]
    B = U_ref[:, cols]
    dev, th = _min_scalar_phase_opnorm(A, B)
    if dev < atol:
        dev = 0.0
    return dev <= epsilon + atol, dev, {
        "method": "subspace_R", "reachable_local": len(cols), "theta": th}


def certify_on_input_subspace(
    U_ex: np.ndarray, U_sel: np.ndarray, inputs: Sequence[int],
    mode: str = "subroutine", tolerance: float = 0.0, atol: float = 1e-8,
) -> Tuple[bool, float, dict]:
    """End-to-end certificate on span{|x> : x in inputs} (see module docstring)."""
    cols = sorted(set(inputs))
    A = U_sel[:, cols]
    B = U_ex[:, cols]
    if mode == "subroutine":
        # Exact decision (tolerance 0) with one SVD. B has orthonormal columns, so if
        # A = e^{i th} B + E with ||E|| <= d, the trace phase th0 = arg tr(B^H A)
        # gives ||A - e^{i th0} B|| <= 2 d. Hence the value at th0 is within a
        # factor 2 of the minimum: certify if it is <= atol, reject if > 2 atol, and
        # run the full minimisation only in between (and for tolerance > 0).
        tr = np.trace(B.conj().T @ A)
        th0 = float(np.angle(tr)) if abs(tr) > 1e-15 else 0.0
        dev0 = float(np.linalg.norm(A - np.exp(1j * th0) * B, 2))
        if tolerance == 0.0 and (dev0 <= atol or dev0 > 2 * atol):
            dev, th = dev0, th0
            info = {"theta": th, "phase": "trace"}
        else:
            dev, th = _min_scalar_phase_opnorm(A, B)
            info = {"theta": th}
    elif mode == "observational":
        # Best phase per output row: phi_y = arg <B_y, A_y>.
        inner = np.einsum("ij,ij->i", B.conj(), A)
        phases = np.where(np.abs(inner) > 1e-15, inner / np.maximum(np.abs(inner), 1e-300), 1.0)
        dev = float(np.linalg.norm(A - phases[:, None] * B, 2))
        info = {}
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if dev < atol:
        dev = 0.0
    return dev <= tolerance + atol, float(dev), {
        "method": f"input_subspace_{mode}", "inputs": len(cols), **info}


def with_ancillas(circ, pinned):
    """Same circuit with the pinned qubits moved into an AncillaRegister (last),
    so QCEC treats them as initialised to |0>."""
    from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
    pinned = list(pinned)
    pset = set(pinned)
    free = [q for q in range(circ.num_qubits) if q not in pset]
    pos = {q: k for k, q in enumerate(free + pinned)}
    regs = [QuantumRegister(len(free), "d")] + ([AncillaRegister(len(pinned), "anc0")] if pinned else [])
    out = QuantumCircuit(*regs)
    for inst in circ.data:
        out.append(inst.operation, [out.qubits[pos[circ.find_bit(q).index]] for q in inst.qubits])
    return out


def qcec_certify_on_subspace(exact, selected, pinned_zero=(), observational=False, timeout=120):
    """Whole-circuit certificate with MQT QCEC on the input subspace (pinned qubits
    are ancillas initialised to |0>). Returns (ok, info). ``ok`` is True only for
    a proof of equivalence up to a global phase; anything else (not equivalent,
    timeout, no information, QCEC unavailable, or a program-semantics circuit,
    which QCEC cannot certify) is False, so callers fail closed."""
    if observational:
        return False, {"reason": "program-semantics output: no decision-diagram certificate"}
    try:
        import mqt.qcec as qcec
    except Exception as exc:
        return False, {"reason": f"QCEC unavailable: {exc!r}"}
    try:
        r = qcec.verify(with_ancillas(exact, pinned_zero), with_ancillas(selected, pinned_zero), timeout=timeout)
    except TypeError:
        r = qcec.verify(with_ancillas(exact, pinned_zero), with_ancillas(selected, pinned_zero))
    eq = str(r.equivalence).split(".")[-1]
    ok = eq in ("equivalent", "equivalent_up_to_global_phase")
    return ok, {"qcec": eq}
