"""Exact local reachability for classical-reversible prefixes (vectorised).

For a Toffoli g on qubits (a, b, t), condition (R) only needs the set L of local
basis strings on (a, b, t) that some valid input can produce at g's input. When the
prefix before g consists of X/CX/CCX/MCX/SWAP gates, every basis input maps to one
basis state, so L is computed EXACTLY by propagating all valid inputs through the
prefix. We do this with numpy on an array of up to 2^22 integers, which covers the
12-24-qubit circuits of the paper when their ancillas are pinned to |0>.

If the prefix contains any other gate the function returns None and the caller
falls back to the sound over-approximation of ``reachable_subspace``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

_CLASSICAL = {"x", "cx", "ccx", "mcx", "mcx_gray", "swap", "id", "barrier"}


def _inputs(n: int, pinned_zero: Iterable[int], max_free_bits: int):
    pinned = set(pinned_zero)
    free = [q for q in range(n) if q not in pinned]
    if len(free) > max_free_bits:
        return None
    k = len(free)
    idx = np.arange(1 << k, dtype=np.uint64)
    s = np.zeros(1 << k, dtype=np.uint64)
    for bit, q in enumerate(free):
        s |= ((idx >> np.uint64(bit)) & np.uint64(1)) << np.uint64(q)
    return s


def propagate_prefix(circuit, gate_index: int, pinned_zero=(), max_free_bits: int = 22):
    """Basis states (as uint64 array) at the input of ``gate_index``, or None."""
    n = circuit.num_qubits
    s = _inputs(n, pinned_zero, max_free_bits)
    if s is None:
        return None
    one = np.uint64(1)
    for inst in circuit.data[:gate_index]:
        name = inst.operation.name.lower()
        if name not in _CLASSICAL:
            return None
        qb = [circuit.find_bit(q).index for q in inst.qubits]
        if name in ("id", "barrier"):
            continue
        if name == "x":
            s ^= one << np.uint64(qb[0])
        elif name == "swap":
            a, b = qb
            ba = (s >> np.uint64(a)) & one
            bb = (s >> np.uint64(b)) & one
            diff = ba ^ bb
            s ^= (diff << np.uint64(a)) | (diff << np.uint64(b))
        else:  # cx / ccx / mcx: controls qb[:-1], target qb[-1]
            m = np.ones_like(s)
            for c in qb[:-1]:
                m &= (s >> np.uint64(c)) & one
            s ^= m << np.uint64(qb[-1])
    return s


def local_reachable(circuit, gate_index: int, qubits: Sequence[int], pinned_zero=(),
                    max_free_bits: int = 22) -> Optional[list]:
    """Exact set of local strings on ``qubits`` (bit j = qubits[j]) reaching
    ``gate_index``, or None if the prefix is not classical or too many free bits."""
    s = propagate_prefix(circuit, gate_index, pinned_zero, max_free_bits)
    if s is None:
        return None
    loc = np.zeros_like(s)
    for j, q in enumerate(qubits):
        loc |= ((s >> np.uint64(q)) & np.uint64(1)) << np.uint64(j)
    return sorted(int(v) for v in np.unique(loc))
