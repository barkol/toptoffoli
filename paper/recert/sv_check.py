"""Niezalezny test rownowaznosci podprocedury przez symulacje wektora stanu:
dla K losowych stanow (Haar na kubitach danych, przypiete = |0>) liczymy <psi|U^dag U'|psi>.
Rownowaznosc z jedna faza <=> |overlap|=1 dla kazdego stanu i ta sama faza dla wszystkich.
Zwraca (ok, min|ov|, rozrzut fazy)."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def random_input(n, pins, rng):
    data = [q for q in range(n) if q not in set(pins)]
    v = rng.normal(size=2 ** len(data)) + 1j * rng.normal(size=2 ** len(data)); v /= np.linalg.norm(v)
    full = np.zeros(2 ** n, dtype=complex)
    idx = np.zeros(2 ** len(data), dtype=np.int64)
    for k, q in enumerate(data):
        idx |= ((np.arange(2 ** len(data)) >> k) & 1) << q
    full[idx] = v
    return Statevector(full)


def sv_equiv(exact, cand, pins=(), K=3, seed=1, tol=1e-6):
    n = exact.num_qubits; rng = np.random.default_rng(seed); ovs = []
    ex = exact.remove_final_measurements(inplace=False); ca = cand.remove_final_measurements(inplace=False)
    for _ in range(K):
        psi = random_input(n, pins, rng)
        a = psi.evolve(ex); b = psi.evolve(ca)
        ovs.append(np.vdot(a.data, b.data))
    mags = [abs(o) for o in ovs]; ph = [np.angle(o) for o in ovs]
    spread = max(abs(np.angle(np.exp(1j * (p - ph[0])))) for p in ph)
    return bool(min(mags) > 1 - tol and spread < 1e-5), float(min(mags)), float(spread)
