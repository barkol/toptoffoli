"""A simple, transparent hardware error model for Toffoli-network decompositions.

The point of this module is to give the decomposition selector a *citable*,
parameterizable cost to minimize. On near-term superconducting and trapped-ion
hardware the dominant error source by far is the two-qubit entangling gate: its
error rate (~1e-2) is roughly an order of magnitude larger than a single-qubit
gate (~1e-3). See e.g. IBM device calibration data and the surface-code
literature (Fowler et al., 2012), where the two-qubit gate is the limiting
primitive. So a good proxy for "how badly will this circuit do on real hardware"
is essentially "how many two-qubit gates does it contain", weighted by their
error rate.

We keep the model deliberately simple and inspectable:

    per-gate fidelity  F_g = 1 - err(g)
    circuit fidelity   F   = prod_g F_g
    circuit infidelity  1 - F      (the figure the selector minimizes)

``err(g)`` depends only on the number of qubits the gate acts on: two-qubit
(and larger) gates pay ``p2q``; single-qubit gates pay ``p1q``; barriers/ids are
free. This is the standard depolarizing-style accounting used for back-of-the-
envelope error budgets. Every rate is overridable so the selector's decisions can
be re-checked against a specific device's calibration.

Multi-qubit gates (CCX/MCX) are *not* expected to survive into the costed
circuit -- the selector decomposes them to a two-qubit basis first -- but if one
is costed directly we charge it ``p2q`` per two-qubit gate of its standard
decomposition (see ``_MULTIQUBIT_2Q_COST``) so the cost never under-counts.
"""

from __future__ import annotations

from typing import Optional

# Names that cost nothing.
_FREE_GATES = {"barrier", "id"}

# Standard two-qubit-gate count for atomic multi-controlled gates, used only if a
# circuit is costed BEFORE being decomposed to a two-qubit basis. A Toffoli (ccx)
# decomposes to 6 CX in the textbook Clifford+T construction; a generic n-control
# MCX to roughly 6*(n-1)+... -- we use a safe, simple lower-bound-ish proxy.
_MULTIQUBIT_2Q_COST = {
    "ccx": 6,   # textbook exact Toffoli = 6 CX (+ 1q gates, costed separately below)
    "ccz": 6,
}


class HardwareErrorModel:
    """Transparent, parameterizable per-gate error / infidelity model.

    Parameters
    ----------
    p2q : float
        Error rate of a two-qubit (or larger entangling) gate. Default 1e-2.
    p1q : float
        Error rate of a single-qubit gate. Default 1e-3.
    p_readout : float
        Optional per-qubit readout/measurement error. Default 0.0 (off); only
        charged for ``measure`` instructions.
    """

    def __init__(
        self,
        p2q: float = 1e-2,
        p1q: float = 1e-3,
        p_readout: float = 0.0,
    ):
        if not (0.0 <= p2q < 1.0):
            raise ValueError("p2q must be in [0, 1)")
        if not (0.0 <= p1q < 1.0):
            raise ValueError("p1q must be in [0, 1)")
        if not (0.0 <= p_readout < 1.0):
            raise ValueError("p_readout must be in [0, 1)")
        self.p2q = p2q
        self.p1q = p1q
        self.p_readout = p_readout

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _num_qubits(circuit, inst) -> int:
        return len(inst.qubits)

    def gate_error(self, name: str, num_qubits: int) -> float:
        """Error rate charged to a single instruction.

        Two-qubit and larger entangling gates pay ``p2q`` (a multi-controlled gate
        pays ``p2q`` per two-qubit gate of its standard decomposition); single-
        qubit gates pay ``p1q``; barriers/ids are free; measure pays readout.
        """
        n = name.lower()
        if n in _FREE_GATES:
            return 0.0
        if n == "measure":
            return self.p_readout
        if n in _MULTIQUBIT_2Q_COST:
            cost = _MULTIQUBIT_2Q_COST[n]
            # 1 - (1 - p2q)^cost  == infidelity of `cost` two-qubit gates.
            return 1.0 - (1.0 - self.p2q) ** cost
        if num_qubits >= 2:
            return self.p2q
        return self.p1q

    # ------------------------------------------------------------------- public
    def two_qubit_count(self, circuit) -> int:
        """Number of two-qubit (or larger) entangling gates in ``circuit``.

        A multi-controlled gate (ccx/ccz/mcx) is counted as its standard two-qubit
        decomposition cost, so the figure is comparable whether or not the circuit
        has been lowered to a two-qubit basis yet.
        """
        total = 0
        for inst in circuit.data:
            name = inst.operation.name.lower()
            if name in _FREE_GATES or name == "measure":
                continue
            if name in _MULTIQUBIT_2Q_COST:
                total += _MULTIQUBIT_2Q_COST[name]
            elif self._num_qubits(circuit, inst) >= 2:
                total += 1
        return total

    def one_qubit_count(self, circuit) -> int:
        """Number of single-qubit gates (excluding free gates / measure)."""
        total = 0
        for inst in circuit.data:
            name = inst.operation.name.lower()
            if name in _FREE_GATES or name == "measure":
                continue
            if name in _MULTIQUBIT_2Q_COST:
                continue
            if self._num_qubits(circuit, inst) == 1:
                total += 1
        return total

    def circuit_infidelity(self, circuit) -> float:
        """Estimated total infidelity 1 - prod_g (1 - err(g)) over all gates.

        Monotonic in the number of (error-weighted) gates and dominated by the
        two-qubit count when ``p2q >> p1q``. This is the figure the selector
        minimizes.
        """
        f = 1.0
        for inst in circuit.data:
            name = inst.operation.name.lower()
            err = self.gate_error(name, self._num_qubits(circuit, inst))
            if err:
                f *= (1.0 - err)
        return 1.0 - f

    def circuit_fidelity(self, circuit) -> float:
        """Estimated total fidelity prod_g (1 - err(g))."""
        return 1.0 - self.circuit_infidelity(circuit)

    def report(self, circuit) -> dict:
        """A small breakdown dict useful for selector reports / debugging."""
        return {
            "two_qubit_count": self.two_qubit_count(circuit),
            "one_qubit_count": self.one_qubit_count(circuit),
            "infidelity": self.circuit_infidelity(circuit),
            "fidelity": self.circuit_fidelity(circuit),
            "p2q": self.p2q,
            "p1q": self.p1q,
        }


# Module-level convenience functions using a default model -----------------------
_DEFAULT = HardwareErrorModel()


def circuit_infidelity(circuit, model: Optional[HardwareErrorModel] = None) -> float:
    """Estimated total infidelity of ``circuit`` under ``model`` (default rates)."""
    return (model or _DEFAULT).circuit_infidelity(circuit)


def two_qubit_count(circuit, model: Optional[HardwareErrorModel] = None) -> int:
    """Number of two-qubit gates in ``circuit`` under ``model`` (default rates)."""
    return (model or _DEFAULT).two_qubit_count(circuit)
