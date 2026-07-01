"""CLI runner for the Toffoli-COUNT reduction mode.

Loads a Toffoli network, builds it as an ATOMIC CCX/MCX circuit (no decomposition),
runs ToffoliCountReducer to minimise the NUMBER of multi-controlled gates -- every
rewrite verified function-preserving by ExactEquivalenceVerifier -- and writes the
result to QASM. Optionally accepts rewrites valid only up to an output-wire
permutation (which it tracks and reports).
"""

from qiskit import QuantumCircuit

from toffoli_optimizer.utils.io_utils import (
    define_loaded_toffoli_network,
    save_circuit_to_qasm,
)
from toffoli_optimizer.core.toffoli_count_reducer import ToffoliCountReducer, count_toffoli


class ReduceCount:
    """Run atomic Toffoli-count reduction on a single network."""

    def __init__(self, args):
        self.args = args

    def _build_atomic_circuit(self, toffoli_gates):
        """Build a QuantumCircuit of atomic x / cx / ccx / mcx from loaded gates.

        `toffoli_gates` is a list of (control_qubits, target_qubit); an empty
        control list is an X gate.
        """
        max_idx = 0
        for controls, target in toffoli_gates:
            for q in list(controls) + [target]:
                max_idx = max(max_idx, int(q))
        n = max(int(self.args.num_qubits or 0), max_idx + 1)
        qc = QuantumCircuit(n)
        for controls, target in toffoli_gates:
            controls = [int(c) for c in controls]
            target = int(target)
            if len(controls) == 0:
                qc.x(target)
            elif len(controls) == 1:
                qc.cx(controls[0], target)
            elif len(controls) == 2:
                qc.ccx(controls[0], controls[1], target)
            else:
                qc.mcx(controls, target)
        return qc

    def run(self):
        print(f"Running Toffoli-count reduction on {self.args.input}")
        result = define_loaded_toffoli_network(self.args.input)
        if not result:
            print(f"Error: Failed to load Toffoli network from {self.args.input}")
            return 1
        toffoli_gates, _output_qubits, _input_qubits = result
        if not toffoli_gates:
            print("Error: no ccx/mcx/x gates found in the input network")
            return 1

        circuit = self._build_atomic_circuit(toffoli_gates)
        allow_perm = bool(getattr(self.args, "allow_permutation", False))
        reducer = ToffoliCountReducer(
            enable_fanout_cse=not getattr(self.args, "no_fanout_cse", False),
            enable_expand_to_cancel=not getattr(self.args, "no_expand_to_cancel", False),
            allow_permutation=allow_perm,
        )

        before = count_toffoli(circuit)
        reduced = reducer.reduce_toffoli_count(circuit)
        after = count_toffoli(reduced)
        rep = reducer.report

        print(f"\nAtomic Toffoli (CCX/MCX) count: {before} -> {after}  "
              f"(removed {before - after})")
        print(f"  rounds={rep.get('rounds')}  rewrites={rep.get('rewrites_applied')}")
        print(f"  all rewrites verified equivalent: {rep.get('all_rewrites_verified')}")
        perm = rep.get("output_permutation") or {}
        if any(int(k) != int(v) for k, v in perm.items()):
            print(f"  OUTPUT-WIRE PERMUTATION (apply at read-out): {perm}")
        elif allow_perm:
            print("  output-wire permutation: identity")

        if save_circuit_to_qasm(reduced, self.args.output):
            print(f"Saved reduced circuit to {self.args.output}")
            return 0
        print("Error: failed to write output QASM")
        return 1
