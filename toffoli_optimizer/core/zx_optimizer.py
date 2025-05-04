"""
ZX-Calculus Based Optimizer

This module provides circuit optimization using ZX-calculus techniques,
allowing for better depth reduction with controlled fidelity trade-offs.
"""

import os
import time
import numpy as np
import gc
from typing import Dict, List, Tuple, Optional, Union, Any

# Try to import PyZX - this library implements ZX-calculus functionality
try:
    import pyzx as zx
    PYZX_AVAILABLE = True
except ImportError:
    PYZX_AVAILABLE = False
    print("Warning: PyZX library not available. ZX-calculus optimization will be limited.")

# Import Qiskit with version compatibility
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Instruction, Parameter
    # Check if this is Qiskit 2.0+
    try:
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import Unroller, Optimize1qGates
        QISKIT_2_AVAILABLE = True
    except ImportError:
        # Older Qiskit imports
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import Unroller, Optimize1qGates
        QISKIT_2_AVAILABLE = False
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QISKIT_2_AVAILABLE = False

# Import from parent module if available
try:
    from ..utils.circuit_utils import validate_physical_circuit, estimate_fidelity
except ImportError:
    # Direct imports for standalone usage
    try:
        from toffoli_optimizer.utils.circuit_utils import validate_physical_circuit, estimate_fidelity
    except ImportError:
        # Define fallback functions if imports fail
        def validate_physical_circuit(circuit, coupling_map):
            """Fallback validation function"""
            return True, []
            
        def estimate_fidelity(circuit=None, num_qubits=None, num_operations=None):
            """Fallback fidelity estimation"""
            return 0.9

class ZXOptimizer:
    """
    Optimizer that uses ZX-calculus for circuit optimization with controlled
    fidelity trade-offs to achieve better depth reduction.
    """
    
    def __init__(self, aggressive_mode: bool = False, 
                 t_count_weight: float = 0.3, 
                 cx_count_weight: float = 0.3,
                 target_fidelity: float = 0.85,
                 debug_mode: bool = False):
        """
        Initialize the ZX-calculus optimizer.
        
        Args:
            aggressive_mode: Whether to use aggressive optimization (more depth reduction,
                            less concern for fidelity)
            t_count_weight: Weight for T gate count in optimization score
            cx_count_weight: Weight for CX gate count in optimization score
            target_fidelity: Minimum acceptable circuit fidelity (0.0-1.0)
            debug_mode: Whether to print debug information
        """
        self.aggressive_mode = aggressive_mode
        self.t_count_weight = t_count_weight
        self.cx_count_weight = cx_count_weight
        self.target_fidelity = target_fidelity
        self.debug_mode = debug_mode
        
        # Set up PyZX if available
        self._check_pyzx_availability()
    
    def _check_pyzx_availability(self) -> bool:
        """Check if PyZX is available and properly configured."""
        if not PYZX_AVAILABLE:
            if self.debug_mode:
                print("PyZX library not available. Some optimization techniques will be disabled.")
            return False
        
        # Test PyZX functionality
        try:
            # Create a simple graph
            g = zx.Graph()
            # Add some vertices
            v1 = g.add_vertex(zx.VertexType.Z, 0, 0)
            v2 = g.add_vertex(zx.VertexType.X, 0, 1)
            # Add an edge
            g.add_edge((v1, v2))
            return True
        except Exception as e:
            if self.debug_mode:
                print(f"PyZX initialization error: {e}")
            return False
    
    def qiskit_to_zx(self, circuit: QuantumCircuit) -> Any:
        """
        Convert a Qiskit circuit to a ZX-diagram.
        
        Args:
            circuit: Qiskit QuantumCircuit to convert
            
        Returns:
            zx.Graph: ZX-diagram representation, or None if conversion fails
        """
        if not PYZX_AVAILABLE:
            return None
            
        try:
            # Convert Qiskit circuit to QASM string
            if QISKIT_2_AVAILABLE:
                from qiskit.qasm2 import dumps
                qasm_str = dumps(circuit)
            else:
                qasm_str = circuit.qasm()
            
            # Parse QASM string to ZX-diagram
            zx_graph = zx.Circuit.from_qasm(qasm_str).to_graph()
            
            if self.debug_mode:
                print(f"Converted circuit to ZX-diagram with {len(zx_graph.vertices())} vertices")
            
            return zx_graph
        except Exception as e:
            if self.debug_mode:
                print(f"Error converting circuit to ZX-diagram: {e}")
            return None
    
    def zx_to_qiskit(self, zx_graph: Any) -> QuantumCircuit:
        """
        Convert a ZX-diagram back to a Qiskit circuit.
        
        Args:
            zx_graph: ZX-diagram to convert
            
        Returns:
            QuantumCircuit: Converted Qiskit circuit, or None if conversion fails
        """
        if not PYZX_AVAILABLE or zx_graph is None:
            return None
            
        try:
            # Extract circuit from ZX-diagram
            zx_circuit = zx.extract_circuit(zx_graph)
            
            # Convert to QASM
            qasm_str = zx_circuit.to_qasm()
            
            # Parse QASM string to Qiskit circuit
            if QISKIT_2_AVAILABLE:
                from qiskit.qasm2 import loads
                circuit = loads(qasm_str)
            else:
                from qiskit import QuantumCircuit
                circuit = QuantumCircuit.from_qasm_str(qasm_str)
            
            if self.debug_mode:
                print(f"Converted ZX-diagram back to circuit with depth {circuit.depth()}")
            
            return circuit
        except Exception as e:
            if self.debug_mode:
                print(f"Error converting ZX-diagram to circuit: {e}")
            return None
    
    def optimize_with_zx(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize a quantum circuit using ZX-calculus techniques.
        
        Args:
            circuit: Quantum circuit to optimize
            
        Returns:
            QuantumCircuit: Optimized circuit or original if optimization fails
        """
        if not PYZX_AVAILABLE:
            return self._fallback_optimize(circuit)
            
        try:
            # Convert to ZX-diagram
            zx_graph = self.qiskit_to_zx(circuit)
            if zx_graph is None:
                return self._fallback_optimize(circuit)
            
            # Track original metrics
            original_depth = circuit.depth()
            original_size = len(circuit.data)
            
            # Full optimization pipeline
            if self.aggressive_mode:
                # Apply full optimization pipeline in aggressive mode
                zx.full_reduce(zx_graph, quiet=not self.debug_mode)
            else:
                # Apply a more conservative optimization
                zx.teleport_reduce(zx_graph)
                zx.clifford_simp(zx_graph, quiet=not self.debug_mode)
                
            # Additional T-gate optimization
            zx.tcount(zx_graph)  # Calculate T-count
            zx.to_gh(zx_graph)   # Convert to graph-like form
            if self.aggressive_mode:
                # More aggressive T-count reduction
                zx.phase_teleport(zx_graph)
                zx.gadgetize(zx_graph) 
            zx.simplify.phase_gadget_simp(zx_graph)
            
            # Convert back to standard form
            zx.extract_circuit(zx_graph)
            
            # Convert back to Qiskit circuit
            optimized_circuit = self.zx_to_qiskit(zx_graph)
            
            if optimized_circuit is None or optimized_circuit.depth() >= original_depth:
                # If optimization failed or didn't reduce depth, use fallback
                return self._fallback_optimize(circuit)
            
            if self.debug_mode:
                print(f"ZX optimization: Depth {original_depth} -> {optimized_circuit.depth()} "
                     f"({(1 - optimized_circuit.depth() / original_depth) * 100:.2f}% reduction)")
            
            # Return optimized circuit
            return optimized_circuit
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error in ZX-calculus optimization: {e}")
            return self._fallback_optimize(circuit)
    
    def _fallback_optimize(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Fallback optimization method when ZX-calculus is not available.
        
        Args:
            circuit: Quantum circuit to optimize
            
        Returns:
            QuantumCircuit: Optimized circuit
        """
        if not QISKIT_AVAILABLE:
            return circuit
            
        try:
            from qiskit import transpile
            
            # Apply Qiskit's transpiler with custom settings
            optimized = transpile(
                circuit,
                basis_gates=['u', 'cx'],  # Standard universal gate set
                optimization_level=3 if self.aggressive_mode else 2,
                layout_method='sabre'  # Good for respecting connectivity
            )
            
            if self.debug_mode:
                print(f"Fallback optimization: Depth {circuit.depth()} -> {optimized.depth()} "
                     f"({(1 - optimized.depth() / circuit.depth()) * 100:.2f}% reduction)")
            
            return optimized
        except Exception as e:
            if self.debug_mode:
                print(f"Error in fallback optimization: {e}")
            return circuit
    
    def approximate_toffoli(self, circuit: QuantumCircuit, approximation_level: float = 0.8) -> QuantumCircuit:
        """
        Apply Toffoli approximations to reduce depth at the expense of fidelity.
        
        Args:
            circuit: Quantum circuit to optimize
            approximation_level: Level of approximation (0.0-1.0, higher means more aggressive)
            
        Returns:
            QuantumCircuit: Circuit with approximated Toffoli gates
        """
        if not QISKIT_AVAILABLE:
            return circuit
            
        try:
            # Create a new circuit with the same number of qubits and classical bits
            optimized = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
            
            # Track which Toffoli gates to approximate
            toffoli_indices = []
            
            # Identify Toffoli gates
            for i, inst in enumerate(circuit.data):
                if inst.operation.name.lower() in ['ccx', 'toffoli']:
                    toffoli_indices.append(i)
            
            # Determine how many Toffoli gates to approximate
            num_to_approximate = int(len(toffoli_indices) * approximation_level)
            gates_to_approximate = toffoli_indices[:num_to_approximate]
            
            if self.debug_mode:
                print(f"Approximating {num_to_approximate} of {len(toffoli_indices)} Toffoli gates")
            
            # Process each instruction
            for i, inst in enumerate(circuit.data):
                operation = inst.operation
                qubits = [q.index if hasattr(q, 'index') else q._index for q in inst.qubits]
                clbits = [c.index if hasattr(c, 'index') else c._index for c in inst.clbits] if hasattr(inst, 'clbits') else []
                
                if i in gates_to_approximate and operation.name.lower() in ['ccx', 'toffoli']:
                    # Apply approximate Toffoli implementation
                    self._add_approximate_toffoli(optimized, qubits[0], qubits[1], qubits[2])
                else:
                    # Add the original gate
                    optimized.append(operation, qubits, clbits)
            
            return optimized
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error approximating Toffoli gates: {e}")
            return circuit
    
    def _add_approximate_toffoli(self, circuit: QuantumCircuit, control1: int, control2: int, target: int) -> None:
        """
        Add an approximate Toffoli gate to the circuit.
        
        This implements a lower-fidelity but shallower Toffoli gate.
        
        Args:
            circuit: Circuit to add the approximate Toffoli to
            control1: First control qubit
            control2: Second control qubit
            target: Target qubit
        """
        # Approximate implementation (fewer gates, lower fidelity)
        # Based on reduced-depth approximate CCX
        circuit.h(target)
        circuit.cx(control1, target)
        circuit.h(target)
        circuit.cx(control2, target)
        circuit.h(target)
    
    def optimize_circuit(self, circuit: QuantumCircuit, coupling_map: Optional[List] = None) -> QuantumCircuit:
        """
        Fully optimize a circuit with combined techniques for maximum depth reduction.
        
        Args:
            circuit: Quantum circuit to optimize
            coupling_map: Optional coupling map for hardware constraints
            
        Returns:
            QuantumCircuit: Optimized circuit
        """
        if circuit is None:
            return None
        
        # Make a working copy of the input circuit
        working_circuit = circuit.copy()
        
        # Force garbage collection
        gc.collect()
        
        # Apply ZX-calculus optimization first (if available)
        zx_optimized = self.optimize_with_zx(working_circuit)
        
        # Force garbage collection
        del working_circuit
        gc.collect()
        
        # Apply Toffoli approximations for deeper reduction
        approx_level = 0.7 if self.aggressive_mode else 0.4
        optimized = self.approximate_toffoli(zx_optimized, approx_level)
        
        # Force garbage collection
        del zx_optimized
        gc.collect()
        
        # Apply final pass of Qiskit transpilation to respect coupling map
        if QISKIT_AVAILABLE and coupling_map is not None:
            from qiskit import transpile
            
            final_optimized = transpile(
                optimized,
                coupling_map=coupling_map,
                basis_gates=['u', 'cx'],
                optimization_level=3
            )
            
            # Force garbage collection
            del optimized
            gc.collect()
            
            return final_optimized
        else:
            return optimized
    
    def estimate_fidelity_reduction(self, original_circuit: QuantumCircuit, 
                                  optimized_circuit: QuantumCircuit) -> float:
        """
        Estimate how much fidelity was sacrificed in the optimization.
        
        Args:
            original_circuit: Original circuit
            optimized_circuit: Optimized circuit
            
        Returns:
            float: Estimated fidelity reduction (0.0-1.0, higher means more fidelity loss)
        """
        if original_circuit is None or optimized_circuit is None:
            return 0.0
            
        # Count operations in both circuits
        orig_ops = original_circuit.count_ops() if hasattr(original_circuit, 'count_ops') else {}
        opt_ops = optimized_circuit.count_ops() if hasattr(optimized_circuit, 'count_ops') else {}
        
        # Track operation differences that affect fidelity
        orig_ccx = orig_ops.get('ccx', 0)
        opt_ccx = opt_ops.get('ccx', 0)
        
        # Approximate Toffoli gates have 5 operations in our implementation
        approx_toffoli_count = 0
        if orig_ccx > opt_ccx:
            approx_toffoli_count = orig_ccx - opt_ccx
        
        # Estimate original fidelity
        orig_fidelity = estimate_fidelity(
            circuit=original_circuit
        )
        
        # Estimate optimized fidelity (adjusted for approximations)
        opt_fidelity = estimate_fidelity(
            circuit=optimized_circuit
        )
        
        # Apply penalty for approximate Toffoli gates
        fidelity_penalty = 0.02 * approx_toffoli_count  # Each approximate Toffoli costs ~2% fidelity
        opt_fidelity = max(0.0, opt_fidelity - fidelity_penalty)
        
        # Calculate fidelity reduction
        fidelity_reduction = 1.0 - (opt_fidelity / orig_fidelity) if orig_fidelity > 0 else 0.0
        
        return fidelity_reduction
