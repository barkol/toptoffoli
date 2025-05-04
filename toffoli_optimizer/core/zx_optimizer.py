"""
ZX-Calculus Based Optimizer

This module provides circuit optimization using ZX-calculus techniques,
allowing for better depth reduction with controlled fidelity trade-offs.
"""

import os
import time
import numpy as np
import traceback
import gc
from typing import Dict, List, Tuple, Optional, Union, Any

# Check for PyZX availability
try:
    import pyzx as zx
    PYZX_AVAILABLE = True
except ImportError:
    PYZX_AVAILABLE = False
    print("Warning: PyZX library not available. ZX-calculus optimization will be limited.")

# Import Qiskit with version compatibility checks
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Instruction, Parameter
    
    # Check if this is Qiskit 2.0+
    try:
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import Unroller, Optimize1qGates
        from qiskit.qasm2 import dumps, loads
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
    print("Warning: Qiskit not available. Circuit optimization will be limited.")

class ZXOptimizer:
    """
    Optimizer that uses ZX-calculus for circuit optimization with controlled
    fidelity trade-offs to achieve better depth reduction.
    """
    
    def __init__(self, aggressive_mode=False,
                 t_count_weight=0.3,
                 cx_count_weight=0.3,
                 target_fidelity=0.85,
                 debug_mode=False):
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
        
        # Check if PyZX is available
        self.pyzx_available = PYZX_AVAILABLE
        
        if self.debug_mode:
            print(f"ZXOptimizer initialized with:")
            print(f"  Aggressive mode: {self.aggressive_mode}")
            print(f"  Target fidelity: {self.target_fidelity}")
            print(f"  PyZX available: {self.pyzx_available}")
            
    def circuit_to_zx(self, circuit):
        """
        Convert a Qiskit circuit to a ZX-diagram.
        
        Args:
            circuit: Qiskit circuit to convert
            
        Returns:
            zx.Graph: ZX-diagram or None if conversion fails
        """
        if not self.pyzx_available or circuit is None:
            return None
            
        try:
            # Convert circuit to QASM
            if QISKIT_2_AVAILABLE:
                qasm_str = dumps(circuit)
            else:
                qasm_str = circuit.qasm()
                
            # Parse QASM to create ZX graph
            zx_circuit = zx.Circuit.from_qasm(qasm_str)
            zx_graph = zx_circuit.to_graph()
            
            if self.debug_mode:
                print(f"Converted circuit to ZX-diagram with {len(zx_graph.vertices())} vertices")
                
            return zx_graph
        except Exception as e:
            if self.debug_mode:
                print(f"Error converting circuit to ZX-diagram: {e}")
                traceback.print_exc()
            return None
            
    def zx_to_circuit(self, zx_graph):
        """
        Convert a ZX-diagram back to a Qiskit circuit.
        
        Args:
            zx_graph: ZX-diagram to convert
            
        Returns:
            QuantumCircuit: Converted circuit or None if conversion fails
        """
        if not self.pyzx_available or zx_graph is None:
            return None
            
        try:
            # Extract circuit from ZX graph
            zx_circuit = zx.extract_circuit(zx_graph)
            
            # Convert to QASM
            qasm_str = zx_circuit.to_qasm()
            
            # Parse QASM back to Qiskit circuit
            if QISKIT_2_AVAILABLE:
                circuit = loads(qasm_str)
            else:
                circuit = QuantumCircuit.from_qasm_str(qasm_str)
                
            if self.debug_mode:
                print(f"Converted ZX-diagram back to circuit with depth {circuit.depth()}")
                
            return circuit
        except Exception as e:
            if self.debug_mode:
                print(f"Error converting ZX-diagram to circuit: {e}")
                traceback.print_exc()
            return None
    
    def optimize_with_zx(self, circuit):
        """
        Optimize a quantum circuit using ZX-calculus techniques.
        
        Args:
            circuit: Quantum circuit to optimize
            
        Returns:
            QuantumCircuit: Optimized circuit or original circuit if optimization fails
        """
        if not self.pyzx_available or circuit is None:
            return self._fallback_optimize(circuit)
            
        try:
            # Convert circuit to ZX diagram
            zx_graph = self.circuit_to_zx(circuit)
            if zx_graph is None:
                return self._fallback_optimize(circuit)
            
            # Record original metrics
            original_depth = circuit.depth()
            original_size = len(circuit.data)
            
            # Apply optimization based on aggressiveness setting
            if self.aggressive_mode:
                # Apply full optimization pipeline in aggressive mode
                if self.debug_mode:
                    print("Applying aggressive ZX optimization")
                zx.full_reduce(zx_graph, quiet=not self.debug_mode)
            else:
                # Apply a more balanced optimization
                if self.debug_mode:
                    print("Applying standard ZX optimization")
                zx.clifford_simp(zx_graph, quiet=not self.debug_mode)
            
            # Phase gadget optimization (good for T-count reduction)
            zx.to_gh(zx_graph)
            if self.aggressive_mode:
                zx.phase_teleport(zx_graph)
            zx.simplify.phase_gadget_simp(zx_graph)
            
            # Extract optimized circuit
            optimized_circuit = self.zx_to_circuit(zx_graph)
            
            # Verify the optimization was successful
            if optimized_circuit is None:
                if self.debug_mode:
                    print("ZX extraction failed, using fallback optimization")
                return self._fallback_optimize(circuit)
                
            # Perform a light clean-up optimization
            optimized_circuit = transpile(optimized_circuit, optimization_level=1)
            
            # Log the results
            if self.debug_mode:
                new_depth = optimized_circuit.depth()
                new_size = len(optimized_circuit.data)
                depth_reduction = (original_depth - new_depth) / original_depth * 100 if original_depth > 0 else 0
                print(f"ZX optimization complete:")
                print(f"  Original depth: {original_depth} -> New depth: {new_depth}")
                print(f"  Depth reduction: {depth_reduction:.2f}%")
                print(f"  Gates: {original_size} -> {new_size}")
            
            # Force garbage collection to free memory
            gc.collect()
            
            return optimized_circuit
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error during ZX optimization: {e}")
                traceback.print_exc()
            
            # Fallback to standard transpiler optimization
            return self._fallback_optimize(circuit)
            
    def _fallback_optimize(self, circuit):
        """
        Fallback optimization method when ZX-calculus is not available or fails.
        
        Args:
            circuit: Quantum circuit to optimize
            
        Returns:
            QuantumCircuit: Optimized circuit
        """
        if not QISKIT_AVAILABLE or circuit is None:
            return circuit
            
        try:
            if self.debug_mode:
                print("Using fallback optimization (Qiskit transpiler)")
                
            # Adjust optimization level based on aggressiveness
            opt_level = 3 if self.aggressive_mode else 2
            
            # Create optimization passes
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import Unroller, Optimize1qGates
            
            pass_manager = PassManager()
            pass_manager.append(Unroller(['u', 'cx']))
            pass_manager.append(Optimize1qGates())
            
            # Apply passes
            optimized = pass_manager.run(circuit)
            
            # Final transpile to fully optimize
            optimized = transpile(
                optimized,
                basis_gates=['u', 'cx'],
                optimization_level=opt_level
            )
            
            if self.debug_mode:
                depth_reduction = (circuit.depth() - optimized.depth()) / circuit.depth() * 100 if circuit.depth() > 0 else 0
                print(f"Fallback optimization complete:")
                print(f"  Original depth: {circuit.depth()} -> New depth: {optimized.depth()}")
                print(f"  Depth reduction: {depth_reduction:.2f}%")
            
            return optimized
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error during fallback optimization: {e}")
                traceback.print_exc()
            
            # Return original circuit if optimization fails
            return circuit
    
    def optimize_circuit(self, circuit, coupling_map=None):
        """
        Optimize a quantum circuit using the best available methods.
        
        This method combines ZX-calculus optimization with hardware mapping
        to produce a circuit optimized for both depth and connectivity constraints.
        
        Args:
            circuit: Quantum circuit to optimize
            coupling_map: Optional coupling map for hardware constraints
            
        Returns:
            QuantumCircuit: Optimized circuit
        """
        if circuit is None:
            return None
            
        if self.debug_mode:
            print(f"Optimizing circuit with depth {circuit.depth()}, gate count {len(circuit.data)}")
            print(f"Target fidelity: {self.target_fidelity}")
            print(f"Coupling map provided: {coupling_map is not None}")
        
        # First optimize the circuit structure using ZX-calculus
        optimized_circuit = self.optimize_with_zx(circuit)
        
        # If we're being aggressive, try approximate Toffoli gates
        if self.aggressive_mode and hasattr(optimized_circuit, 'depth') and optimized_circuit.depth() > 10:
            approximation_level = 0.7 if self.target_fidelity < 0.9 else 0.4
            if self.debug_mode:
                print(f"Applying approximate Toffoli gates (level {approximation_level})")
            approximated_circuit = self.approximate_toffoli(optimized_circuit, approximation_level)
            
            # Only use approximation if it actually reduces depth
            if approximated_circuit.depth() < optimized_circuit.depth():
                optimized_circuit = approximated_circuit
                if self.debug_mode:
                    print(f"Approximation reduced depth to {optimized_circuit.depth()}")
        
        # Map to hardware if coupling map is provided
        if coupling_map is not None:
            mapped_circuit = self.map_to_hardware(optimized_circuit, coupling_map)
            
            if self.debug_mode:
                print(f"Hardware mapping: depth {optimized_circuit.depth()} -> {mapped_circuit.depth()}")
            
            # If mapping significantly increased depth, try a different approach
            if mapped_circuit.depth() > optimized_circuit.depth() * 1.5:
                if self.debug_mode:
                    print("Mapping significantly increased depth, trying alternative mapping")
                
                # Try direct transpilation with a different strategy
                alt_mapped = transpile(
                    optimized_circuit,
                    coupling_map=coupling_map,
                    layout_method='sabre',
                    routing_method='stochastic',
                    optimization_level=3
                )
                
                # Use the better result
                if alt_mapped.depth() < mapped_circuit.depth():
                    mapped_circuit = alt_mapped
                    if self.debug_mode:
                        print(f"Alternative mapping found better result: depth {mapped_circuit.depth()}")
            
            # Use the mapped circuit
            final_circuit = mapped_circuit
        else:
            # No coupling constraints - just clean up the circuit
            final_circuit = transpile(optimized_circuit, optimization_level=1)
        
        # Log final metrics
        if self.debug_mode:
            original_depth = circuit.depth()
            final_depth = final_circuit.depth()
            depth_reduction = (original_depth - final_depth) / original_depth * 100 if original_depth > 0 else 0
            print(f"Optimization complete:")
            print(f"  Original depth: {original_depth} -> Final depth: {final_depth}")
            print(f"  Depth reduction: {depth_reduction:.2f}%")
            print(f"  Original gates: {len(circuit.data)} -> Final gates: {len(final_circuit.data)}")
        
        # Force garbage collection to free memory
        gc.collect()
        
        return final_circuit
    
    def map_to_hardware(self, circuit, coupling_map):
        """
        Map a circuit to hardware respecting coupling constraints.
        
        Args:
            circuit: Circuit to map
            coupling_map: Coupling map constraints
            
        Returns:
            QuantumCircuit: Hardware-mapped circuit
        """
        if not QISKIT_AVAILABLE or circuit is None:
            return circuit
            
        try:
            if self.debug_mode:
                print("Mapping circuit to hardware...")
                
            # Use different layout methods depending on circuit size
            if circuit.num_qubits < 10:
                layout_method = 'dense'
            else:
                layout_method = 'sabre'
                
            # Perform the mapping
            mapped_circuit = transpile(
                circuit,
                coupling_map=coupling_map,
                layout_method=layout_method,
                routing_method='sabre',
                optimization_level=1  # Light optimization to preserve ZX benefits
            )
            
            return mapped_circuit
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error mapping to hardware: {e}")
                
            # Try a simpler approach
            try:
                return transpile(
                    circuit,
                    coupling_map=coupling_map,
                    optimization_level=1
                )
            except:
                # If all else fails, return the original circuit
                return circuit
    
    def estimate_fidelity(self, circuit):
        """
        Estimate the fidelity of a quantum circuit.
        
        Args:
            circuit: Quantum circuit to estimate fidelity for
            
        Returns:
            float: Estimated fidelity (0.0-1.0)
        """
        if not QISKIT_AVAILABLE or circuit is None:
            return 0.9  # Default value
            
        try:
            # Count gates
            gate_counts = circuit.count_ops()
            
            # Extract specific gate counts
            cx_count = gate_counts.get('cx', 0)
            t_count = gate_counts.get('t', 0) + gate_counts.get('tdg', 0)
            single_qubit_gates = sum(gate_counts.get(g, 0) for g in
                                    ['h', 'x', 'y', 'z', 's', 'sdg', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz'])
            
            # Gate error rates
            cx_error = 0.01  # 1% error per CNOT gate
            t_error = 0.002  # 0.2% error per T gate
            single_error = 0.0001  # 0.01% error per single-qubit gate
            
            # Compute total fidelity
            fidelity = (1 - cx_error) ** cx_count * \
                       (1 - t_error) ** t_count * \
                       (1 - single_error) ** single_qubit_gates
            
            # Account for depth effects
            depth = circuit.depth()
            num_qubits = circuit.num_qubits
            
            # Simple decoherence model
            depth_factor = max(0.9, 1.0 - (0.001 * depth * np.log(1 + num_qubits)))
            fidelity *= depth_factor
            
            # Ensure fidelity is in valid range
            fidelity = max(0.0, min(1.0, fidelity))
            
            return fidelity
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error estimating fidelity: {e}")
            
            # Default fallback value
            return 0.9
    
    def approximate_toffoli(self, circuit, approximation_level=0.5):
        """
        Apply Toffoli approximations to reduce depth at the expense of fidelity.
        
        Args:
            circuit: Circuit to optimize
            approximation_level: Level of approximation (0.0-1.0, higher is more aggressive)
            
        Returns:
            QuantumCircuit: Circuit with approximated Toffoli gates
        """
        if not QISKIT_AVAILABLE or circuit is None:
            return circuit
            
        try:
            # Create a new circuit with the same structure
            result = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
            
            # Count total Toffoli gates
            toffoli_count = 0
            for inst in circuit.data:
                if inst.operation.name == 'ccx' or inst.operation.name == 'mcx':
                    toffoli_count += 1
            
            # Determine how many Toffoli gates to approximate
            num_to_approximate = int(toffoli_count * approximation_level)
            approximated = 0
            
            # Process each gate
            for inst in circuit.data:
                if inst.operation.name == 'ccx' and approximated < num_to_approximate:
                    # Get qubits
                    qubits = [q.index for q in inst.qubits]
                    control1, control2, target = qubits
                    
                    # Use an approximate implementation
                    self._add_approximate_toffoli(result, control1, control2, target)
                    approximated += 1
                else:
                    # Keep the original gate
                    result.append(inst.operation, inst.qubits, inst.clbits if hasattr(inst, 'clbits') else [])
            
            if self.debug_mode:
                print(f"Approximated {approximated} of {toffoli_count} Toffoli gates")
                print(f"Depth before: {circuit.depth()}, after: {result.depth()}")
            
            return result
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error approximating Toffoli gates: {e}")
                traceback.print_exc()
            
            # Return original circuit if approximation fails
            return circuit
            
    def _add_approximate_toffoli(self, circuit, control1, control2, target):
        """
        Add an approximate Toffoli gate implementation to a circuit.
        
        Args:
            circuit: Circuit to add the gate to
            control1: First control qubit
            control2: Second control qubit
            target: Target qubit
        """
        # Implement a lower-depth, lower-fidelity Toffoli gate
        circuit.h(target)
        circuit.cx(control1, target)
        circuit.tdg(target)
        circuit.cx(control2, target)
        circuit.t(target)
        circuit.cx(control1, target)
        circuit.tdg(target)
        circuit.cx(control2, target)
        circuit.t(target)
        circuit.h(target)
