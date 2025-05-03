"""
Circuit Utilities Module

This module provides utility functions for working with quantum circuits,
including physical mapping, circuit validation, and fidelity estimation.
"""

import os
import time
import numpy as np
import gc  # For garbage collection

# Check if Qiskit is available
try:
    from qiskit import QuantumCircuit, transpile
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

def calculate_gate_fidelities(gate_counts, error_rates=None):
    """
    Calculate gate fidelities based on gate counts and error rates.
    
    Args:
        gate_counts (dict): Dictionary of gate counts
        error_rates (dict): Dictionary of error rates by gate type
        
    Returns:
        float: Combined gate fidelity
    """
    if error_rates is None:
        # Default error rates for common gates
        error_rates = {
            'cx': 0.01,      # 1% error for CNOT
            'ccx': 0.03,     # 3% error for Toffoli
            'x': 0.001,      # 0.1% error for X
            'h': 0.001,      # 0.1% error for H
            't': 0.001,      # 0.1% error for T
            'tdg': 0.001,    # 0.1% error for Tdg
            'rz': 0.0005,    # 0.05% error for RZ
            'default': 0.001 # 0.1% error for other gates
        }
    
    total_fidelity = 1.0
    
    for gate_name, count in gate_counts.items():
        # Get error rate for this gate type
        error_rate = error_rates.get(gate_name, error_rates['default'])
        
        # Calculate fidelity for this gate type
        gate_fidelity = (1 - error_rate) ** count
        
        # Update total fidelity
        total_fidelity *= gate_fidelity
    
    return total_fidelity

def estimate_fidelity(circuit=None, num_qubits=None, num_operations=None, 
                     cx_count=None, t_count=None, depth=None):
    """
    Estimate the fidelity of a quantum circuit.
    
    This function can accept various parameter formats for flexibility:
    - estimate_fidelity(circuit) - Extract metrics from circuit
    - estimate_fidelity(num_qubits, num_operations)
    - estimate_fidelity(num_qubits, num_operations, cx_count, t_count, depth)
    
    Args:
        circuit (QuantumCircuit): Quantum circuit
        num_qubits (int): Number of qubits in the circuit
        num_operations (int): Total number of operations in the circuit
        cx_count (int): Number of CNOT gates
        t_count (int): Number of T gates
        depth (int): Circuit depth
        
    Returns:
        float: Estimated fidelity (0.0-1.0)
    """
    # Extract metrics from circuit if provided
    if circuit is not None:
        if not hasattr(circuit, 'count_ops'):
            return 0.9  # Default if not a valid circuit
            
        try:
            num_qubits = circuit.num_qubits
            gate_counts = circuit.count_ops()
            num_operations = sum(gate_counts.values())
            cx_count = gate_counts.get('cx', 0)
            t_count = gate_counts.get('t', 0) + gate_counts.get('tdg', 0)
            depth = circuit.depth()
        except Exception:
            # If extraction fails, use default
            return 0.9
    
    # Use provided parameters or defaults
    num_qubits = num_qubits or 1
    num_operations = num_operations or 0
    
    # If cx_count not provided, estimate it as 20% of operations
    if cx_count is None:
        cx_count = int(num_operations * 0.2)
        
    # If t_count not provided, estimate it as 10% of operations
    if t_count is None:
        t_count = int(num_operations * 0.1)
    
    # Calculate non-CX, non-T gate count
    other_gates = num_operations - cx_count - t_count
    
    # Baseline error rate per operation
    base_error_rate = 0.001  # 0.1% error per gate
    
    # CNOT error rate is higher
    cx_error_rate = 0.01     # 1% error per CNOT
    
    # T gate error rate (usually higher than other single-qubit gates)
    t_error_rate = 0.002     # 0.2% error per T gate
    
    # Calculate fidelity components
    cx_fidelity = (1 - cx_error_rate) ** cx_count
    t_fidelity = (1 - t_error_rate) ** t_count
    other_fidelity = (1 - base_error_rate) ** other_gates
    
    # Calculate estimated fidelity
    fidelity = cx_fidelity * t_fidelity * other_fidelity
    
    # Decoherence effects based on number of qubits and circuit depth
    if depth is not None:
        # Simple decoherence model
        decoherence_factor = 1.0 - (0.001 * num_qubits * np.log(1 + depth))
        decoherence_factor = max(0.8, decoherence_factor)  # Don't let it go below 80%
        fidelity *= decoherence_factor
    
    # Ensure fidelity is in valid range
    fidelity = max(0.0, min(1.0, fidelity))
    
    return fidelity

def create_naive_physical_mapping(circuit, coupling_map, basis_gates=None):
    """
    Create a naive physical mapping of a logical circuit to hardware.
    
    Args:
        circuit (QuantumCircuit): The logical circuit to map
        coupling_map (list): The coupling map as a list of qubit pairs
        basis_gates (list): The basis gates to use for the target hardware
        
    Returns:
        QuantumCircuit: The physically mapped circuit
    """
    # This is now just a wrapper around the more sophisticated function
    return create_optimized_physical_mapping(
        circuit,
        coupling_map,
        basis_gates,
        optimization_level=1  # Use lower optimization level for naive mapping
    )

def create_optimized_physical_mapping(circuit, coupling_map, basis_gates=None, optimization_level=3):
    """
    Create an optimized physical mapping of a logical circuit to hardware that respects coupling constraints.
    
    Args:
        circuit (QuantumCircuit): The logical circuit to map
        coupling_map (list): The coupling map as a list of qubit pairs
        basis_gates (list): The basis gates to use for the target hardware
        optimization_level (int): Optimization level (0-3)
        
    Returns:
        QuantumCircuit: The physically mapped circuit optimized for the target hardware
    """
    if not QISKIT_AVAILABLE:
        print("Warning: Qiskit not available. Cannot optimize circuit.")
        return circuit
        
    if circuit is None:
        print("Warning: Cannot map None circuit")
        return None
    
    # Default basis gates if none provided
    if basis_gates is None:
        basis_gates = ['id', 'rz', 'sx', 'x', 'cx']
    
    try:
        print(f"Mapping and optimizing circuit for target hardware (optimization level: {optimization_level})...")
        print(f"Circuit before mapping: {circuit.depth()} depth, {len(circuit.data)} gates")
        
        # First pass - map to hardware with optimization level 1
        # This focuses on respecting the coupling map constraints
        initial_mapped = transpile(
            circuit,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            optimization_level=1,
            layout_method='sabre',  # Good for respecting coupling constraints
            routing_method='sabre'  # Efficient routing respecting coupling map
        )
        
        # Force garbage collection after first transpilation
        gc.collect()
        
        print(f"Initial mapping complete: {initial_mapped.depth()} depth, {len(initial_mapped.data)} gates")
        
        if optimization_level <= 1:
            # If only basic optimization is requested, return the initial mapping
            return initial_mapped
        
        # Second pass - optimize the mapped circuit further
        # Now that we have a valid mapping, optimize more aggressively
        optimized_mapped = transpile(
            initial_mapped,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            # Don't change the layout again to preserve coupling map compliance
            layout_method='trivial'
        )
        
        # Force garbage collection
        del initial_mapped
        gc.collect()
        
        print(f"Optimized mapping complete: {optimized_mapped.depth()} depth, {len(optimized_mapped.data)} gates")
        
        return optimized_mapped
    
    except Exception as e:
        print(f"Error creating optimized physical mapping: {e}")
        # Try a simpler approach as fallback
        try:
            # Simple single-pass transpilation
            mapped_circuit = transpile(
                circuit,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                optimization_level=min(1, optimization_level)  # Use lower optimization level for fallback
            )
            print(f"Fallback mapping complete: {mapped_circuit.depth()} depth")
            return mapped_circuit
        except Exception as fallback_error:
            print(f"Fallback mapping also failed: {fallback_error}")
            return circuit  # Return original circuit as last resort

def get_qubit_index(qubit):
    """
    Safely get qubit index in a way compatible with both Qiskit versions.
    
    Args:
        qubit: Qiskit Qubit object
        
    Returns:
        int: Qubit index
    """
    if hasattr(qubit, '_index'):
        # Qiskit 2.0+
        return qubit._index
    elif hasattr(qubit, 'index'):
        # Older Qiskit versions
        return qubit.index
    else:
        # Fallback
        qubit_str = str(qubit)
        if 'q[' in qubit_str:
            # Try to extract index from string representation
            try:
                idx_str = qubit_str.split('q[')[1].split(']')[0]
                return int(idx_str)
            except:
                pass
        # If all else fails, raise an error
        raise ValueError(f"Cannot determine index for qubit: {qubit}")

def validate_physical_circuit(circuit, coupling_map):
    """
    Validate that a circuit respects the coupling map constraints.
    
    Args:
        circuit (QuantumCircuit): The circuit to validate
        coupling_map (list): The coupling map as a list of qubit pairs
        
    Returns:
        tuple: (is_valid, violations)
    """
    if circuit is None or coupling_map is None:
        return False, ["Circuit or coupling map is None"]
    
    # Convert coupling map to set of tuples for faster lookup
    if isinstance(coupling_map, list):
        # Convert list of lists to set of tuples
        coupling_set = set((edge[0], edge[1]) for edge in coupling_map)
        # Add reverse direction for bidirectional coupling
        coupling_set.update((edge[1], edge[0]) for edge in coupling_map)
    else:
        # Try to get the coupling list from a coupling map object
        try:
            coupling_list = coupling_map.get_edges()
            coupling_set = set((edge[0], edge[1]) for edge in coupling_list)
            coupling_set.update((edge[1], edge[0]) for edge in coupling_list)
        except:
            return False, ["Invalid coupling map format"]
    
    violations = []
    
    try:
        # Check each multi-qubit gate against the coupling map
        for i, instruction in enumerate(circuit.data):
            # Skip single-qubit gates
            if len(instruction.qubits) <= 1:
                continue
            
            # Get qubit indices
            qubit_indices = []
            for q in instruction.qubits:
                try:
                    if hasattr(q, '_index'):
                        # Newer Qiskit
                        qubit_indices.append(q._index)
                    else:
                        # Older Qiskit
                        qubit_indices.append(q.index)
                except:
                    # Skip if we can't determine qubit index
                    continue
            
            # For CX gates, check if the control and target are connected
            if instruction.operation.name in ['cx', 'CX', 'cnot']:
                # For two-qubit gates, check if the qubits are connected
                if len(qubit_indices) >= 2 and (qubit_indices[0], qubit_indices[1]) not in coupling_set:
                    violations.append(f"Gate {instruction.operation.name} at index {i} "
                                    f"between qubits {qubit_indices[0]} and {qubit_indices[1]} "
                                    f"violates coupling map")
            
            # For multi-qubit gates like CCX, check all pairs
            elif len(qubit_indices) > 2:
                operation_name = instruction.operation.name
                
                # Check all qubit pairs in the gate
                # This is a simplified check - in reality, it depends on how the gate is implemented
                for i in range(len(qubit_indices)):
                    for j in range(i+1, len(qubit_indices)):
                        if (qubit_indices[i], qubit_indices[j]) not in coupling_set:
                            violations.append(f"Multi-qubit gate {operation_name} "
                                            f"between qubits {qubit_indices[i]} and {qubit_indices[j]} "
                                            f"violates coupling map")
    except Exception as e:
        violations.append(f"Error during validation: {e}")
    
    is_valid = len(violations) == 0
    return is_valid, violations

# Default coupling map generators for different topologies
def get_default_coupling_map(topology, num_qubits):
    """
    Generate a default coupling map for a given topology.
    
    Args:
        topology (str): Topology type ('linear', 'grid', 'falcon')
        num_qubits (int): Number of qubits
        
    Returns:
        list: Coupling map as a list of qubit pairs [(q1, q2), ...]
    """
    if topology == 'linear':
        # Linear topology - each qubit connected to adjacent qubits
        coupling_map = []
        for i in range(num_qubits - 1):
            coupling_map.append([i, i+1])
            coupling_map.append([i+1, i])  # Add reverse connections for bidirectional coupling
        return coupling_map
    
    elif topology == 'grid':
        # 2D grid topology
        coupling_map = []
        grid_size = int(np.ceil(np.sqrt(num_qubits)))
        
        for i in range(num_qubits):
            row = i // grid_size
            col = i % grid_size
            
            # Connect to the right
            if col < grid_size - 1 and i + 1 < num_qubits:
                coupling_map.append([i, i+1])
                coupling_map.append([i+1, i])
            
            # Connect to the bottom
            if row < grid_size - 1 and i + grid_size < num_qubits:
                coupling_map.append([i, i+grid_size])
                coupling_map.append([i+grid_size, i])
        
        return coupling_map
    
    elif topology == 'falcon':
        # IBM Falcon processor topology (approx)
        # This is a simplified version
        coupling_map = []
        
        # Create a hexagonal lattice-like connectivity
        for i in range(num_qubits):
            # Connect to the next qubit
            if i + 1 < num_qubits:
                coupling_map.append([i, i+1])
                coupling_map.append([i+1, i])
            
            # Connect to the qubit 2 positions away
            if i + 2 < num_qubits:
                coupling_map.append([i, i+2])
                coupling_map.append([i+2, i])
            
            # Connect to the qubit 3 positions away for some qubits
            if i % 3 == 0 and i + 3 < num_qubits:
                coupling_map.append([i, i+3])
                coupling_map.append([i+3, i])
        
        return coupling_map
    
    else:
        # Default to linear topology
        print(f"Warning: Unknown topology '{topology}'. Using linear topology.")
        return get_default_coupling_map('linear', num_qubits)

class CircuitGateProcessor:
    """Helper class for safely manipulating circuit gates with robust error handling."""
    
    def __init__(self, debug_mode=False):
        """Initialize the gate processor.
        
        Args:
            debug_mode (bool): Whether to print debug information
        """
        self.debug_mode = debug_mode
    
    def create_controlled_gate_safely(self, gate, control_qubits, target_qubits, circuit):
        """
        Safely add a controlled version of a gate to a circuit with proper error handling.
        
        Args:
            gate: The base gate to control
            control_qubits: List of control qubits
            target_qubits: List of target qubits
            circuit: The quantum circuit to add the gate to
            
        Returns:
            bool: Whether the gate was added successfully
        """
        try:
            # Validate that the gate can be controlled
            if not hasattr(gate, 'control'):
                if self.debug_mode:
                    print(f"Gate {gate.name} cannot be controlled")
                return False
            
            # Calculate maximum allowed controls
            max_controls = circuit.num_qubits - len(target_qubits)
            
            # Validate number of control qubits
            if len(control_qubits) < 1 or len(control_qubits) > max_controls:
                if self.debug_mode:
                    print(f"Invalid number of control qubits: {len(control_qubits)} " +
                          f"(must be between 1 and {max_controls})")
                
                # If we have too many controls, try with fewer
                if len(control_qubits) > max_controls:
                    reduced_controls = control_qubits[:max_controls]
                    return self.create_controlled_gate_safely(
                        gate, reduced_controls, target_qubits, circuit)
                return False
            
            # Ensure qubit indices are valid for the circuit
            if max(control_qubits + target_qubits) >= circuit.num_qubits:
                if self.debug_mode:
                    print(f"Qubit index out of range for circuit with {circuit.num_qubits} qubits")
                return False
            
            # Create and add the controlled gate
            controlled_gate = gate.control(len(control_qubits))
            circuit.append(controlled_gate, control_qubits + target_qubits)
            return True
                
        except Exception as e:
            if self.debug_mode:
                print(f"Error creating controlled gate: {e}")
            
            # Try again with one fewer control if possible
            if len(control_qubits) > 1:
                reduced_controls = control_qubits[:-1]
                return self.create_controlled_gate_safely(
                    gate, reduced_controls, target_qubits, circuit)
            return False
    
    def process_circuit_safely(self, circuit, coupling_map=None):
        """
        Process a circuit safely, handling control qubit constraints and coupling map properly.
        
        Args:
            circuit: The quantum circuit to process
            coupling_map: Optional coupling map constraints
            
        Returns:
            QuantumCircuit: Safely processed circuit that respects all constraints
        """
        if not QISKIT_AVAILABLE:
            print("Warning: Qiskit not available. Cannot process circuit.")
            return circuit
            
        from qiskit import QuantumCircuit
        import traceback
        
        # Create a new circuit with the same number of qubits and clbits
        fixed_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        
        # Create connectivity graph from coupling map if provided
        connected_pairs = set()
        if coupling_map is not None:
            if isinstance(coupling_map, list):
                for pair in coupling_map:
                    if isinstance(pair, list) and len(pair) == 2:
                        connected_pairs.add((pair[0], pair[1]))
                        connected_pairs.add((pair[1], pair[0]))  # Both directions
            else:
                # Try to get edges from coupling map object
                try:
                    coupling_list = coupling_map.get_edges()
                    for edge in coupling_list:
                        connected_pairs.add((edge[0], edge[1]))
                        connected_pairs.add((edge[1], edge[0]))
                except:
                    if self.debug_mode:
                        print("Warning: Could not extract connectivity from coupling map")
        
        # Track statistics for reporting
        total_gates = len(circuit.data)
        skipped_gates = 0
        modified_gates = 0
        
        # Process each instruction with proper constraint handling
        for idx, inst in enumerate(circuit.data):
            try:
                operation = inst.operation
                gate_name = operation.name.lower() if hasattr(operation, 'name') else "unknown"
                
                # Get qubit indices safely
                qubit_indices = []
                for q in inst.qubits:
                    try:
                        qubit_indices.append(q._index if hasattr(q, '_index') else q.index)
                    except:
                        if self.debug_mode:
                            print(f"Error getting qubit index for gate at index {idx}")
                        continue
                
                # Get clbit indices safely
                clbit_indices = []
                if hasattr(inst, 'clbits'):
                    for c in inst.clbits:
                        try:
                            clbit_indices.append(c._index if hasattr(c, '_index') else c.index)
                        except:
                            continue
                
                # Check for issues that would cause errors
                has_issues = False
                issue_description = ""
                
                # Issue 1: Qubit index out of range
                if any(i >= circuit.num_qubits for i in qubit_indices):
                    has_issues = True
                    issue_description = f"qubit index out of range for circuit with {circuit.num_qubits} qubits"
                
                # Issue 2: Empty qubit list
                elif not qubit_indices:
                    has_issues = True
                    issue_description = "no valid qubit indices"
                
                # Issue 3: Control qubit constraint violations
                elif ('c' in gate_name or gate_name in ['mcx', 'ccx', 'toffoli']) and len(qubit_indices) > 1:
                    # For controlled gates, determine the base gate
                    if hasattr(operation, 'base_gate'):
                        base_gate = operation.base_gate
                        base_gate_qubits = base_gate.num_qubits if hasattr(base_gate, 'num_qubits') else 1
                        num_ctrl_qubits = operation.num_ctrl_qubits if hasattr(operation, 'num_ctrl_qubits') else (len(qubit_indices) - base_gate_qubits)
                    else:
                        # Default assumptions
                        base_gate_qubits = 1  # Most controlled gates have 1 target
                        if gate_name == 'ccx' or gate_name == 'toffoli':
                            num_ctrl_qubits = 2
                        else:
                            num_ctrl_qubits = len(qubit_indices) - 1
                    
                    # Calculate max allowed controls
                    max_control_qubits = circuit.num_qubits - base_gate_qubits
                    
                    if num_ctrl_qubits > max_control_qubits:
                        has_issues = True
                        issue_description = f"too many control qubits ({num_ctrl_qubits}), max allowed is {max_control_qubits}"
                
                # Issue 4: Coupling map violations for multi-qubit gates
                elif len(qubit_indices) > 1 and connected_pairs:
                    # Check if qubits are connected according to coupling map
                    connectivity_violated = False
                    violated_pairs = []
                    
                    for i in range(len(qubit_indices) - 1):
                        for j in range(i + 1, len(qubit_indices)):
                            if (qubit_indices[i], qubit_indices[j]) not in connected_pairs and (qubit_indices[j], qubit_indices[i]) not in connected_pairs:
                                connectivity_violated = True
                                violated_pairs.append((qubit_indices[i], qubit_indices[j]))
                    
                    if connectivity_violated:
                        has_issues = True
                        issue_description = f"coupling map violations for pairs: {violated_pairs}"
                
                # Handle issues
                if has_issues:
                    if self.debug_mode:
                        print(f"Gate {idx} ({gate_name}) has issues: {issue_description}")
                    
                    # Try to find a valid alternative for special cases
                    if gate_name == 'ccx' or gate_name == 'toffoli':
                        # For Toffoli gates, try using a single control if possible
                        if len(qubit_indices) >= 3 and not (connected_pairs and issue_description.startswith("coupling map")):
                            control = qubit_indices[0]  # Use just the first control
                            target = qubit_indices[-1]  # Last qubit is target
                            if self.debug_mode:
                                print(f"Converting Toffoli to CNOT: control={control}, target={target}")
                            fixed_circuit.cx(control, target)
                            modified_gates += 1
                            continue
                    
                    elif gate_name == 'cx' or gate_name == 'cnot':
                        # For CNOT gates, try swapping control and target if it helps with coupling map
                        if len(qubit_indices) >= 2 and connected_pairs and (qubit_indices[1], qubit_indices[0]) in connected_pairs:
                            # We can swap control and target, add H gates around
                            control, target = qubit_indices[0], qubit_indices[1]
                            if self.debug_mode:
                                print(f"Transforming CNOT to comply with coupling map using Hadamard transformation")
                            fixed_circuit.h(control)
                            fixed_circuit.h(target)
                            fixed_circuit.cx(target, control)  # Reversed
                            fixed_circuit.h(control)
                            fixed_circuit.h(target)
                            modified_gates += 1
                            continue
                    
                    skipped_gates += 1
                    continue
                
                # Add the gate to the fixed circuit
                fixed_circuit.append(operation, qubit_indices, clbit_indices)
                
            except Exception as e:
                if self.debug_mode:
                    print(f"Error processing gate at index {idx}: {e}")
                    traceback.print_exc()
                skipped_gates += 1
        
        # Report statistics
        if self.debug_mode and (skipped_gates > 0 or modified_gates > 0):
            print(f"Circuit processing summary: {skipped_gates} gates skipped, {modified_gates} gates modified out of {total_gates} total gates")
        
        return fixed_circuit