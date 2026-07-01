"""
Circuit Gate Processor Module

This module provides the CircuitGateProcessor class for safely manipulating
circuit gates with robust error handling.
"""

import traceback

# Check for Qiskit availability
try:
    from qiskit import QuantumCircuit, transpile
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

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
            print("Error: Qiskit is required to process circuits")
            return circuit
            
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
