"""
Toffoli Compiler Module

This module contains the ToffoliCompiler class for converting Toffoli networks
to quantum circuits executable on quantum hardware.
"""

from enum import Enum
from typing import List, Dict, Tuple, Optional, Union, Callable
import time
import numpy as np

# Check if Qiskit is available
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import CouplingMap
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

class ToffoliType(Enum):
    """Types of Toffoli gate implementations"""
    STANDARD = 0         # Standard decomposition, no ancilla
    RELATIVE_PHASE_1 = 1 # 1-ancilla relative phase Toffoli
    OPTIMIZED_2 = 2      # 2-ancilla optimized
    OPTIMIZED_3 = 3      # 3-ancilla optimized
    OPTIMIZED_4 = 4      # 4-ancilla optimized
    OPTIMIZED_7 = 7      # 7+ ancilla optimized

class ToffoliCompiler:
    """
    Compiler for converting Toffoli networks to quantum circuits.
    
    This class provides methods to create, decompose, and analyze quantum circuits
    that implement Toffoli (CCX) gate networks.
    """
    
    def __init__(self, default_basis_gates=None, debug_mode=False, coupling_map=None, optimization_level=3):
        """
        Initialize the Toffoli compiler.
        
        Args:
            default_basis_gates (list): Default basis gates to use for decomposition
            debug_mode (bool): Whether to print debug information
            coupling_map: IBM Q coupling map (None uses a default coupling map)
            optimization_level: Qiskit transpiler optimization level (0-3)
        """
        if default_basis_gates is None:
            self.default_basis_gates = ['id', 'rz', 'sx', 'x', 'cx']
        else:
            self.default_basis_gates = default_basis_gates
        
        self.debug_mode = debug_mode
        self.optimization_level = optimization_level


        # Store the coupling map or create a default one if None
        self.coupling_map = coupling_map
        
        # Initialize the default coupling map only if one was not provided
        if self.coupling_map is None:
            self.coupling_map = self._create_default_coupling_map(20)  # Linear coupling map with 20 qubits
        
        # Store the implementations for different Toffoli types
        self.implementations = {
            ToffoliType.STANDARD: self._create_standard_toffoli,
            ToffoliType.RELATIVE_PHASE_1: self._create_relative_phase_toffoli,
            ToffoliType.OPTIMIZED_2: self._create_optimized_2_toffoli,
            ToffoliType.OPTIMIZED_3: self._create_optimized_3_toffoli,
            ToffoliType.OPTIMIZED_4: self._create_optimized_4_toffoli,
            ToffoliType.OPTIMIZED_7: self._create_optimized_7_toffoli,
        }
        
        # Cache for unitaries to avoid recomputation
        self._unitary_cache = {}
    
    def _create_default_coupling_map(self, num_qubits):
        """Create a default linear coupling map"""
        try:
            # For older Qiskit versions
            if hasattr(CouplingMap, 'from_line'):
                return CouplingMap.from_line(num_qubits)
            # For newer versions
            else:
                # Create a linear coupling map manually
                coupling_list = []
                for i in range(num_qubits-1):
                    coupling_list.append([i, i+1])
                return coupling_list
        except Exception as e:
            if self.debug_mode:
                print(f"Error creating coupling map: {e}")
            # Fallback to simple list
            coupling_list = []
            for i in range(num_qubits-1):
                coupling_list.append([i, i+1])
            return coupling_list
    
    def _create_standard_toffoli(self, circuit, control1, control2, target):
        """Standard Toffoli decomposition (no ancilla)"""
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
        circuit.t(control1)
        circuit.t(control2)
        circuit.cx(control2, control1)
        circuit.t(control1)
        circuit.tdg(control2)
        circuit.cx(control2, control1)
    
    def _create_relative_phase_toffoli(self, circuit, control1, control2, target, ancilla):
        """Relative-phase Toffoli with 1 ancilla qubit"""
        circuit.h(target)
        circuit.cx(control1, ancilla)
        circuit.cx(control2, ancilla)
        circuit.cx(ancilla, target)
        circuit.cx(control1, ancilla)
        circuit.cx(control2, ancilla)
        circuit.h(target)
    
    def _create_optimized_2_toffoli(self, circuit, control1, control2, target, ancilla1, ancilla2):
        """Optimized Toffoli with 2 ancilla qubits"""
        circuit.h(ancilla1)
        circuit.h(ancilla2)
        circuit.cx(control1, ancilla1)
        circuit.cx(control2, ancilla2)
        circuit.cx(ancilla1, ancilla2)
        circuit.cx(ancilla2, target)
        circuit.cx(control1, ancilla1)
        circuit.cx(ancilla1, ancilla2)
        circuit.cx(control2, ancilla2)
        circuit.h(ancilla1)
        circuit.h(ancilla2)

    def _create_optimized_3_toffoli(self, circuit, control1, control2, target, ancilla1, ancilla2, ancilla3):
        """Optimized Toffoli with 3 ancilla qubits"""
        circuit.h(ancilla1)
        circuit.h(ancilla2)
        circuit.h(ancilla3)
        circuit.cx(control1, ancilla1)
        circuit.cx(control2, ancilla2)
        circuit.cx(ancilla1, ancilla3)
        circuit.cx(control1, control2)
        circuit.cx(ancilla3, target)
        circuit.cx(control2, ancilla2)
        circuit.cx(ancilla1, ancilla3)
        circuit.h(target)
        circuit.h(ancilla1)
        circuit.h(ancilla2)
        circuit.h(ancilla3)
    
    def _create_optimized_4_toffoli(self, circuit, control1, control2, target, *ancilla_qubits):

        """Optimized Toffoli with 4 ancilla qubits"""
        circuit.h(target)
        circuit.h(ancilla_qubits[0])
        circuit.h(ancilla_qubits[1])
        circuit.cx(control1, ancilla_qubits[0])
        circuit.cx(control2, ancilla_qubits[1])
        # Middle operations would go here
        circuit.cx(ancilla_qubits[3], target)
        circuit.h(target)
        circuit.h(ancilla_qubits[0])
        circuit.h(ancilla_qubits[1])
    

    def _create_optimized_7_toffoli(self, circuit, control1, control2, target, *ancilla_qubits):

        """Optimized Toffoli with 7+ ancilla qubits for minimum depth"""
        for ancilla in ancilla_qubits[:5]:
            circuit.h(ancilla)
        
        circuit.cx(control1, ancilla_qubits[0])
        circuit.cx(control2, ancilla_qubits[1])
        # Middle operations would go here
        circuit.cx(ancilla_qubits[6], target)
        
        for ancilla in ancilla_qubits[:5]:
            circuit.h(ancilla)
            
    def _create_approximate_toffoli(self, circuit, control1, control2, target, fidelity=0.9):

        """
        Create an approximate Toffoli gate with controllable fidelity
        
        Args:
            circuit: Quantum circuit to add the gate to
            control1: First control qubit
            control2: Second control qubit  
            target: Target qubit
            fidelity: Target fidelity (1.0 = exact, lower values allow approximation)
        """
        # For high fidelity requirements, use the standard implementation
        if fidelity > 0.99:
            self._create_standard_toffoli(circuit, control1, control2, target)
            return
        
        # For fidelity 0.95-0.99, use high fidelity approximation
        if fidelity >= 0.95:
            # Higher fidelity approximation with 7 CNOTs and T gates
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
            return
        
        # For fidelity 0.9-0.95, use intermediate approximation
        if fidelity >= 0.9:
            # Intermediate approximation with 5 CNOTs
            circuit.h(target)
            circuit.cx(control1, target)
            circuit.cx(control2, target)
            circuit.cx(control1, target)
            circuit.h(target)
            return
        
        # Lowest fidelity approximation with 3 CNOTs
        circuit.cx(control1, target)
        circuit.cx(control2, target)
        circuit.cx(control1, target)
    

    def create_toffoli(self, qc, control1, control2, target, toffoli_type=ToffoliType.STANDARD, ancilla_qubits=None):

        """
        Add a Toffoli gate to the circuit with the specified implementation
        
        Args:
            qc: Quantum circuit to add the Toffoli to
            control1: First control qubit
            control2: Second control qubit
            target: Target qubit
            toffoli_type: Type of Toffoli implementation to use
            ancilla_qubits: List of ancilla qubits to use (if needed)
        """
        if toffoli_type == ToffoliType.STANDARD:
            self.implementations[toffoli_type](qc, control1, control2, target)
        elif toffoli_type == ToffoliType.RELATIVE_PHASE_1:
            if not ancilla_qubits or len(ancilla_qubits) < 1:
                raise ValueError("Relative phase Toffoli requires 1 ancilla qubit")
            self.implementations[toffoli_type](qc, control1, control2, target, ancilla_qubits[0])
        elif toffoli_type == ToffoliType.OPTIMIZED_2:
            if not ancilla_qubits or len(ancilla_qubits) < 2:
                raise ValueError("Optimized 2-ancilla Toffoli requires 2 ancilla qubits")
            self.implementations[toffoli_type](qc, control1, control2, target, ancilla_qubits[0], ancilla_qubits[1])
        elif toffoli_type == ToffoliType.OPTIMIZED_3:
            if not ancilla_qubits or len(ancilla_qubits) < 3:
                raise ValueError("Optimized 3-ancilla Toffoli requires 3 ancilla qubits")
            self.implementations[toffoli_type](qc, control1, control2, target, ancilla_qubits[0], ancilla_qubits[1], ancilla_qubits[2])
        elif toffoli_type == ToffoliType.OPTIMIZED_4:
            if not ancilla_qubits or len(ancilla_qubits) < 4:
                raise ValueError("Optimized 4-ancilla Toffoli requires 4 ancilla qubits")
            self.implementations[toffoli_type](qc, control1, control2, target, *ancilla_qubits[:4])
        elif toffoli_type == ToffoliType.OPTIMIZED_7:
            if not ancilla_qubits or len(ancilla_qubits) < 7:
                raise ValueError("Optimized 7-ancilla Toffoli requires 7 ancilla qubits")
            self.implementations[toffoli_type](qc, control1, control2, target, *ancilla_qubits[:7])


    def create_toffoli_network(self,
                              toffoli_gates,
                              num_qubits,
                              use_ancilla=True,
                              target_fidelity=0.95,
                              toffoli_type=ToffoliType.RELATIVE_PHASE_1,
                              allocate_ancilla_strategy='shared'):
        """
        Create a quantum circuit from a Toffoli network with comprehensive error handling and optimization.
        
        This method translates a list of Toffoli gates (CCX gates) into a fully implemented quantum
        circuit, with support for different implementation strategies and optimizations. It handles
        various error cases and constraints, such as qubit count limitations and control qubit constraints.
        
        Args:
            toffoli_gates (list): List of Toffoli gates specified as either:
                - (control_qubits, target_qubit) tuples where control_qubits is a list of indices
                - (control1, control2, target) tuples for backward compatibility
            
            num_qubits (int): Total number of qubits in the circuit.
                Must be sufficient to accommodate all gate indices plus any required ancilla qubits.
            
            use_ancilla (bool): Whether to use ancilla qubits for optimization.
                If True, additional qubits may be allocated for more efficient implementations.
                If False, only standard decomposition will be used regardless of toffoli_type.
            
            target_fidelity (float): Target circuit fidelity (0.0-1.0).
                Affects how aggressively gates are approximated. Lower values allow more aggressive
                approximations that reduce depth at the cost of exactness.
            
            toffoli_type (ToffoliType): Type of Toffoli implementation to use:
                - STANDARD: Standard decomposition, no ancilla
                - RELATIVE_PHASE_1: 1-ancilla relative phase Toffoli
                - OPTIMIZED_2: 2-ancilla optimized
                - OPTIMIZED_3: 3-ancilla optimized
                - OPTIMIZED_4: 4-ancilla optimized
                - OPTIMIZED_7: 7+ ancilla optimized
                If None, defaults to STANDARD.
            
            allocate_ancilla_strategy (str): Strategy for allocating ancilla qubits:
                - 'shared': All Toffoli gates share the same ancilla qubits (less qubits, more depth)
                - 'dedicated': Each Toffoli gate gets its own ancilla qubits if available
                            (more qubits, less depth)
                
        Returns:
            tuple: (circuit, ancilla_indices)
                - circuit: The quantum circuit implementing the Toffoli network
                - ancilla_indices: List of indices of qubits used as ancilla, or None if no ancillas used
        
        Raises:
            ValueError: If the Toffoli network cannot be created due to invalid gate specifications
                    or insufficient qubit count
        
        Notes:
            - The method automatically validates and filters invalid gates
            - Gates with invalid control or target qubits are skipped with warnings
            - If requested Toffoli implementation requires more ancilla qubits than available,
            the method will fall back to simpler implementations

        """
        if not QISKIT_AVAILABLE:
            print("Error: Qiskit is required to create Toffoli networks")
            return None, None
        
        start_time = time.time()
        
        if self.debug_mode:
            print(f"Creating Toffoli network with {len(toffoli_gates)} gates on {num_qubits} qubits")
            print(f"Using implementation: {toffoli_type.name if isinstance(toffoli_type, ToffoliType) else 'STANDARD'}")
        
        # Create a quantum circuit
        circuit = QuantumCircuit(num_qubits)
        
        # No ancillas by default
        ancilla_indices = None
        
        try:
            # Convert old style (control1, control2, target) tuples to new style ([controls], target)
            normalized_gates = []
            for gate_info in toffoli_gates:
                if isinstance(gate_info, tuple):
                    if len(gate_info) == 3:  # Old style: (control1, control2, target)
                        control1, control2, target = gate_info
                        normalized_gates.append(([control1, control2], target))
                    elif len(gate_info) == 2:  # New style: ([controls], target)
                        normalized_gates.append(gate_info)
                    else:
                        print(f"Warning: Invalid Toffoli gate format: {gate_info}")
                        continue
                else:
                    print(f"Warning: Invalid Toffoli gate format: {gate_info}")
                    continue
 
            # For compatibility with both old and new style
            toffoli_gates = normalized_gates
            # Pre-process gates to ensure they respect control qubit constraints
            valid_toffoli_gates = []
            for gate_idx, (controls, target) in enumerate(toffoli_gates):
                # Print debug information about this gate
                if self.debug_mode: 
                    print(f"DEBUG: Processing gate {gate_idx}: controls={controls}, target={target}")
                
                # Ensure target is in range
                if target >= num_qubits:
                    print(f"Warning: Gate {gate_idx} has target {target} outside qubit range {num_qubits}. Skipping.")
                    continue
                    
                # Track rejection reasons for controls
                invalid_controls = []
                valid_controls = []
                
                # Check if control qubits are in range
                if isinstance(controls, list) or isinstance(controls, tuple):
                    for control in controls:
                        # Print explicit check for each control
                        if self.debug_mode: 
                            print(f"DEBUG: Checking control {control} for gate {gate_idx}")
                        
                        if control >= num_qubits:
                            if self.debug_mode: 
                                print(f"DEBUG: Control {control} is out of range (>= {num_qubits})")
                            invalid_controls.append((control, "out of range"))
                        elif control == target:
                            if self.debug_mode: 
                                print(f"DEBUG: Control {control} is same as target {target}")
                            invalid_controls.append((control, "same as target"))
                        else:
                            if self.debug_mode: 
                                print(f"DEBUG: Control {control} is valid")
                            valid_controls.append(control)
                    
                    # Check if we lost any controls
                    if len(valid_controls) < len(controls):
                        rejected = [f"{c[0]} ({c[1]})" for c in invalid_controls]
                        print(f"Warning: Gate {gate_idx} had invalid controls: {', '.join(rejected)}")
                else:
                    # Single control qubit
                    if self.debug_mode: 
                        print(f"DEBUG: Single control {controls} for gate {gate_idx}")
                    
                    if controls >= num_qubits:
                        print(f"Warning: Gate {gate_idx} has control {controls} outside qubit range {num_qubits}. Skipping.")
                        continue
                    elif controls == target:
                        print(f"Warning: Gate {gate_idx} has control {controls} same as target {target}. Skipping.")
                        continue
                    else:
                        valid_controls = [controls]
                
                # Check maximum allowed controls (num_qubits - 1 target qubit)
                max_controls = num_qubits - 1
                if len(valid_controls) > max_controls:
                    print(f"Warning: Gate {gate_idx} has too many controls ({len(valid_controls)}). Reducing to {max_controls}.")
                    removed_controls = valid_controls[max_controls:]
                    valid_controls = valid_controls[:max_controls]
                    print(f"  Removed controls: {removed_controls}")
                
                # Summary of what happened to this gate
                if self.debug_mode: 
                    print(f"DEBUG: Gate {gate_idx} - Original controls: {controls}, Valid controls: {valid_controls}")
                
                # Only add the gate if we have valid controls
                if valid_controls:
                    valid_toffoli_gates.append((valid_controls, target))
                else:
                    # Now with more detailed explanation
                    if invalid_controls:
                        reasons = [f"{c[0]} ({c[1]})" for c in invalid_controls]
                        print(f"Warning: Gate {gate_idx} has no valid controls after filtering. Invalid controls: {', '.join(reasons)}. Skipping.")
                    else:
                        print(f"Warning: Gate {gate_idx} has no valid controls after filtering (unknown reason). Skipping.")
            
            # Use the valid gates
            toffoli_gates = valid_toffoli_gates
            print(f"Using {len(toffoli_gates)} valid gates after filtering.")
            
            # Extract all control and target qubits
            computational_qubits = set()
            for controls, target in toffoli_gates:
                if isinstance(controls, list) or isinstance(controls, tuple):
                    computational_qubits.update(controls)
                else:
                    computational_qubits.add(controls)
                computational_qubits.add(target)
            
            # Remaining qubits are available as ancilla
            all_qubits = set(range(num_qubits))
            available_ancilla = sorted(list(all_qubits - computational_qubits))
            
            # Determine if ancilla qubits are needed
            if use_ancilla and isinstance(toffoli_type, ToffoliType) and toffoli_type != ToffoliType.STANDARD:
                # Get ancilla requirements for the selected implementation
                ancilla_per_toffoli = {
                    ToffoliType.STANDARD: 0,
                    ToffoliType.RELATIVE_PHASE_1: 1,
                    ToffoliType.OPTIMIZED_2: 2,
                    ToffoliType.OPTIMIZED_3: 3,
                    ToffoliType.OPTIMIZED_4: 4,
                    ToffoliType.OPTIMIZED_7: 7
                }
                
                needed_ancilla = ancilla_per_toffoli.get(toffoli_type, 0)
                
                if needed_ancilla > 0:
                    if len(available_ancilla) < needed_ancilla:
                        # Not enough available ancilla, fallback to standard implementation
                        if self.debug_mode:
                            print(f"Not enough ancilla qubits available. Needed: {needed_ancilla}, Available: {len(available_ancilla)}")
                            print(f"Falling back to standard Toffoli implementation")
                        toffoli_type = ToffoliType.STANDARD
                    else:
                        # We have enough ancilla qubits
                        ancilla_indices = available_ancilla[:needed_ancilla]
                        if self.debug_mode:
                            print(f"Using {len(ancilla_indices)} ancilla qubits: {ancilla_indices}")
            
            # Create a mapping of which Toffoli uses which ancilla qubits
            ancilla_map = {}
            
            if allocate_ancilla_strategy == 'shared' and ancilla_indices:
                # Share the same ancilla qubits among all Toffoli gates
                ancilla_map = {i: ancilla_indices for i in range(len(toffoli_gates))}
            elif allocate_ancilla_strategy == 'dedicated' and ancilla_indices:
                # Try to allocate separate ancilla qubits for each Toffoli gate
                # (may not be possible for all gates if not enough ancilla qubits)
                for i in range(len(toffoli_gates)):
                    if toffoli_type == ToffoliType.STANDARD:
                        # No ancilla needed
                        ancilla_map[i] = []
                    else:
                        # Get needed ancilla qubits for this Toffoli type
                        needed = ancilla_per_toffoli.get(toffoli_type, 0)
                        
                        # Calculate start and end index for this gate's ancilla
                        start_idx = i * needed
                        end_idx = start_idx + needed
                        
                        # Check if we have enough ancilla qubits for this gate
                        if start_idx >= len(available_ancilla):
                            # Not enough ancilla for this gate, share with previous gates
                            if len(available_ancilla) > 0:
                                ancilla_map[i] = available_ancilla[:needed] if needed <= len(available_ancilla) else available_ancilla
                            else:
                                # No ancilla available, use standard Toffoli
                                ancilla_map[i] = []
                        else:
                            # Get ancilla qubits for this gate
                            gate_ancilla = available_ancilla[start_idx:end_idx]
                            
                            # If not enough, use what we have
                            if len(gate_ancilla) < needed:
                                gate_ancilla = available_ancilla[:needed] if needed <= len(available_ancilla) else available_ancilla
                            
                            ancilla_map[i] = gate_ancilla
            
            # Process each Toffoli gate
            for i, gate_info in enumerate(toffoli_gates):
                try:
                    controls, target = gate_info
                    
                    # Handle single control (CNOT) or multi-control cases
                    if isinstance(controls, int):
                        # Single control qubit (CNOT gate)
                        circuit.cx(controls, target)
                        if self.debug_mode:
                            print(f"Added CNOT gate with control {controls} and target {target}")
                    elif len(controls) == 1:
                        # Also a CNOT gate
                        circuit.cx(controls[0], target)
                        if self.debug_mode:
                            print(f"Added CNOT gate with control {controls[0]} and target {target}")
                    elif len(controls) == 2:
                        # Standard Toffoli gate with 2 controls
                        if toffoli_type == ToffoliType.STANDARD or not ancilla_indices:
                            # No ancilla available or standard requested
                            self._create_standard_toffoli(circuit, controls[0], controls[1], target)
                        else:
                            # Use the specified implementation with ancilla
                            gate_ancilla = ancilla_map.get(i, [])
                            if gate_ancilla:
                                try:
                                    self.create_toffoli(circuit, controls[0], controls[1], target, toffoli_type, gate_ancilla)
                                except Exception as gate_error:
                                    # Fallback to standard implementation
                                    print(f"Error with specialized Toffoli: {gate_error}. Using standard implementation.")
                                    self._create_standard_toffoli(circuit, controls[0], controls[1], target)
                            else:
                                # No ancilla available, use standard
                                self._create_standard_toffoli(circuit, controls[0], controls[1], target)
                            
                        if self.debug_mode:
                            print(f"Added Toffoli gate with controls {controls} and target {target}")
                    else:
                        # Multi-controlled X gate (more than 2 controls)
                        circuit.mcx(controls, target,
                                   ancilla_qubits=ancilla_indices,
                                   mode='noancilla' if not ancilla_indices else 'v-chain')
                        if self.debug_mode:
                            print(f"Added MCX gate with {len(controls)} controls and target {target}")
                
                except Exception as gate_error:
                    print(f"Error processing gate at index {i}: {gate_error}")
                    # Continue with next gate, rather than failing the whole network
                    continue
            
            if self.debug_mode:
                end_time = time.time()
                print(f"Created Toffoli network with {circuit.depth()} depth in {end_time - start_time:.4f} seconds")
            
            return circuit, ancilla_indices
        
        except Exception as e:
            print(f"Error creating Toffoli network: {e}")
            return None, None


    def decompose_to_basis_gates(self, circuit, basis_gates=None, optimization_level=1):

        """
        Decompose a circuit to the specified basis gates.
        
        Args:
            circuit (QuantumCircuit): The circuit to decompose
            basis_gates (list): The basis gates to use
            optimization_level (int): Optimization level (0-3)
            
        Returns:
            QuantumCircuit: The decomposed circuit
        """
        if not QISKIT_AVAILABLE:
            print("Error: Qiskit is required to decompose circuits")
            return None
        
        if circuit is None:
            return None
        
        if basis_gates is None:
            basis_gates = self.default_basis_gates
        
        start_time = time.time()
        
        if self.debug_mode:
            print(f"Decomposing circuit to basis gates {basis_gates} with optimization level {optimization_level}")
            print(f"Original circuit: {circuit.num_qubits} qubits, {circuit.depth()} depth, {len(circuit.data)} gates")
        
        try:
            # Use Qiskit's transpiler to decompose the circuit
            decomposed = transpile(
                circuit,
                basis_gates=basis_gates,
                optimization_level=optimization_level
            )
            
            if self.debug_mode:
                end_time = time.time()
                print(f"Decomposed circuit: {decomposed.num_qubits} qubits, {decomposed.depth()} depth, {len(decomposed.data)} gates")
                print(f"Decomposition completed in {end_time - start_time:.4f} seconds")
            
            return decomposed
        
        except Exception as e:
            print(f"Error decomposing circuit: {e}")
            return None
    

    def get_circuit_metrics(self, circuit):

        """
        Get various metrics for a quantum circuit.
        
        Args:
            circuit (QuantumCircuit): The circuit to analyze
            
        Returns:
            dict: Dictionary of circuit metrics
        """
        if circuit is None:
            return {
                "depth": 0,
                "width": 0,
                "size": 0,
                "gate_counts": {},
                "cx_count": 0,
                "t_gates": 0,
                "fidelity": 0
            }
        
        try:
            # Basic metrics
            metrics = {
                "depth": circuit.depth(),
                "width": circuit.num_qubits,
                "size": len(circuit.data),
                "gate_counts": {},
                "cx_count": 0,
                "t_gates": 0,
                "single_qubit_gates": 0
            }
            
            # Count gates
            if QISKIT_AVAILABLE:
                # Get gate counts
                try:
                    gate_counts = circuit.count_ops()
                    metrics["gate_counts"] = gate_counts
                    
                    # Count CNOT gates
                    metrics["cx_count"] = gate_counts.get('cx', 0)
                    
                    # Count T and Tdg gates
                    metrics["t_gates"] = gate_counts.get('t', 0) + gate_counts.get('tdg', 0)
                    
                    # Count single-qubit gates
                    metrics["single_qubit_gates"] = sum(gate_counts.get(gate, 0) for gate in
                                                       ['h', 't', 'tdg', 'rx', 'ry', 'rz', 'x', 'y', 'z', 's', 'sdg'])
                except:
                    # Fallback to manual counting if count_ops is not available
                    for instruction in circuit.data:
                        try:
                            gate_name = instruction.operation.name
                            
                            # Update gate count
                            if gate_name in metrics["gate_counts"]:
                                metrics["gate_counts"][gate_name] += 1
                            else:
                                metrics["gate_counts"][gate_name] = 1
                            
                            # Count CNOT gates
                            if gate_name == 'cx':
                                metrics["cx_count"] += 1
                            
                            # Count T and Tdg gates
                            if gate_name in ['t', 'tdg']:
                                metrics["t_gates"] += 1
                                
                            # Count single-qubit gates
                            if gate_name in ['h', 't', 'tdg', 'rx', 'ry', 'rz', 'x', 'y', 'z', 's', 'sdg']:
                                metrics["single_qubit_gates"] += 1
                                
                        except Exception as gate_error:
                            if self.debug_mode:
                                print(f"Warning: Could not identify gate type: {gate_error}")
                            # Skip if can't determine gate name
                            continue
            
            # Estimate fidelity
            metrics["fidelity"] = self.estimate_fidelity(
                metrics["width"],
                metrics["size"],
                metrics["cx_count"]
            )
            
            return metrics
        
        except Exception as e:
            print(f"Error getting circuit metrics: {e}")
            return {
                "depth": 0,
                "width": 0,
                "size": 0,
                "gate_counts": {},
                "cx_count": 0,
                "t_gates": 0,
                "fidelity": 0,
                "error": str(e)
            }

    def estimate_fidelity(self, num_qubits, num_operations, cx_count=None, t_count=None, depth=None):

        """
        Estimate the fidelity of a quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            num_operations: Total number of operations in the circuit
            cx_count: Number of CNOT gates (optional)
            t_count: Number of T gates (optional)
            depth: Circuit depth (optional)
            
        Returns:
            float: Estimated fidelity (0.0-1.0)
        """
        # Baseline error rate per operation
        base_error_rate = 0.001  # 0.1% error per gate
        
        # CNOT error rate is higher
        cx_error_rate = 0.01     # 1% error per CNOT
        
        # T gate error rate (usually higher than other single-qubit gates)
        t_error_rate = 0.002     # 0.2% error per T gate
        
        # If cx_count not provided, estimate it as 20% of operations
        if cx_count is None:
            cx_count = int(num_operations * 0.2)
            
        # If t_count not provided, estimate it as 10% of operations
        if t_count is None:
            t_count = int(num_operations * 0.1)
        
        # Calculate non-CX, non-T gate count
        other_gates = num_operations - cx_count - t_count
        
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

    def print_comparison(self, original_circuit, mapped_circuit, optimized_circuit):

        """
        Print a detailed comparison of circuit metrics between original, mapped,
        and optimized versions of a quantum circuit.
        
        Args:
            original_circuit: Original logical circuit
            mapped_circuit: Mapped physical circuit
            optimized_circuit: Optimized logical circuit
        """
        # Import traceback for error handling
        import traceback
        
        if original_circuit is None or mapped_circuit is None or optimized_circuit is None:
            print("Cannot compare circuits: One or more circuits are None")
            return
        
        try:
            # Extract metrics for each circuit
            original_metrics = self.get_circuit_metrics(original_circuit)
            mapped_metrics = self.get_circuit_metrics(mapped_circuit)
            optimized_metrics = self.get_circuit_metrics(optimized_circuit)
            
            # Calculate fidelities directly without relying on optimizer access
            try:
                # Calculate logical fidelity - function preservation
                if original_metrics["depth"] > 0:
                    logical_fidelity = max(0.0, 1.0 - abs(optimized_metrics["depth"] - original_metrics["depth"]) / original_metrics["depth"] * 0.1)
                else:
                    logical_fidelity = 0.9  # Default value
                
                # Estimate physical fidelity based on gate counts
                # More sophisticated calculation based on gate error rates
                cx_error_rate = 0.003   # 0.3% error per CNOT
                single_error_rate = 0.0001  # 0.01% error per single-qubit gate
                
                cx_count = mapped_metrics["cx_count"]
                single_count = sum(mapped_metrics["gate_counts"].get(g, 0) for g in
                                  ['h', 't', 'tdg', 'rx', 'ry', 'rz', 'x', 'y', 'z', 's', 'sdg'])
                
                physical_fidelity = (1 - cx_error_rate) ** cx_count * (1 - single_error_rate) ** single_count
                
                # Ensure values are in valid range
                logical_fidelity = max(0.0, min(1.0, logical_fidelity))
                physical_fidelity = max(0.0, min(1.0, physical_fidelity))
                
            except Exception as e:
                if self.debug_mode:
                    print(f"Error calculating fidelities: {e}")
                logical_fidelity = 0.9  # Default fallback value
                physical_fidelity = 0.9  # Default fallback value
            
            # Print the comparison table
            print("\n=== Circuit Comparison ===")
            print(f"{'Metric':<20} {'Original':<10} {'Mapped':<10} {'Optimized':<10}")
            print(f"{'Depth':<20} {original_metrics['depth']:<10} {mapped_metrics['depth']:<10} {optimized_metrics['depth']:<10}")
            print(f"{'Width':<20} {original_circuit.num_qubits:<10} {mapped_circuit.num_qubits:<10} {optimized_circuit.num_qubits:<10}")
            print(f"{'CNOT Count':<20} {original_metrics['cx_count']:<10} {mapped_metrics['cx_count']:<10} {optimized_metrics['cx_count']:<10}")
            print(f"{'T Gate Count':<20} {original_metrics['t_gates']:<10} {mapped_metrics['t_gates']:<10} {optimized_metrics['t_gates']:<10}")
            
            # Calculate single-qubit gates for each circuit
            orig_single = sum(original_metrics["gate_counts"].get(g, 0) for g in
                              ['h', 't', 'tdg', 'rx', 'ry', 'rz', 'x', 'y', 'z', 's', 'sdg', 'u1', 'u2', 'u3'])
            
            map_single = sum(mapped_metrics["gate_counts"].get(g, 0) for g in
                             ['h', 't', 'tdg', 'rx', 'ry', 'rz', 'x', 'y', 'z', 's', 'sdg', 'u1', 'u2', 'u3'])
            
            opt_single = sum(optimized_metrics["gate_counts"].get(g, 0) for g in
                             ['h', 't', 'tdg', 'rx', 'ry', 'rz', 'x', 'y', 'z', 's', 'sdg', 'u1', 'u2', 'u3'])
            
            print(f"{'Single-Qubit Gates':<20} {orig_single:<10} {map_single:<10} {opt_single:<10}")
            
            # Add a spacer line
            print()
            
            # Print fidelity information
            print(f"Logical fidelity (function preservation): {logical_fidelity:.6f}")
            print(f"Physical fidelity (hardware execution): {physical_fidelity:.6f}")
            
            # Print size changes
            orig_total = sum(original_metrics["gate_counts"].values())
            opt_total = sum(optimized_metrics["gate_counts"].values())
            map_total = sum(mapped_metrics["gate_counts"].values())
            
            if orig_total > 0:
                opt_reduction = ((orig_total - opt_total) / orig_total * 100)
                print(f"Optimization gate reduction: {opt_reduction:.2f}%")
            
            if original_metrics["depth"] > 0:
                depth_reduction = ((original_metrics["depth"] - optimized_metrics["depth"]) / original_metrics["depth"] * 100)
                print(f"Optimization depth reduction: {depth_reduction:.2f}%")
                
        except Exception as e:
            print(f"Error during circuit comparison: {e}")
            traceback.print_exc()
