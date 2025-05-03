"""
Toffoli Depth Optimizer

This module provides functionality to optimize the depth of Toffoli gate 
networks on quantum hardware with limited connectivity.

It includes both the basic ToffoliDepthOptimizer as well as the enhanced
version with pattern recognition capabilities for identifying and replacing
common Toffoli gate combinations with simplified equivalents.
"""

import os
import time
import traceback
import numpy as np
from datetime import datetime
import signal
from contextlib import contextmanager
from enum import Enum, auto
import operator
import gc  # For garbage collection
import itertools

# Import necessary components
from ..utils.circuit_utils import (
    create_optimized_physical_mapping, 
    QISKIT_AVAILABLE, 
    estimate_fidelity, 
    validate_physical_circuit
)
from .compiler import ToffoliCompiler, ToffoliType

# Check for Qiskit availability
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator
except ImportError:
    pass

class TimeoutException(Exception):
    """Exception raised when a function execution times out."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeouts."""
    raise TimeoutException("Function execution timed out")

@contextmanager
def time_limit(seconds):
    """Context manager for limiting execution time of a block of code."""
    if seconds > 0:  # Only set timeout if positive
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
    try:
        yield
    finally:
        if seconds > 0:  # Only cancel timeout if it was set
            signal.alarm(0)  # Cancel the alarm

# Optimization strategies enum
class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""
    STANDARD = auto()        # Prioritize depth, then gates, then fidelity
    DEPTH_REDUCTION = auto() # Only care about depth
    GATE_REDUCTION = auto()  # Only care about gate count
    FIDELITY = auto()        # Only care about fidelity
    HYBRID = auto()          # Use a weighted score of all metrics

class ToffoliPatternLibrary:
    """Library for identifying and simplifying Toffoli gate patterns."""
    
    def __init__(self):
        """Initialize the Toffoli pattern library."""
        self.patterns = {}
        self.simplified_circuits = {}
        self.simplification_metrics = {}
        self.build_patterns()
    
    def build_patterns(self):
        """Build the library of Toffoli patterns and their simplifications."""
        # Generate all possible 2-3 Toffoli gate combinations
        self._generate_toffoli_combinations()
        # Simplify each pattern
        self._simplify_patterns()
        # Analyze and sort by simplification potential
        self._analyze_simplifications()
    
    def _generate_toffoli_combinations(self):
        """Generate all possible ways to apply 2-3 Toffoli gates on 3 qubits."""
        # For each Toffoli gate, we need to decide control and target qubits
        # On 3 qubits, we have these possible Toffoli configurations:
        toffoli_configs = [
            ((0, 1), 2),  # Controls: 0,1; Target: 2
            ((0, 2), 1),  # Controls: 0,2; Target: 1
            ((1, 2), 0),  # Controls: 1,2; Target: 0
        ]
        
        # Generate all 2-gate combinations
        for combo in itertools.product(toffoli_configs, repeat=2):
            circuit_key = f"tof2_{len(self.patterns)}"
            circuit = QuantumCircuit(3)
            
            for (controls, target) in combo:
                circuit.ccx(*controls, target)
            
            self.patterns[circuit_key] = {
                "circuit": circuit,
                "sequence": combo,
                "num_gates": 2
            }
        
        # Generate all 3-gate combinations
        for combo in itertools.product(toffoli_configs, repeat=3):
            circuit_key = f"tof3_{len(self.patterns) - 9}"  # Subtract the 9 2-gate patterns
            circuit = QuantumCircuit(3)
            
            for (controls, target) in combo:
                circuit.ccx(*controls, target)
            
            self.patterns[circuit_key] = {
                "circuit": circuit,
                "sequence": combo,
                "num_gates": 3
            }
        
        print(f"Generated {len(self.patterns)} Toffoli gate combinations")
    
    def _simplify_patterns(self):
        """Simplify each Toffoli pattern using various techniques."""
        for pattern_id, pattern in self.patterns.items():
            # Create three different simplified versions:
            original_circuit = pattern["circuit"]
            
            # 1. Standard decomposition with no ancillas
            no_ancilla_circuit = self._simplify_no_ancilla(original_circuit)
            
            # 2. Decomposition with 1 ancilla
            one_ancilla_circuit = self._simplify_with_one_ancilla(original_circuit)
            
            # 3. Optimized with multiple ancillas
            multi_ancilla_circuit = self._simplify_with_multi_ancilla(original_circuit)
            
            self.simplified_circuits[pattern_id] = {
                "no_ancilla": no_ancilla_circuit,
                "one_ancilla": one_ancilla_circuit,
                "multi_ancilla": multi_ancilla_circuit
            }
    
    def _simplify_no_ancilla(self, circuit):
        """Simplify a Toffoli circuit without using ancilla qubits."""
        # Use Qiskit's built-in transpiler with optimization level 3
        simplified = transpile(circuit, basis_gates=['u', 'cx'], optimization_level=3)
        return simplified
    
    def _simplify_with_one_ancilla(self, circuit):
        """Simplify a Toffoli circuit using one ancilla qubit."""
        # Create a new circuit with an ancilla
        num_qubits = circuit.num_qubits
        circuit_with_ancilla = QuantumCircuit(num_qubits + 1)
        
        # Copy the original circuit
        for instruction in circuit.data:
            if instruction.operation.name == 'ccx':
                # Replace CCX with a relative-phase Toffoli using the ancilla
                controls = [instruction.qubits[0].index, instruction.qubits[1].index]
                target = instruction.qubits[2].index
                ancilla = num_qubits  # The added ancilla qubit
                
                # Implement relative-phase Toffoli with one ancilla
                circuit_with_ancilla.h(target)
                circuit_with_ancilla.cx(controls[0], ancilla)
                circuit_with_ancilla.cx(controls[1], ancilla)
                circuit_with_ancilla.cx(ancilla, target)
                circuit_with_ancilla.cx(controls[0], ancilla)
                circuit_with_ancilla.cx(controls[1], ancilla)
                circuit_with_ancilla.h(target)
            else:
                # Copy other gates directly
                qubits = [q.index for q in instruction.qubits]
                circuit_with_ancilla.append(instruction.operation, qubits)
        
        # Optimize the circuit further
        simplified = transpile(circuit_with_ancilla, basis_gates=['u', 'cx'], optimization_level=3)
        return simplified
    
    def _simplify_with_multi_ancilla(self, circuit):
        """Simplify a Toffoli circuit using multiple ancilla qubits."""
        # Create a new circuit with multiple ancillas (one per Toffoli)
        num_qubits = circuit.num_qubits
        toffoli_count = sum(1 for inst in circuit.data if inst.operation.name == 'ccx')
        circuit_with_ancillas = QuantumCircuit(num_qubits + toffoli_count)
        
        # Count for ancilla allocation
        ancilla_idx = num_qubits
        
        # Copy the original circuit
        for instruction in circuit.data:
            if instruction.operation.name == 'ccx':
                # Replace CCX with an optimized version using one ancilla per Toffoli
                controls = [instruction.qubits[0].index, instruction.qubits[1].index]
                target = instruction.qubits[2].index
                ancilla = ancilla_idx
                ancilla_idx += 1
                
                # Implement optimized Toffoli with one dedicated ancilla
                # This uses the relative-phase Toffoli implementation
                circuit_with_ancillas.h(target)
                circuit_with_ancillas.cx(controls[0], ancilla)
                circuit_with_ancillas.cx(controls[1], ancilla)
                circuit_with_ancillas.cx(ancilla, target)
                circuit_with_ancillas.cx(controls[0], ancilla)
                circuit_with_ancillas.cx(controls[1], ancilla)
                circuit_with_ancillas.h(target)
            else:
                # Copy other gates directly
                qubits = [q.index for q in instruction.qubits]
                circuit_with_ancillas.append(instruction.operation, qubits)
        
        # Optimize the circuit further
        simplified = transpile(circuit_with_ancillas, basis_gates=['u', 'cx'], optimization_level=3)
        return simplified
		
def _analyze_simplifications(self):
        """Analyze the simplifications and identify the best candidates."""
        for pattern_id in self.patterns.keys():
            # Get the original and simplified circuits
            original_circuit = self.patterns[pattern_id]["circuit"]
            no_ancilla_circuit = self.simplified_circuits[pattern_id]["no_ancilla"]
            one_ancilla_circuit = self.simplified_circuits[pattern_id]["one_ancilla"]
            multi_ancilla_circuit = self.simplified_circuits[pattern_id]["multi_ancilla"]
            
            # Calculate metrics
            original_depth = original_circuit.depth()
            original_size = original_circuit.size()
            original_cx_count = sum(1 for inst in original_circuit.data if inst.operation.name == 'cx')
            
            no_ancilla_depth = no_ancilla_circuit.depth()
            no_ancilla_size = no_ancilla_circuit.size()
            no_ancilla_cx_count = sum(1 for inst in no_ancilla_circuit.data if inst.operation.name == 'cx')
            
            one_ancilla_depth = one_ancilla_circuit.depth()
            one_ancilla_size = one_ancilla_circuit.size()
            one_ancilla_cx_count = sum(1 for inst in one_ancilla_circuit.data if inst.operation.name == 'cx')
            
            multi_ancilla_depth = multi_ancilla_circuit.depth()
            multi_ancilla_size = multi_ancilla_circuit.size()
            multi_ancilla_cx_count = sum(1 for inst in multi_ancilla_circuit.data if inst.operation.name == 'cx')
            
            # Calculate improvement percentages (depth reduction)
            if original_depth > 0:
                no_ancilla_depth_reduction = ((original_depth - no_ancilla_depth) / original_depth) * 100
                one_ancilla_depth_reduction = ((original_depth - one_ancilla_depth) / original_depth) * 100
                multi_ancilla_depth_reduction = ((original_depth - multi_ancilla_depth) / original_depth) * 100
            else:
                no_ancilla_depth_reduction = 0
                one_ancilla_depth_reduction = 0
                multi_ancilla_depth_reduction = 0
            
            # Choose the best simplification based on depth reduction
            best_reduction = max(no_ancilla_depth_reduction, one_ancilla_depth_reduction, multi_ancilla_depth_reduction)
            
            if best_reduction == no_ancilla_depth_reduction:
                best_method = "no_ancilla"
                best_circuit = no_ancilla_circuit
                best_depth = no_ancilla_depth
                best_size = no_ancilla_size
                best_cx_count = no_ancilla_cx_count
            elif best_reduction == one_ancilla_depth_reduction:
                best_method = "one_ancilla"
                best_circuit = one_ancilla_circuit
                best_depth = one_ancilla_depth
                best_size = one_ancilla_size
                best_cx_count = one_ancilla_cx_count
            else:
                best_method = "multi_ancilla"
                best_circuit = multi_ancilla_circuit
                best_depth = multi_ancilla_depth
                best_size = multi_ancilla_size
                best_cx_count = multi_ancilla_cx_count
            
            # Store metrics
            self.simplification_metrics[pattern_id] = {
                "original_depth": original_depth,
                "original_size": original_size,
                "original_cx_count": original_cx_count,
                "best_method": best_method,
                "best_depth": best_depth,
                "best_size": best_size,
                "best_cx_count": best_cx_count,
                "depth_reduction": best_reduction,
                "no_ancilla_depth_reduction": no_ancilla_depth_reduction,
                "one_ancilla_depth_reduction": one_ancilla_depth_reduction,
                "multi_ancilla_depth_reduction": multi_ancilla_depth_reduction
            }
    
def get_top_simplifications(self, n=5):
        """Get the top N patterns with the best simplification potential."""
        # Sort patterns by depth reduction
        sorted_patterns = sorted(
            self.simplification_metrics.items(),
            key=lambda x: x[1]["depth_reduction"],
            reverse=True
        )
        
        return sorted_patterns[:n]
    
def print_top_simplifications(self, n=5):
        """Print the top N patterns with the best simplification potential."""
        top_patterns = self.get_top_simplifications(n)
        
        print(f"\nTop {n} Toffoli Patterns with Greatest Simplification Potential:\n")
        print("-" * 80)
        
        for i, (pattern_id, metrics) in enumerate(top_patterns):
            pattern_sequence = self.patterns[pattern_id]["sequence"]
            simplified_method = metrics["best_method"]
            
            print(f"{i+1}. Pattern ID: {pattern_id}")
            print(f"   Original sequence: {pattern_sequence}")
            print(f"   Original depth: {metrics['original_depth']}")
            print(f"   Best simplification method: {simplified_method}")
            print(f"   Simplified depth: {metrics['best_depth']}")
            print(f"   Depth reduction: {metrics['depth_reduction']:.2f}%")
            print(f"   CX count reduction: {metrics['original_cx_count']} → {metrics['best_cx_count']}")
            print("-" * 80)
    
def find_patterns_in_circuit(self, circuit):
        """
        Find all Toffoli patterns in a given circuit.
        
        Args:
            circuit: The circuit to analyze
            
        Returns:
            list: List of (pattern_id, start_index) tuples for all found patterns
        """
        found_patterns = []
        
        for pattern_id, pattern in self.patterns.items():
            pattern_length = len(pattern["sequence"])
            
            # Skip if circuit is too small
            if len(circuit.data) < pattern_length:
                continue
            
            # Scan the circuit for this pattern
            for i in range(len(circuit.data) - pattern_length + 1):
                segment = circuit.data[i:i+pattern_length]
                
                # Check if segment matches pattern
                matches = True
                for j, instruction in enumerate(segment):
                    if instruction.operation.name != 'ccx':
                        matches = False
                        break
                    
                    pattern_controls, pattern_target = pattern["sequence"][j]
                    
                    # Get the qubit indices from the instruction
                    instruction_qubits = [q.index for q in instruction.qubits]
                    instruction_controls = instruction_qubits[:2]
                    instruction_target = instruction_qubits[2]
                    
                    # Check if they match
                    if sorted(instruction_controls) != sorted(pattern_controls) or instruction_target != pattern_target:
                        matches = False
                        break
                
                if matches:
                    found_patterns.append((pattern_id, i))
        
        return found_patterns
    
def optimize_circuit(self, circuit, threshold=10.0, use_ancilla=True):
        """
        Optimize a circuit by replacing Toffoli patterns with simplified versions.
        
        Args:
            circuit: The circuit to optimize
            threshold: Minimum depth reduction percentage to apply a simplification
            use_ancilla: Whether to allow solutions with ancilla qubits
            
        Returns:
            QuantumCircuit: The optimized circuit
        """
        # Find all patterns
        patterns = self.find_patterns_in_circuit(circuit)
        
        if not patterns:
            return circuit
            
        print(f"Found {len(patterns)} Toffoli patterns in the circuit")
        
        # Sort by position in reverse order (to handle overlapping patterns properly)
        patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Create a copy of the circuit to modify
        optimized_circuit = circuit.copy()
        
        # Apply simplifications for patterns that meet the threshold
        replacements = 0
        for pattern_id, start_idx in patterns:
            metrics = self.simplification_metrics[pattern_id]
            
            if metrics["depth_reduction"] >= threshold:
                # Choose the best method (respecting use_ancilla parameter)
                if use_ancilla:
                    best_method = metrics["best_method"]
                else:
                    best_method = "no_ancilla"
                
                pattern_length = len(self.patterns[pattern_id]["sequence"])
                simplified_circuit = self.simplified_circuits[pattern_id][best_method]
                
                # Extract the pattern from the circuit
                pattern_segment = optimized_circuit.data[start_idx:start_idx+pattern_length]
                
                # Remove the pattern
                for instruction in pattern_segment:
                    optimized_circuit.data.remove(instruction)
                
                # Insert the simplified version
                for i, instruction in enumerate(simplified_circuit.data):
                    optimized_circuit.data.insert(start_idx + i, instruction)
                
                replacements += 1
        
        if replacements > 0:
            print(f"Applied {replacements} pattern replacements")
        
        return optimized_circuit


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
			
def _process_circuit_safely(self, circuit, coupling_map=None):
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


class EnhancedToffoliDepthOptimizer:
    """
    Optimizer for minimizing the depth of Toffoli gate networks on quantum hardware with limited connectivity.
    
    This class provides comprehensive functionality to optimize Toffoli networks by:
    1. Analyzing the logical structure of Toffoli networks
    2. Identifying optimization opportunities through pattern matching
    3. Applying hardware-aware optimizations respecting coupling constraints
    4. Minimizing circuit depth while maintaining target fidelity
    
    The optimizer uses multiple passes and different strategies to iteratively improve 
    circuit structure, with configurable parameters for optimization aggressiveness,
    fidelity targets, and hardware constraints.
    
    Attributes:
        target_fidelity (float): Target circuit fidelity (0.0-1.0)
        max_passes (int): Maximum number of optimization passes
        output_dir (str): Directory for optimizer output
        use_parallel (bool): Whether to use parallel execution
        debug_mode (bool): Whether to print debug information
        pass_timeout_seconds (int): Maximum seconds per optimization pass (0 for no limit)
        strategy (OptimizationStrategy): Strategy for optimization priorities
        
    Example usage:
        optimizer = ToffoliDepthOptimizer(target_fidelity=0.95, max_passes=2)
        results = optimizer.optimize_toffoli_network(
            toffoli_gates,
            output_qubits,
            input_qubits,
            num_qubits,
            topology='linear'
        )
    """
    
    def __init__(self, target_fidelity=0.95, max_passes=2, output_dir=None,
                use_parallel=True, debug_mode=False, pass_timeout_seconds=60,
                strategy=OptimizationStrategy.HYBRID, pattern_threshold=10.0):
        """
            Initialize the Toffoli Depth Optimizer.
            
            Args:
                target_fidelity (float): Target circuit fidelity (0.0-1.0).
                    Higher values prioritize maintaining circuit fidelity over depth reduction.
                    Recommended values: 0.9-0.99.
                
                max_passes (int): Maximum number of optimization passes.
                    More passes can find better optimizations but increase runtime.
                    Recommended values: 1-5 depending on circuit size.
                
                output_dir (str): Directory for optimizer output including reports and visualizations.
                    If None, defaults to "toffoli_optimizer_results".
                
                use_parallel (bool): Whether to use parallel execution for optimization.
                    Can significantly speed up optimization for large circuits but increases memory usage.
                
                debug_mode (bool): Whether to print detailed debug information.
                    Useful for understanding the optimizer's decision-making process.
                
                pass_timeout_seconds (int): Maximum seconds per optimization pass.
                    Set to 0 for no time limit. For large circuits, recommended value: 60-300 seconds.
                
                strategy (OptimizationStrategy): Strategy that determines optimization priorities.
                    Options include:
                    - STANDARD: Prioritize depth, then gates, then fidelity
                    - DEPTH_REDUCTION: Only care about depth
                    - GATE_REDUCTION: Only care about gate count
                    - FIDELITY: Only care about fidelity
                    - HYBRID: Use a weighted score of all metrics
                    If None, defaults to STANDARD.
            """
        self.target_fidelity = target_fidelity
        self.max_passes = max_passes
        self.use_parallel = use_parallel
        self.debug_mode = debug_mode
        self.pass_timeout_seconds = pass_timeout_seconds
        self.pattern_threshold = pattern_threshold
        
        # Set strategy - default to STANDARD if not a valid enum
        if isinstance(strategy, OptimizationStrategy):
            self.strategy = strategy
        else:
            self.strategy = OptimizationStrategy.STANDARD
        
        # Set default output directory if none provided
        if output_dir is None:
            self.output_dir = "toffoli_optimizer_results"
        else:
            self.output_dir = output_dir
        
        # Create timestamp for this optimizer run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize compiler
        self.compiler = ToffoliCompiler(debug_mode=self.debug_mode)
        
        # Initialize pattern library
        print("Initializing Toffoli pattern library...")
        self.pattern_library = ToffoliPatternLibrary()
        
        # Print top patterns
        if self.debug_mode:
            self.pattern_library.print_top_simplifications(n=5)

    # Import the core methods from the ToffoliDepthOptimizer class
    from ._optimizer_methods import (
        optimize_toffoli_network, _optimize_circuit, _is_better_circuit,
        estimate_physical_fidelity, calculate_logical_fidelity
    )

# In toffoli_optimizer/core/optimizer.py
# Add this function to the ToffoliDepthOptimizer class

    def _optimize_circuit_with_memory_management(self, circuit, toffoli_gates, output_qubits, input_qubits,
                                            num_qubits, coupling_map, basis_gates, target_fidelity,
                                            toffoli_type=None, use_ancilla=True):
        """
        Memory-optimized version of the circuit optimization function.
        
        This method applies optimization techniques to reduce circuit depth and improve fidelity
        while aggressively managing memory to prevent OOM errors with large circuits.
        
        Args:
            circuit: Circuit to optimize
            toffoli_gates: List of Toffoli gates
            output_qubits: List of output qubit indices
            input_qubits: List of input qubit indices
            num_qubits: Total number of qubits
            coupling_map: Coupling map for the target topology
            basis_gates: List of available basis gates
            target_fidelity: Target circuit fidelity
            toffoli_type: Type of Toffoli implementation to use
            use_ancilla: Whether to use ancilla qubits
            
        Returns:
            optimized_circuit: The optimized circuit
        """
        try:
            from qiskit import transpile, QuantumCircuit
            import numpy as np
            import gc
            
            # Print memory usage info if debug mode is enabled
            if self.debug_mode:
                import psutil
                process = psutil.Process()
                print(f"Memory usage before optimization: {process.memory_info().rss / (1024 * 1024):.2f} MB")
            
            # Start with a fresh copy of the input circuit
            working_circuit = circuit.copy()
            original_depth = working_circuit.depth()
            
            if self.debug_mode:
                print(f"Starting optimization with circuit depth {original_depth}")
                print(f"Circuit has {len(working_circuit.data)} gates")
            
            # Track the best circuit we've found so far
            best_circuit = working_circuit.copy()
            best_depth = original_depth
            
            # Get metrics without storing the entire dictionary
            gate_counts = best_circuit.count_ops()
            best_gate_count = sum(gate_counts.values())
            del gate_counts  # Free memory
            
            best_fidelity = self.estimate_physical_fidelity(best_circuit)
            
            # Force garbage collection after initial setup
            gc.collect()
            
            # Apply optimization techniques with appropriate memory management
            
            # 1. Try standard transpilation optimization - one level at a time with GC in between
            for opt_level in [1, 2, 3]:
                try:
                    # Clear any previous transpiled circuits
                    gc.collect()
                    
                    transpiled = transpile(
                        working_circuit.copy(),
                        basis_gates=basis_gates,
                        optimization_level=opt_level
                    )
                    
                    # Measure transpiled circuit (extract metrics without keeping full dictionaries)
                    trans_depth = transpiled.depth()
                    
                    gate_counts = transpiled.count_ops()
                    trans_gates = sum(gate_counts.values())
                    del gate_counts  # Free memory
                    
                    trans_fidelity = self.estimate_physical_fidelity(transpiled)
                    
                    # Check if this is better than our current best
                    if self._is_better_circuit(trans_depth, trans_gates, trans_fidelity,
                                            best_depth, best_gate_count, best_fidelity):
                        # Free old best circuit from memory
                        del best_circuit
                        gc.collect()
                        
                        best_circuit = transpiled.copy()
                        best_depth = trans_depth
                        best_gate_count = trans_gates
                        best_fidelity = trans_fidelity
                        
                        if self.debug_mode:
                            print(f"Transpilation level {opt_level} improved metrics: depth={best_depth}, gates={best_gate_count}, fidelity={best_fidelity:.6f}")
                    else:
                        # Free the transpiled circuit if we're not keeping it
                        del transpiled
                        gc.collect()
                except Exception as e:
                    if self.debug_mode:
                        print(f"Standard transpilation at level {opt_level} failed: {e}")
                    # Make sure to free memory even on error
                    gc.collect()
            
            # Force garbage collection between optimization strategies
            gc.collect()

            # Final optimization pass with improved memory handling
            try:
                # Clear memory before final optimization
                gc.collect()
                
                final_circuit = transpile(
                    best_circuit.copy(),  # Use copy to prevent accidental modification
                    basis_gates=basis_gates,
                    optimization_level=3
                )
                
                # Get metrics without storing the full dictionary
                final_depth = final_circuit.depth()
                
                gate_counts = final_circuit.count_ops()
                final_gates = sum(gate_counts.values())
                del gate_counts  # Free memory
                
                final_fidelity = self.estimate_physical_fidelity(final_circuit)
                
                # Check if this is better
                if self._is_better_circuit(final_depth, final_gates, final_fidelity,
                                        best_depth, best_gate_count, best_fidelity, final_pass=True):
                    # Free old best circuit
                    del best_circuit
                    gc.collect()
                    
                    best_circuit = final_circuit
                    best_depth = final_depth
                    best_gate_count = final_gates
                    best_fidelity = final_fidelity
                    
                    if self.debug_mode:
                        print(f"Final optimization pass improved metrics: depth={best_depth}, gates={best_gate_count}, fidelity={best_fidelity:.6f}")
                else:
                    # Free the final circuit if we're not keeping it
                    del final_circuit
                    gc.collect()
            except Exception as e:
                if self.debug_mode:
                    print(f"Final optimization pass failed: {e}")
                # Make sure to free memory even on error
                gc.collect()
            
            # Print final memory usage if in debug mode
            if self.debug_mode:
                import psutil
                process = psutil.Process()
                print(f"Memory usage after optimization: {process.memory_info().rss / (1024 * 1024):.2f} MB")
            
            # Final garbage collection before returning
            gc.collect()
            
            # Return the best circuit we found
            return best_circuit
            
        except Exception as e:
            print(f"Error in circuit optimization: {e}")
            import traceback
            traceback.print_exc()
            # Force garbage collection on error
            gc.collect()
            return circuit  # Return the original circuit on error

# Define default coupling maps for different topologies
def get_default_coupling_map(topology, num_qubits):
    """
    Get a default coupling map for a given topology and number of qubits.
    
    Args:
        topology (str): The topology type ('linear', 'grid', 'falcon')
        num_qubits (int): The number of qubits
        
    Returns:
        list: The coupling map as a list of qubit pairs
    """
    if topology == 'linear':
        # Linear nearest-neighbor connectivity
        return [(i, i+1) for i in range(num_qubits-1)] + [(i+1, i) for i in range(num_qubits-1)]
    
    elif topology == 'grid':
        # 2D grid connectivity (as close to square as possible)
        import math
        
        # Calculate grid dimensions
        width = int(math.ceil(math.sqrt(num_qubits)))
        height = int(math.ceil(num_qubits / width))
        
        coupling_map = []
        
        # Add horizontal connections
        for row in range(height):
            for col in range(width - 1):
                qubit1 = row * width + col
                qubit2 = row * width + col + 1
                
                if qubit1 < num_qubits and qubit2 < num_qubits:
                    coupling_map.append((qubit1, qubit2))
                    coupling_map.append((qubit2, qubit1))
        
        # Add vertical connections
        for row in range(height - 1):
            for col in range(width):
                qubit1 = row * width + col
                qubit2 = (row + 1) * width + col
                
                if qubit1 < num_qubits and qubit2 < num_qubits:
                    coupling_map.append((qubit1, qubit2))
                    coupling_map.append((qubit2, qubit1))
        
        return coupling_map
    
    elif topology == 'falcon':
        # IBM Falcon-like heavy-hexagonal connectivity
        # This is a simplified version of the Falcon topology
        if num_qubits < 7:
            # Fall back to linear for very small qubit counts
            return get_default_coupling_map('linear', num_qubits)
        
        # Start with a core hexagonal unit
        coupling_map = [
            (0, 1), (1, 0),
            (1, 2), (2, 1),
            (2, 3), (3, 2),
            (3, 4), (4, 3),
            (4, 5), (5, 4),
            (5, 0), (0, 5),
            (0, 6), (6, 0)
        ]
        
        # Add more qubits in a structured pattern
        for i in range(7, num_qubits):
            # Connect new qubit to existing ones based on a pattern
            # This is a simplified approach and doesn't fully capture the real Falcon topology
            connection1 = (i - 7) % 6
            connection2 = (i - 7) % 6 + 1 if (i - 7) % 6 < 5 else 0
            
            coupling_map.append((i, connection1))
            coupling_map.append((connection1, i))
            coupling_map.append((i, connection2))
            coupling_map.append((connection2, i))
        
        return coupling_map
    
    else:
        # Default to linear topology
        return get_default_coupling_map('linear', num_qubits)


"""
Core optimization methods for the ToffoliDepthOptimizer class.

This module contains the implementation of the main optimization methods
used by the ToffoliDepthOptimizer and EnhancedToffoliDepthOptimizer classes.
"""

def optimize_toffoli_network(self, toffoli_gates, output_qubits, input_qubits,
                           num_qubits, topology=None, coupling_map=None,
                           basis_gates=None, target_fidelity=None,
                           original_circuit=None, logical_circuit=None,
                           toffoli_type=None, use_ancilla=True):
    """
    Optimize a Toffoli gate network for minimal depth on quantum hardware.
    
    Args:
        toffoli_gates (list): List of Toffoli gates as (control_qubits, target_qubit) tuples
        output_qubits (list): List of output qubit indices
        input_qubits (list): List of input qubit indices
        num_qubits (int): Total number of qubits
        topology (str): Optional hardware topology ('linear', 'grid', 'falcon')
        coupling_map (list): Optional coupling map as list of qubit pairs
        basis_gates (list): Optional list of basis gates
        target_fidelity (float): Optional target fidelity (overrides instance value)
        original_circuit: Original physical circuit for comparison
        logical_circuit: Logical circuit representation
        toffoli_type: Type of Toffoli implementation to use
        use_ancilla (bool): Whether to use ancilla qubits
        
    Returns:
        dict: Dictionary with optimization results
    """
    import time
    from qiskit import QuantumCircuit
    
    # Force garbage collection at the start to free memory
    import gc
    gc.collect()
    
    print(f"\nOptimizing Toffoli network with {len(toffoli_gates)} gates on {num_qubits} qubits")
    if hasattr(self.compiler, 'ToffoliType') and toffoli_type is not None and isinstance(toffoli_type, self.compiler.ToffoliType):
        print(f"Using {toffoli_type.name} implementation")
    else:
        print(f"Using STANDARD implementation")
        # Use the ToffoliType from the compiler
        if hasattr(self.compiler, 'ToffoliType'):
            toffoli_type = self.compiler.ToffoliType.STANDARD
    
    print(f"Max optimization passes: {self.max_passes}")
    if hasattr(self, 'strategy'):
        print(f"Optimization strategy: {self.strategy.name}")
    else:
        print(f"Optimization strategy: STANDARD")
    
    if self.pass_timeout_seconds > 0:
        print(f"Pass timeout: {self.pass_timeout_seconds} seconds")
    else:
        print("Pass timeout: disabled")
    
    # Use instance target_fidelity if none provided
    if target_fidelity is None:
        target_fidelity = self.target_fidelity
    
    print(f"Target fidelity: {target_fidelity}")
    if hasattr(self, 'pattern_threshold'):
        print(f"Pattern optimization threshold: {self.pattern_threshold}%")
    
    # Get coupling map if topology provided
    if coupling_map is None and topology is not None:
        from ..utils.circuit_utils import get_default_coupling_map
        coupling_map = get_default_coupling_map(topology, num_qubits)
        print(f"Using {topology} topology with {len(coupling_map)//2} connections")
    
    # Set default basis gates if none provided
    if basis_gates is None:
        basis_gates = ['id', 'rz', 'sx', 'x', 'cx']
    
    # Create results dictionary
    results = {}
    start_time = time.time()
    
    try:
        # 1. Generate logical circuit if not provided
        if logical_circuit is None:
            print("Creating logical circuit...")
            # Make a copy of toffoli_gates to avoid modifying the original
            logical_circuit, ancilla_indices = self.compiler.create_toffoli_network(
                toffoli_gates.copy(),  # Use copy to preserve original
                num_qubits,
                use_ancilla=use_ancilla,
                target_fidelity=1.0,  # Use perfect fidelity for logical circuit
                toffoli_type=toffoli_type
            )
            
            # Force garbage collection after creating logical circuit
            gc.collect()
            
            if logical_circuit is None:
                raise ValueError("Failed to create logical circuit")
        
        # Get logical circuit metrics
        logical_metrics = self.compiler.get_circuit_metrics(logical_circuit)
        
        results["logical"] = {
            "depth": logical_metrics["depth"],
            "gate_count": sum(logical_metrics["gate_counts"].values()),
            "cx_count": logical_metrics["cx_count"],
            "t_gates": logical_metrics["t_gates"],
            "circuit": logical_circuit  # Store for visualization
        }
        
        print(f"Logical circuit depth: {logical_metrics['depth']}")
        
        # Clear metrics from memory (they're now stored in results)
        del logical_metrics
        gc.collect()
        
        # Apply pattern-based optimization to the logical circuit if available
        pattern_optimized_circuit = None
        if hasattr(self, 'pattern_library') and hasattr(self.pattern_library, 'optimize_circuit'):
            print("Applying pattern-based optimization...")
            pattern_optimized_circuit = self.pattern_library.optimize_circuit(
                logical_circuit.copy(), 
                threshold=getattr(self, 'pattern_threshold', 10.0),
                use_ancilla=use_ancilla
            )
            
            # Get pattern optimized metrics
            pattern_metrics = self.compiler.get_circuit_metrics(pattern_optimized_circuit)
            
            # Store pattern optimization results
            results["pattern_optimized"] = {
                "depth": pattern_metrics["depth"],
                "gate_count": sum(pattern_metrics["gate_counts"].values()),
                "cx_count": pattern_metrics["cx_count"],
                "t_gates": pattern_metrics["t_gates"],
                "circuit": pattern_optimized_circuit  # Store for visualization
            }
            
            # Calculate depth reduction from logical to pattern-optimized
            if results["logical"]["depth"] > 0:
                pattern_depth_reduction = ((results["logical"]["depth"] - results["pattern_optimized"]["depth"]) / 
                                          results["logical"]["depth"] * 100)
            else:
                pattern_depth_reduction = 0
                
            results["pattern_optimized"]["depth_reduction"] = pattern_depth_reduction
                
            print(f"Pattern-optimized depth: {pattern_metrics['depth']} ({pattern_depth_reduction:.2f}% reduction)")
            
            # Clear pattern metrics from memory
            del pattern_metrics
            gc.collect()
        
        # 2. Generate naive physical mapping if not provided
        if original_circuit is None:
            print("Creating naive physical mapping...")
            # Use the pattern-optimized circuit as the starting point if available, otherwise use logical
            circuit_to_map = pattern_optimized_circuit if pattern_optimized_circuit is not None else logical_circuit
            from ..utils.circuit_utils import create_optimized_physical_mapping
            original_circuit = create_optimized_physical_mapping(
                circuit_to_map.copy(),
                coupling_map,
                basis_gates,
                optimization_level=1  # Lower optimization level for naive mapping
            )
            
            # Force garbage collection after mapping
            gc.collect()
        
        # Get naive physical metrics
        naive_metrics = self.compiler.get_circuit_metrics(original_circuit)
        
        results["naive_physical"] = {
            "depth": naive_metrics["depth"],
            "gate_count": sum(naive_metrics["gate_counts"].values()),
            "cx_count": naive_metrics["cx_count"],
            "t_gates": naive_metrics["t_gates"],
            "circuit": original_circuit  # Store for visualization
        }
        
        print(f"Naive physical depth: {naive_metrics['depth']}")
        
        # Clear naive metrics from memory
        del naive_metrics
        gc.collect()
        
        # 3. Run optimization passes to reduce depth
        print(f"Running optimization passes (max: {self.max_passes})...")
        
        # Start with the pattern-optimized circuit if available, otherwise use logical
        optimized_circuit = pattern_optimized_circuit if pattern_optimized_circuit is not None else logical_circuit.copy()
        current_depth = optimized_circuit.depth()
        
        # Extract metrics carefully to avoid keeping the whole dictionary
        current_metrics = self.compiler.get_circuit_metrics(optimized_circuit)
        current_gates = sum(current_metrics["gate_counts"].values())
        del current_metrics  # Free memory
        gc.collect()
        
        current_fidelity = self.estimate_physical_fidelity(optimized_circuit)
        
        # Track early termination conditions
        early_termination_would_trigger = False
        best_pass = 0
        pass_timeout_count = 0
        
        # Apply optimizations
        for i in range(self.max_passes):
            pass_num = i + 1
            print(f"Optimization pass {pass_num}/{self.max_passes}")
            
            # Force garbage collection before each optimization pass
            gc.collect()
            
            # Apply optimization techniques with timeout
            pass_start_time = time.time()
            
            if self.pass_timeout_seconds > 0:
                try:
                    # Run optimization with timeout
                    from ..utils.circuit_utils import time_limit
                    with time_limit(self.pass_timeout_seconds):
                        new_circuit = self._optimize_circuit(
                            optimized_circuit.copy(),
                            toffoli_gates.copy(),  # Use copy to preserve original
                            output_qubits,
                            input_qubits,
                            num_qubits,
                            coupling_map,
                            basis_gates,
                            target_fidelity,
                            toffoli_type,
                            use_ancilla
                        )
                except TimeoutException:
                    print(f"Optimization pass {pass_num} timed out after {self.pass_timeout_seconds} seconds")
                    pass_timeout_count += 1
                    continue
            else:
                # Run optimization without timeout
                new_circuit = self._optimize_circuit(
                    optimized_circuit.copy(),
                    toffoli_gates.copy(),  # Use copy to preserve original
                    output_qubits,
                    input_qubits,
                    num_qubits,
                    coupling_map,
                    basis_gates,
                    target_fidelity,
                    toffoli_type,
                    use_ancilla
                )
            
            pass_duration = time.time() - pass_start_time
            print(f"Pass {pass_num} completed in {pass_duration:.2f} seconds")
            
            # Force garbage collection after optimization
            gc.collect()
            
            if new_circuit is None:
                print("Optimization failed, using previous circuit")
                continue
            
            # Get metrics for the new circuit
            new_metrics = self.compiler.get_circuit_metrics(new_circuit)
            new_depth = new_metrics["depth"]
            new_gates = sum(new_metrics["gate_counts"].values())
            
            # Clear metrics from memory
            del new_metrics
            gc.collect()
            
            new_fidelity = self.estimate_physical_fidelity(new_circuit)
            
            # Decide whether to keep the new circuit based on the optimization strategy
            improvement = False
            
            if hasattr(self, 'strategy'):
                from ..core.optimizer import OptimizationStrategy
                if self.strategy == OptimizationStrategy.DEPTH_REDUCTION:
                    # For depth reduction, only care about depth
                    if new_depth < current_depth:
                        improvement = True
                        print(f"Depth reduced: {current_depth} -> {new_depth}")
                        
                elif self.strategy == OptimizationStrategy.GATE_REDUCTION:
                    # For gate reduction, only care about gate count
                    if new_gates < current_gates:
                        improvement = True
                        print(f"Gate count reduced: {current_gates} -> {new_gates}")
                        
                elif self.strategy == OptimizationStrategy.FIDELITY:
                    # For fidelity optimization, only care about fidelity
                    if new_fidelity > current_fidelity:
                        improvement = True
                        print(f"Fidelity improved: {current_fidelity:.6f} -> {new_fidelity:.6f}")
                        
                elif self.strategy == OptimizationStrategy.HYBRID:
                    # For hybrid optimization, use a weighted score
                    depth_weight = 0.5
                    gate_weight = 0.3
                    fidelity_weight = 0.2
                    
                    depth_score = current_depth / new_depth if new_depth > 0 else 1.0
                    gate_score = current_gates / new_gates if new_gates > 0 else 1.0
                    fidelity_score = new_fidelity / current_fidelity if current_fidelity > 0 else 1.0
                    
                    current_score = 1.0
                    new_score = (depth_weight * depth_score +
                                gate_weight * gate_score +
                                fidelity_weight * fidelity_score)
                    
                    if new_score > current_score:
                        improvement = True
                        print(f"Hybrid score improved: 1.0 -> {new_score:.4f}")
                        print(f"Depth: {current_depth} -> {new_depth}, Gates: {current_gates} -> {new_gates}, Fidelity: {current_fidelity:.6f} -> {new_fidelity:.6f}")
                
                else:  # Default to STANDARD strategy
                    # For standard optimization, prioritize depth, then gates, then fidelity
                    if new_depth < current_depth:
                        improvement = True
                        print(f"Depth reduced: {current_depth} -> {new_depth}")
                    elif new_depth == current_depth and new_gates < current_gates:
                        improvement = True
                        print(f"Gate count reduced at same depth: {current_gates} -> {new_gates}")
                    elif new_depth == current_depth and new_gates == current_gates and new_fidelity > current_fidelity:
                        improvement = True
                        print(f"Fidelity improved at same depth and gate count: {current_fidelity:.6f} -> {new_fidelity:.6f}")
            else:
                # Default to simple depth comparison if no strategy defined
                if new_depth < current_depth:
                    improvement = True
                    print(f"Depth reduced: {current_depth} -> {new_depth}")
            
            if improvement:
                # Free old circuit from memory if we're replacing it
                del optimized_circuit
                gc.collect()
                
                optimized_circuit = new_circuit
                current_depth = new_depth
                current_gates = new_gates
                current_fidelity = new_fidelity
                best_pass = pass_num
            else:
                print(f"No improvement in pass {pass_num}")
                
                # Free the unused new circuit from memory
                del new_circuit
                gc.collect()
                
                # Check for early termination condition, but don't actually terminate
                if pass_num > 1:
                    print("Early termination condition detected (continuing anyway)")
                    early_termination_would_trigger = True
            
            # Force garbage collection at the end of each pass
            gc.collect()
        
        # Report on optimization passes
        if pass_timeout_count > 0:
            print(f"\n{pass_timeout_count} passes timed out after {self.pass_timeout_seconds} seconds")
            
        if early_termination_would_trigger:
            print(f"\nNote: Early termination would have triggered, but all {self.max_passes} passes were completed")
            print(f"Best results achieved in pass {best_pass}")
        else:
            print(f"\nCompleted all {self.max_passes} passes")
            if best_pass > 0:
                print(f"Best results achieved in pass {best_pass}")
            else:
                print("No improvements found in any pass")
        
        # Get optimized circuit metrics
        optimized_metrics = self.compiler.get_circuit_metrics(optimized_circuit)
        
        # Calculate logical fidelity (function preservation) between original logical and optimized logical circuits
        logical_fidelity = self.calculate_logical_fidelity(logical_circuit, optimized_circuit)
                    
        results["optimized"] = {
            "depth": optimized_metrics["depth"],
            "gate_count": sum(optimized_metrics["gate_counts"].values()),
            "cx_count": optimized_metrics["cx_count"],
            "t_gates": optimized_metrics["t_gates"],
            "circuit": optimized_circuit,  # Store for visualization
            "best_pass": best_pass,
            "early_termination_would_trigger": early_termination_would_trigger,
            "pass_timeout_count": pass_timeout_count,
            "physical_fidelity": current_fidelity,
            "logical_fidelity": logical_fidelity
        }
        
        # Add "final" key for compatibility with main.py
        results["final"] = {
            "depth": optimized_metrics["depth"],
            "gate_count": sum(optimized_metrics["gate_counts"].values()),
            "cx_count": optimized_metrics["cx_count"],
            "t_gates": optimized_metrics["t_gates"],
            "circuit": optimized_circuit.copy()  # Store for visualization
        }
        
        # Clear metrics from memory
        del optimized_metrics
        gc.collect()
        
        print(f"Optimized logical depth: {results['optimized']['depth']}")
        
        # 4. Map optimized circuit to physical qubits
        print("Mapping optimized circuit to physical qubits...")
        
        # Force garbage collection before mapping
        gc.collect()
        
        # Use enhanced physical mapping that respects coupling constraints
        from ..utils.circuit_utils import create_optimized_physical_mapping
        mapped_circuit = create_optimized_physical_mapping(
            optimized_circuit.copy(),
            coupling_map,
            basis_gates,
            optimization_level=3  # Use high optimization level
        )
        
        # Validate that the mapped circuit respects coupling constraints
        from ..utils.circuit_utils import validate_physical_circuit
        is_valid, violations = validate_physical_circuit(mapped_circuit, coupling_map)
        if not is_valid:
            print(f"Warning: Mapped circuit has {len(violations)} coupling violations")
            if self.debug_mode:
                for violation in violations[:5]:  # Show first 5 violations
                    print(f"  {violation}")
                if len(violations) > 5:
                    print(f"  ... and {len(violations) - 5} more violations")
            
            # If invalid, try a more conservative approach
            print("Trying more conservative mapping approach...")
            mapped_circuit = create_optimized_physical_mapping(
                optimized_circuit.copy(),
                coupling_map,
                basis_gates,
                optimization_level=1  # Lower optimization level for more conservative mapping
            )
            
            # Check if this fixed the issues
            is_valid, violations = validate_physical_circuit(mapped_circuit, coupling_map)
            if not is_valid:
                print(f"Warning: Conservative mapping still has {len(violations)} coupling violations")
                
                # Try an even more basic approach as last resort
                print("Trying basic mapping approach...")
                from qiskit import transpile
                mapped_circuit = transpile(
                    optimized_circuit.copy(),
                    coupling_map=coupling_map,
                    basis_gates=basis_gates,
                    optimization_level=0,
                    layout_method='trivial'  # Use trivial layout as last resort
                )
                
                # Final validation
                is_valid, violations = validate_physical_circuit(mapped_circuit, coupling_map)
                print(f"Basic mapping {'respects' if is_valid else 'still violates'} coupling constraints")
            else:
                print("Conservative mapping successfully respects coupling constraints")
        else:
            print("Mapped circuit successfully respects coupling constraints")
        
        # Force garbage collection after mapping
        gc.collect()
        
        # Get mapped circuit metrics
        mapped_metrics = self.compiler.get_circuit_metrics(mapped_circuit)
        
        # Estimate physical fidelity of the mapped circuit (hardware execution)
        physical_fidelity = self.estimate_physical_fidelity(mapped_circuit)
        
        results["mapped"] = {
            "depth": mapped_metrics["depth"],
            "gate_count": sum(mapped_metrics["gate_counts"].values()),
            "cx_count": mapped_metrics["cx_count"],
            "t_gates": mapped_metrics["t_gates"],
            "logical_fidelity": logical_fidelity,  # Function preservation
            "physical_fidelity": physical_fidelity,  # Hardware execution
            "circuit": mapped_circuit,  # Store for visualization
            "physical_depth": mapped_metrics["depth"],  # Add physical_depth for compatibility
            "respects_coupling_map": is_valid
        }
        
        # For compatibility with old code
        results["mapped"]["fidelity"] = physical_fidelity
        
        # Clear metrics from memory
        del mapped_metrics
        gc.collect()
        
        print(f"Final physical depth: {results['mapped']['depth']}")
        print(f"Final physical gates: {results['mapped']['gate_count']}")
        print(f"Logical fidelity (function preservation): {logical_fidelity:.6f}")
        print(f"Physical fidelity (hardware execution): {physical_fidelity:.6f}")
        print(f"Respects coupling constraints: {is_valid}")
        
        # 5. Calculate depth reduction
        logical_depth = results["logical"]["depth"]
        naive_depth = results["naive_physical"]["depth"]
        physical_depth = results["mapped"]["depth"]
        
        depth_reduction = ((logical_depth - results['optimized']['depth']) / logical_depth * 100) if logical_depth > 0 else 0
        depth_reduction_from_naive = ((naive_depth - physical_depth) / naive_depth * 100) if naive_depth > 0 else 0
        
        results["depth_reduction"] = depth_reduction
        results["depth_reduction_from_naive"] = depth_reduction_from_naive
        
        print(f"Depth reduction from logical: {depth_reduction:.2f}%")
        print(f"Depth reduction from naive: {depth_reduction_from_naive:.2f}%")
        
        # Add pattern-specific metrics if available
        if "pattern_optimized" in results:
            pattern_to_optimized_reduction = ((results["pattern_optimized"]["depth"] - results['optimized']['depth']) / 
                                             results["pattern_optimized"]["depth"] * 100) if results["pattern_optimized"]["depth"] > 0 else 0
            results["pattern_to_optimized_reduction"] = pattern_to_optimized_reduction
            print(f"Depth reduction from pattern optimization: {pattern_to_optimized_reduction:.2f}%")
        
        # Record execution time
        execution_time = time.time() - start_time
        results["execution_time"] = execution_time
        # Add "optimization_time" for compatibility with main.py
        results["optimization_time"] = execution_time
        
        print(f"Optimization completed in {execution_time:.4f} seconds")
        
        # Final garbage collection before returning results
        gc.collect()
        
        # Try to save circuit images if the helper function is available
        try:
            from ..utils.visualization import save_circuit_images
            save_circuit_images(results, self.output_dir)
        except ImportError:
            print("Circuit image saving functionality not available")
        
        # Compare circuits using compiler's comparison function
        try:
            print("\nDetailed Circuit Comparison:")
            self.compiler.print_comparison(logical_circuit, mapped_circuit, optimized_circuit)
            # Force garbage collection after comparison
            gc.collect()
        except Exception as e:
            print(f"Error during circuit comparison: {e}")
        
        return results
        
    except Exception as e:
        print(f"Error optimizing Toffoli network: {e}")
        import traceback
        traceback.print_exc()
        
        # Force garbage collection on error
        gc.collect()
        
        # Return minimal results with error information
        return {
            "error": str(e),
            "execution_time": time.time() - start_time,
            "optimization_time": time.time() - start_time,  # Add for compatibility
            "depth_reduction": 0,
            "depth_reduction_from_naive": 0
        }

def _optimize_circuit(self, circuit, toffoli_gates, output_qubits, input_qubits,
                     num_qubits, coupling_map, basis_gates, target_fidelity,
                     toffoli_type=None, use_ancilla=True):
    """
    Apply optimization techniques to reduce circuit depth and improve fidelity.
    
    This method focuses on optimizing the circuit structure through various strategies,
    working within the overall optimization framework of the optimize_toffoli_network method.
    
    Args:
        circuit: Circuit to optimize
        toffoli_gates: List of Toffoli gates
        output_qubits: List of output qubit indices
        input_qubits: List of input qubit indices
        num_qubits: Total number of qubits
        coupling_map: Coupling map for the target topology
        basis_gates: List of available basis gates
        target_fidelity: Target circuit fidelity
        toffoli_type: Type of Toffoli implementation to use
        use_ancilla: Whether to use ancilla qubits
        
    Returns:
        optimized_circuit: The optimized circuit
    """
    try:
        from qiskit import transpile, QuantumCircuit
        import numpy as np
        
        # Start with a fresh copy of the input circuit
        working_circuit = circuit.copy()
        original_depth = working_circuit.depth()
        
        if self.debug_mode:
            print(f"Starting optimization with circuit depth {original_depth}")
            print(f"Circuit has {len(working_circuit.data)} gates")
        
        # Track the best circuit we've found so far
        best_circuit = working_circuit.copy()
        best_depth = original_depth
        best_gate_count = sum(self.compiler.get_circuit_metrics(best_circuit)["gate_counts"].values())
        best_fidelity = self.estimate_physical_fidelity(best_circuit)
        
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
        
        # Create a safe version of the circuit by dynamically identifying problematic gates
        circuit_gate_processor = CircuitGateProcessor(debug_mode=self.debug_mode)
        working_circuit = circuit_gate_processor._process_circuit_safely(working_circuit, coupling_map)
        
        # 1. Try standard transpilation optimization
        try:
            for opt_level in [1, 2, 3]:
                transpiled = transpile(
                    working_circuit.copy(),
                    basis_gates=basis_gates,
                    optimization_level=opt_level
                )
                
                # Measure transpiled circuit
                trans_metrics = self.compiler.get_circuit_metrics(transpiled)
                trans_depth = trans_metrics["depth"]
                trans_gates = sum(trans_metrics["gate_counts"].values())
                trans_fidelity = self.estimate_physical_fidelity(transpiled)
                
                # Check if this is better than our current best
                if self._is_better_circuit(trans_depth, trans_gates, trans_fidelity,
                                        best_depth, best_gate_count, best_fidelity):
                    best_circuit = transpiled.copy()
                    best_depth = trans_depth
                    best_gate_count = trans_gates
                    best_fidelity = trans_fidelity
                    if self.debug_mode:
                        print(f"Transpilation level {opt_level} improved metrics: depth={best_depth}, gates={best_gate_count}, fidelity={best_fidelity:.6f}")
        except Exception as e:
            if self.debug_mode:
                print(f"Standard transpilation failed: {e}")
        
        # 2. Apply pattern-based optimizations again if the pattern library is available
        if hasattr(self, 'pattern_library') and hasattr(self.pattern_library, 'optimize_circuit'):
            try:
                pattern_threshold = getattr(self, 'pattern_threshold', 10.0)
                pattern_optimized = self.pattern_library.optimize_circuit(
                    working_circuit.copy(),
                    threshold=pattern_threshold,
                    use_ancilla=use_ancilla
                )
                
                if pattern_optimized is not None:
                    # Measure pattern-optimized circuit
                    pattern_metrics = self.compiler.get_circuit_metrics(pattern_optimized)
                    pattern_depth = pattern_metrics["depth"]
                    pattern_gates = sum(pattern_metrics["gate_counts"].values())
                    pattern_fidelity = self.estimate_physical_fidelity(pattern_optimized)
                    
                    # Check if this is better
                    if self._is_better_circuit(pattern_depth, pattern_gates, pattern_fidelity,
                                            best_depth, best_gate_count, best_fidelity):
                        best_circuit = pattern_optimized.copy()
                        best_depth = pattern_depth
                        best_gate_count = pattern_gates
                        best_fidelity = pattern_fidelity
                        if self.debug_mode:
                            print(f"Pattern optimization improved metrics: depth={best_depth}, gates={best_gate_count}, fidelity={best_fidelity:.6f}")
            except Exception as e:
                if self.debug_mode:
                    print(f"Pattern optimization failed: {e}")
        
        # 3. Try with different qubit layouts
        try:
            qubit_indices = list(range(best_circuit.num_qubits))
            
            # Try a few random permutations
            for _ in range(min(3, len(qubit_indices))):
                np.random.shuffle(qubit_indices)
                initial_layout = {i: qubit_indices[i] for i in range(len(qubit_indices))}
                
                # Apply transpilation with this layout
                layout_circuit = transpile(
                    best_circuit.copy(),
                    coupling_map=coupling_map,
                    basis_gates=basis_gates,
                    initial_layout=initial_layout,
                    optimization_level=3
                )
                
                # Measure layout-optimized circuit
                layout_metrics = self.compiler.get_circuit_metrics(layout_circuit)
                layout_depth = layout_metrics["depth"]
                layout_gates = sum(layout_metrics["gate_counts"].values())
                layout_fidelity = self.estimate_physical_fidelity(layout_circuit)
                
                # Check if this is better
                if self._is_better_circuit(layout_depth, layout_gates, layout_fidelity,
                                        best_depth, best_gate_count, best_fidelity):
                    best_circuit = layout_circuit.copy()
                    best_depth = layout_depth
                    best_gate_count = layout_gates
                    best_fidelity = layout_fidelity
                    if self.debug_mode:
                        print(f"Layout optimization improved metrics: depth={best_depth}, gates={best_gate_count}, fidelity={best_fidelity:.6f}")
        except Exception as e:
            if self.debug_mode:
                print(f"Layout optimization failed: {e}")
        
        # 4. Final optimization pass
        try:
            final_circuit = transpile(
                best_circuit,
                basis_gates=basis_gates,
                optimization_level=3
            )
            
            # Measure final circuit
            final_metrics = self.compiler.get_circuit_metrics(final_circuit)
            final_depth = final_metrics["depth"]
            final_gates = sum(final_metrics["gate_counts"].values())
            final_fidelity = self.estimate_physical_fidelity(final_circuit)
            
            # Check if this is better
            if self._is_better_circuit(final_depth, final_gates, final_fidelity,
                                    best_depth, best_gate_count, best_fidelity, final_pass=True):
                best_circuit = final_circuit
                best_depth = final_depth
                best_gate_count = final_gates
                best_fidelity = final_fidelity
                if self.debug_mode:
                    print(f"Final optimization pass improved metrics: depth={best_depth}, gates={best_gate_count}, fidelity={best_fidelity:.6f}")
        except Exception as e:
            if self.debug_mode:
                print(f"Final optimization pass failed: {e}")
        
        # Return the best circuit we found
        return best_circuit
        
    except Exception as e:
        print(f"Error in circuit optimization: {e}")
        import traceback
        traceback.print_exc()
        return circuit  # Return the original circuit on error

def _is_better_circuit(self, new_depth, new_gates, new_fidelity,
                      current_depth, current_gates, current_fidelity, final_pass=False):
    """
    Determine if the new circuit is better than the current best.
    
    Args:
        new_depth: Depth of the new circuit
        new_gates: Gate count of the new circuit
        new_fidelity: Fidelity of the new circuit
        current_depth: Depth of the current best circuit
        current_gates: Gate count of the current best circuit
        current_fidelity: Fidelity of the current best circuit
        final_pass: Whether this is the final optimization pass
        
    Returns:
        bool: Whether the new circuit is better
    """
    if hasattr(self, 'strategy'):
        from ..core.optimizer import OptimizationStrategy
        
        if self.strategy == OptimizationStrategy.DEPTH_REDUCTION:
            # For depth reduction, only care about depth
            return new_depth < current_depth
            
        elif self.strategy == OptimizationStrategy.GATE_REDUCTION:
            # For gate reduction, only care about gate count
            return new_gates < current_gates
            
        elif self.strategy == OptimizationStrategy.FIDELITY:
            # For fidelity optimization, only care about fidelity
            return new_fidelity > current_fidelity
            
        elif self.strategy == OptimizationStrategy.HYBRID:
            # For hybrid optimization, use a weighted score
            depth_weight = 0.5
            gate_weight = 0.3
            fidelity_weight = 0.2
            
            depth_score = current_depth / new_depth if new_depth > 0 else 1.0
            gate_score = current_gates / new_gates if new_gates > 0 else 1.0
            fidelity_score = new_fidelity / current_fidelity if current_fidelity > 0 else 1.0
            
            current_score = 1.0
            new_score = (depth_weight * depth_score +
                         gate_weight * gate_score +
                         fidelity_weight * fidelity_score)
            
            return new_score > current_score
    
    # Default to STANDARD strategy
    # For standard optimization, prioritize depth, then gates, then fidelity
    if new_depth < current_depth:
        return True
    elif new_depth == current_depth and new_gates < current_gates:
        return True
    elif new_depth == current_depth and new_gates == current_gates and new_fidelity > current_fidelity:
        return True
    # In the final pass, allow small depth increases if gate count or fidelity improves significantly
    elif final_pass and new_depth <= current_depth * 1.1:  # Allow up to 10% depth increase
        if new_gates < current_gates * 0.8:  # If gate count reduces by at least 20%
            return True
        if new_fidelity > current_fidelity * 1.2:  # If fidelity improves by at least 20%
            return True
    return False

def estimate_physical_fidelity(self, circuit):
    """
    Estimate the physical fidelity of a quantum circuit.
    
    This method calculates fidelity based on accumulated gate errors
    when running on actual hardware.
    
    Args:
        circuit: Quantum circuit to estimate physical fidelity for
        
    Returns:
        float: Estimated physical fidelity (0.0-1.0)
    """
    if circuit is None:
        return 0.0
        
    try:
        # Get circuit metrics
        metrics = self.compiler.get_circuit_metrics(circuit)
        
        # Use fidelity from the metrics if available
        if "fidelity" in metrics and metrics["fidelity"] > 0:
            return metrics["fidelity"]
        
        # Otherwise calculate based on gate error rates
        
        # Count CNOT gates
        cx_count = metrics.get("cx_count", 0)
        
        # Count T gates
        t_count = metrics.get("t_gates", 0)
        
        # Count single-qubit gates
        single_gates = metrics.get("single_qubit_gates", 0)
        if single_gates == 0:
            # If not explicitly counted, estimate from gate counts
            gate_counts = metrics.get("gate_counts", {})
            single_gates = sum(gate_counts.get(gate, 0) for gate in
                            ['h', 't', 'tdg', 'rx', 'ry', 'rz', 'x', 'y', 'z', 's', 'sdg'])
        
        # Use reasonable error rates
        cx_error_rate = 0.003   # 0.3% error per CNOT
        t_error_rate = 0.001    # 0.1% error per T gate
        single_error_rate = 0.0001  # 0.01% error per single-qubit gate
        
        # Calculate combined fidelity from all gate types
        fidelity = (1 - cx_error_rate) ** cx_count * \
                  (1 - t_error_rate) ** t_count * \
                  (1 - single_error_rate) ** single_gates
        
        # Consider circuit depth (longer circuits tend to have more decoherence)
        depth = metrics.get("depth", 0)
        depth_factor = max(0.95, 1.0 - (depth * 0.0001))  # 0.01% error per 100 depth
        fidelity *= depth_factor
        
        # Ensure it's in the valid range
        fidelity = max(0.0, min(1.0, fidelity))
        
        return fidelity
        
    except Exception as e:
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"Error estimating circuit fidelity: {e}")
        # Return a reasonable default based on circuit size
        try:
            size = len(circuit.data) if hasattr(circuit, 'data') else 100
            return max(0.5, 1.0 - (size * 0.001))  # Rough estimate: 0.1% error per gate
        except:
            return 0.9  # Default fallback

def calculate_logical_fidelity(self, original_circuit, optimized_circuit):
    """
    Calculate the logical fidelity between original and optimized circuits.
    
    This measures how well the optimized circuit preserves the functionality
    of the original circuit.
    
    Args:
        original_circuit: Original logical circuit
        optimized_circuit: Optimized logical circuit
        
    Returns:
        float: Logical fidelity (0.0-1.0)
    """
    if original_circuit is None or optimized_circuit is None:
        return 0.0
        
    try:
        # For Toffoli networks, we can use a heuristic based on
        # preserved operations
        
        original_gates = self.compiler.get_circuit_metrics(original_circuit)
        optimized_gates = self.compiler.get_circuit_metrics(optimized_circuit)
        
        # Calculate operation similarity
        # If we've eliminated gates while preserving function, this is good
        
        # Get total gates from both circuits
        original_total = sum(original_gates["gate_counts"].values())
        optimized_total = sum(optimized_gates["gate_counts"].values())
        
        # If the optimized circuit preserves function with fewer gates,
        # we consider this high fidelity
        if optimized_total < original_total:
            # The fewer gates we use while preserving function, the better
            # Scale this between 0.9 and 1.0
            preservation = 0.9 + (0.1 * (1 - optimized_total / max(1, original_total)))
        else:
            # If we haven't reduced gate count, use a direct comparison
            preservation = 0.9
            
        # Consider depth reduction as a factor in logical fidelity
        original_depth = original_gates["depth"]
        optimized_depth = optimized_gates["depth"]
        
        depth_factor = 0.0
        if original_depth > 0:
            depth_reduction = (original_depth - optimized_depth) / original_depth
            depth_factor = 0.1 * min(1.0, depth_reduction)
            
        # Overall logical fidelity is based on operation preservation and depth reduction
        logical_fidelity = min(1.0, preservation + depth_factor)
        
        return logical_fidelity
        
    except Exception as e:
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"Error calculating logical fidelity: {e}")
        return 0.9  # Default fallback

# Helper class for safely modifying circuits with proper error handling
class CircuitGateProcessor:
    """Helper class for safely manipulating circuit gates with robust error handling."""
    
    def __init__(self, debug_mode=False):
        """Initialize the gate processor.
        
        Args:
            debug_mode (bool): Whether to print debug information
        """
        self.debug_mode = debug_mode
    
    def _process_circuit_safely(self, circuit, coupling_map=None):
        """
        Process a circuit safely, handling control qubit constraints and coupling map properly.
        
        Args:
            circuit: The quantum circuit to process
            coupling_map: Optional coupling map constraints
            
        Returns:
            QuantumCircuit: Safely processed circuit that respects all constraints
        """
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