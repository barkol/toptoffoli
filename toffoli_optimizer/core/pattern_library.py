"""
Toffoli Pattern Library

This module provides a comprehensive library for identifying and optimizing patterns
of Toffoli gates, incorporating advanced optimization techniques.
"""

import itertools
import numpy as np
import gc
import os
import time
import traceback
import pickle
from contextlib import contextmanager
import signal
from enum import Enum, auto
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

# Define timeout exception and context manager
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

class ToffoliPatternGenerator:
    """Generator for Toffoli gate patterns and their simplified equivalents."""
    
    def __init__(self, num_qubits=3, max_toffolis=3, debug_mode=False):
        """
        Initialize the Toffoli pattern generator.
        
        Args:
            num_qubits: Number of qubits in the base register
            max_toffolis: Maximum number of Toffoli gates in a sequence
            debug_mode: Whether to print debug information
        """
        self.num_qubits = num_qubits
        self.max_toffolis = max_toffolis
        self.debug_mode = debug_mode
        self.patterns = {}
        self.pattern_operators = {}
        self.simplified_circuits = {}
        self.pattern_matchers = {}
        self.simplification_metrics = {}
        
        # Check optimizer availability
        self.optimizer_available = False
        try:
            from ..core.optimizer import ToffoliDepthOptimizer
            from ..core.compiler import ToffoliCompiler, ToffoliType
            self.compiler = ToffoliCompiler(debug_mode=debug_mode)
            self.optimizer = ToffoliDepthOptimizer(
                target_fidelity=0.95,
                max_passes=2,
                debug_mode=debug_mode,
                pass_timeout_seconds=30  # Limit each pass to 30 seconds
            )
            self.optimizer_available = True
        except ImportError:
            self.compiler = None
            self.optimizer = None
    
    def generate_all_patterns(self):
        """
        Generate all possible patterns of Toffoli gates on the register.
        """
        print(f"Generating all patterns of up to {self.max_toffolis} Toffoli gates on {self.num_qubits} qubits...")
        
        # Generate all possible Toffoli gate configurations (control1, control2, target)
        toffoli_configs = []
        for control1 in range(self.num_qubits):
            for control2 in range(self.num_qubits):
                for target in range(self.num_qubits):
                    # Skip invalid configurations (controls must be different from each other and target)
                    if control1 != control2 and control1 != target and control2 != target:
                        toffoli_configs.append((control1, control2, target))
        
        # Generate patterns with 1, 2, or 3 Toffoli gates
        pattern_id = 0
        for num_gates in range(1, self.max_toffolis + 1):
            # Generate all possible sequences of Toffoli gates
            for toffoli_sequence in itertools.product(toffoli_configs, repeat=num_gates):
                pattern_id += 1
                pattern_key = f"pattern_{pattern_id}"
                self.patterns[pattern_key] = toffoli_sequence
                
                # Create a circuit for this pattern
                circuit = QuantumCircuit(self.num_qubits)
                for control1, control2, target in toffoli_sequence:
                    circuit.ccx(control1, control2, target)
                
                # Store the circuit and its unitary operator
                self._compute_operator(pattern_key, circuit)
                
                # Print progress periodically
                if pattern_id % 100 == 0:
                    print(f"Generated {pattern_id} patterns...")
                    # Force garbage collection to manage memory
                    gc.collect()
        
        print(f"Generated {pattern_id} unique Toffoli gate patterns.")
        return self.patterns
    
    def _compute_operator(self, pattern_key, circuit):
        """Compute and store the unitary operator for a pattern."""
        try:
            # Calculate the unitary operator for this circuit
            operator = Operator(circuit)
            self.pattern_operators[pattern_key] = operator
        except Exception as e:
            if self.debug_mode:
                print(f"Error computing operator for {pattern_key}: {e}")

    def simplify_patterns_with_optimizer(self, pattern_keys=None):
        """
        Simplify patterns using ToffoliDepthOptimizer techniques.
        
        Args:
            pattern_keys: List of pattern keys to simplify (None for all)
            
        Returns:
            dict: Simplified circuits for the specified patterns
        """
        if not self.optimizer_available:
            print("ToffoliDepthOptimizer not available. Using transpiler simplification instead.")
            return self.simplify_patterns()
        
        print("Simplifying patterns using ToffoliDepthOptimizer techniques...")
        
        # If no keys specified, use all patterns
        if pattern_keys is None:
            pattern_keys = list(self.patterns.keys())
        
        # Process each pattern
        for pattern_id in pattern_keys:
            if pattern_id not in self.patterns:
                continue
                
            pattern = self.patterns[pattern_id]
            
            # Skip if already simplified
            if pattern_id in self.simplified_circuits:
                continue
            
            # Convert pattern to Toffoli gates format for optimizer
            toffoli_gates = []
            for control1, control2, target in pattern:
                toffoli_gates.append(([control1, control2], target))
            
            # Define arbitrary input/output qubits
            input_qubits = list(range(self.num_qubits))
            output_qubits = list(range(self.num_qubits))
            
            try:
                # Use ToffoliDepthOptimizer with a timeout
                with time_limit(30):  # 30 second timeout per pattern
                    optimizer_results = self.optimizer.optimize_toffoli_network(
                        toffoli_gates,
                        output_qubits,
                        input_qubits,
                        self.num_qubits,
                        topology='linear',  # Simple topology for patterns
                        use_ancilla=False   # No ancilla qubits for basic patterns
                    )
                
                # Extract the optimized circuit
                if optimizer_results and "optimized" in optimizer_results:
                    optimized_circuit = optimizer_results["optimized"]["circuit"]
                    self.simplified_circuits[pattern_id] = optimized_circuit
                    
                    if self.debug_mode:
                        original_depth = self.num_qubits * len(pattern)  # Rough estimate
                        optimized_depth = optimized_circuit.depth()
                        print(f"Optimized {pattern_id}: {original_depth} -> {optimized_depth} depth")
                else:
                    # Fallback to transpiler if optimizer fails
                    self._simplify_with_transpiler(pattern_id)
            
            except TimeoutException:
                print(f"Timeout while optimizing {pattern_id}. Using transpiler fallback.")
                self._simplify_with_transpiler(pattern_id)
                
            except Exception as e:
                if self.debug_mode:
                    print(f"Error optimizing {pattern_id}: {e}")
                    traceback.print_exc()
                self._simplify_with_transpiler(pattern_id)
            
            # Print progress periodically
            if int(pattern_id.split('_')[1]) % 50 == 0:
                print(f"Simplified {pattern_id}...")
                # Force garbage collection
                gc.collect()
        
        print(f"Simplified {len(pattern_keys)} patterns.")
        return {k: self.simplified_circuits[k] for k in pattern_keys if k in self.simplified_circuits}
    
    def _simplify_with_transpiler(self, pattern_id):
        """Fallback method to simplify a pattern using Qiskit's transpiler."""
        pattern = self.patterns[pattern_id]
        
        # Create circuit from pattern
        circuit = QuantumCircuit(self.num_qubits)
        for control1, control2, target in pattern:
            circuit.ccx(control1, control2, target)
        
        # Simplify using transpiler
        basis_gates = ['id', 'u1', 'u2', 'u3', 'cx', 'x', 'y', 'z', 'h', 's', 't']
        simplified = transpile(
            circuit,
            basis_gates=basis_gates,
            optimization_level=3
        )
        
        # Store the simplified circuit
        self.simplified_circuits[pattern_id] = simplified
    
    def simplify_patterns(self, optimization_level=3):
        """
        Simplify all patterns using Qiskit's transpiler (fallback method).
        
        Args:
            optimization_level: Optimization level for transpilation (0-3)
        """
        print("Simplifying patterns using Qiskit's transpiler...")
        
        basis_gates = ['id', 'u1', 'u2', 'u3', 'cx', 'x', 'y', 'z', 'h', 's', 't']
        
        # Process each pattern
        for pattern_id, pattern in self.patterns.items():
            # Skip if already simplified
            if pattern_id in self.simplified_circuits:
                continue
                
            # Create circuit from pattern
            circuit = QuantumCircuit(self.num_qubits)
            for control1, control2, target in pattern:
                circuit.ccx(control1, control2, target)
            
            # Simplify using transpiler
            simplified = transpile(
                circuit,
                basis_gates=basis_gates,
                optimization_level=optimization_level
            )
            
            # Store the simplified circuit
            self.simplified_circuits[pattern_id] = simplified
            
            # Print progress periodically
            if int(pattern_id.split('_')[1]) % 100 == 0:
                print(f"Simplified {pattern_id}...")
                # Force garbage collection
                gc.collect()
        
        print(f"Simplified {len(self.patterns)} patterns.")
        return self.simplified_circuits
    
    def identify_equivalent_patterns(self, tolerance=1e-10):
        """
        Identify patterns that are functionally equivalent.
        
        Args:
            tolerance: Numerical tolerance for comparing operators
        
        Returns:
            dict: Dictionary mapping pattern groups to their equivalent circuits
        """
        print("Identifying equivalent patterns...")
        
        # Group patterns by their operator
        equivalent_groups = {}
        
        # Create a list of pattern keys
        pattern_keys = list(self.pattern_operators.keys())
        
        # Compare each pattern with all others
        for i, key1 in enumerate(pattern_keys):
            # Skip if already in a group
            if any(key1 in group for group in equivalent_groups.values()):
                continue
            
            # Create a new group with this pattern
            group = [key1]
            op1 = self.pattern_operators[key1]
            
            # Compare with all other patterns
            for j in range(i + 1, len(pattern_keys)):
                key2 = pattern_keys[j]
                op2 = self.pattern_operators[key2]
                
                # Check if operators are equivalent
                try:
                    if np.allclose(op1.data, op2.data, atol=tolerance):
                        group.append(key2)
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error comparing operators {key1} and {key2}: {e}")
            
            # Add group if it contains more than one pattern
            if len(group) > 1:
                group_key = f"group_{len(equivalent_groups) + 1}"
                equivalent_groups[group_key] = group
            
            # Print progress periodically
            if i % 100 == 0:
                print(f"Processed {i}/{len(pattern_keys)} patterns...")
                # Force garbage collection
                gc.collect()
        
        print(f"Found {len(equivalent_groups)} groups of equivalent patterns.")
        return equivalent_groups
    
    def analyze_simplifications(self):
        """Analyze the simplifications and identify the best candidates."""
        for pattern_id in self.patterns.keys():
            # Skip if already analyzed
            if pattern_id in self.simplification_metrics:
                continue
                
            # Get the original and simplified circuits
            original_circuit = None
            simplified_circuit = None
            
            # Create original circuit
            try:
                original_circuit = QuantumCircuit(self.num_qubits)
                for control1, control2, target in self.patterns[pattern_id]["sequence"]:
                    original_circuit.ccx(control1, control2, target)
            except:
                # If we can't access the sequence directly, try recreating it
                try:
                    pattern = self.patterns[pattern_id]
                    original_circuit = QuantumCircuit(self.num_qubits)
                    for control1, control2, target in pattern:
                        original_circuit.ccx(control1, control2, target)
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error creating original circuit for {pattern_id}: {e}")
                    continue
            
            # Get simplified circuit
            simplified_circuit = self.simplified_circuits.get(pattern_id)
            
            if original_circuit is None or simplified_circuit is None:
                continue
            
            # Calculate metrics
            try:
                original_depth = original_circuit.depth()
                original_size = len(original_circuit.data)
                original_cx_count = sum(1 for inst in original_circuit.data if inst.operation.name == 'cx')
                
                simplified_depth = simplified_circuit.depth()
                simplified_size = len(simplified_circuit.data)
                simplified_cx_count = sum(1 for inst in simplified_circuit.data if inst.operation.name == 'cx')
                
                # Calculate improvement percentages
                if original_depth > 0:
                    depth_reduction = ((original_depth - simplified_depth) / original_depth) * 100
                else:
                    depth_reduction = 0
                
                if original_size > 0:
                    size_reduction = ((original_size - simplified_size) / original_size) * 100
                else:
                    size_reduction = 0
                    
                if original_cx_count > 0:
                    cx_reduction = ((original_cx_count - simplified_cx_count) / original_cx_count) * 100
                else:
                    cx_reduction = 0
                
                # Store metrics
                self.simplification_metrics[pattern_id] = {
                    "original_depth": original_depth,
                    "original_size": original_size,
                    "original_cx_count": original_cx_count,
                    "simplified_depth": simplified_depth,
                    "simplified_size": simplified_size,
                    "simplified_cx_count": simplified_cx_count,
                    "depth_reduction": depth_reduction,
                    "size_reduction": size_reduction,
                    "cx_reduction": cx_reduction
                }
            except Exception as e:
                if self.debug_mode:
                    print(f"Error analyzing pattern {pattern_id}: {e}")
        
        print(f"Analyzed {len(self.simplification_metrics)} pattern simplifications")
    
    def get_top_simplifications(self, n=5):
        """Get the top N patterns with the best simplification potential."""
        # Make sure we've analyzed all patterns
        if len(self.simplification_metrics) < len(self.patterns):
            self.analyze_simplifications()
        
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
            pattern_sequence = None
            try:
                pattern_sequence = self.patterns[pattern_id]["sequence"]
            except:
                pattern_sequence = self.patterns[pattern_id]
            
            print(f"{i+1}. Pattern ID: {pattern_id}")
            print(f"   Original sequence: {pattern_sequence}")
            print(f"   Original depth: {metrics['original_depth']}")
            print(f"   Simplified depth: {metrics['simplified_depth']}")
            print(f"   Depth reduction: {metrics['depth_reduction']:.2f}%")
            print(f"   CX count reduction: {metrics['original_cx_count']} → {metrics['simplified_cx_count']}")
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
        
        # First extract all Toffoli gates from the circuit
        toffoli_gates = []
        for i, instruction in enumerate(circuit.data):
            if instruction.operation.name == 'ccx':
                qubits = [q.index if hasattr(q, 'index') else q._index for q in instruction.qubits]
                control1, control2, target = qubits
                toffoli_gates.append((i, (control1, control2, target)))
        
        # Check against each pattern
        for pattern_id, pattern in self.patterns.items():
            pattern_length = len(pattern)
            
            # Skip if circuit doesn't have enough Toffoli gates
            if len(toffoli_gates) < pattern_length:
                continue
            
            # Scan through Toffoli gates looking for pattern matches
            for i in range(len(toffoli_gates) - pattern_length + 1):
                # Get the sequence of gates to compare
                gate_sequence = [gate for _, gate in toffoli_gates[i:i+pattern_length]]
                
                # Check if this matches the pattern
                try:
                    # For different pattern storage formats
                    if isinstance(pattern, dict) and "sequence" in pattern:
                        pattern_sequence = pattern["sequence"]
                    else:
                        pattern_sequence = pattern
                    
                    # Compare gate sequences
                    if gate_sequence == pattern_sequence:
                        found_patterns.append((pattern_id, toffoli_gates[i][0]))
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error comparing pattern {pattern_id}: {e}")
        
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
        # Make sure we've analyzed all patterns
        if len(self.simplification_metrics) < len(self.patterns):
            self.analyze_simplifications()
        
        # Find all patterns in the circuit
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
            # Skip if we don't have metrics for this pattern
            if pattern_id not in self.simplification_metrics:
                continue
                
            metrics = self.simplification_metrics[pattern_id]
            
            if metrics["depth_reduction"] >= threshold:
                # Get simplified circuit for this pattern
                if pattern_id not in self.simplified_circuits:
                    continue
                    
                simplified = self.simplified_circuits[pattern_id]
                
                # Get pattern length
                pattern_length = 0
                try:
                    if isinstance(self.patterns[pattern_id], dict) and "sequence" in self.patterns[pattern_id]:
                        pattern_length = len(self.patterns[pattern_id]["sequence"])
                    else:
                        pattern_length = len(self.patterns[pattern_id])
                except:
                    continue
                
                # Find the Toffoli gates to replace
                toffoli_indices = []
                toffoli_qubits = []
                
                # Find all Toffoli gates in sequence starting from start_idx
                gate_count = 0
                for i, instruction in enumerate(optimized_circuit.data[start_idx:]):
                    if instruction.operation.name == 'ccx':
                        toffoli_indices.append(start_idx + i)
                        qubits = [q.index if hasattr(q, 'index') else q._index for q in instruction.qubits]
                        toffoli_qubits.append(qubits)
                        gate_count += 1
                    
                    if gate_count >= pattern_length:
                        break
                
                # Check if we found enough gates
                if len(toffoli_indices) < pattern_length:
                    continue
                
                # Get unique qubits used in this pattern
                all_qubits = set()
                for control1, control2, target in toffoli_qubits[:pattern_length]:
                    all_qubits.add(control1)
                    all_qubits.add(control2)
                    all_qubits.add(target)
                all_qubits = sorted(list(all_qubits))
                
                # Create qubit mapping between pattern and actual circuit
                qubit_map = {}
                for i, q in enumerate(range(simplified.num_qubits)):
                    if i < len(all_qubits):
                        qubit_map[q] = all_qubits[i]
                
                # Remove the pattern gates from the circuit
                for idx in sorted(toffoli_indices[:pattern_length], reverse=True):
                    optimized_circuit.data.pop(idx)
                
                # Create a new subcircuit with the simplified pattern
                subcircuit = QuantumCircuit(optimized_circuit.num_qubits)
                
                # Add the simplified gates with mapped qubits
                for instruction in simplified.data:
                    try:
                        # Get operation and qubits
                        operation = instruction.operation
                        qubits = [qubit_map.get(q.index, q.index) if hasattr(q, 'index') else 
                                  qubit_map.get(q._index, q._index) for q in instruction.qubits]
                        
                        # Add to subcircuit
                        subcircuit.append(operation, qubits)
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Error adding simplified instruction: {e}")
                
                # Insert the subcircuit at the position of the first gate
                for i, instruction in enumerate(subcircuit.data):
                    optimized_circuit.data.insert(start_idx + i, instruction)
                
                replacements += 1
        
        if replacements > 0:
            print(f"Applied {replacements} pattern replacements")
            
            # Final transpilation to clean up
            try:
                from qiskit import transpile
                optimized_circuit = transpile(optimized_circuit, optimization_level=3)
            except:
                pass
        
        return optimized_circuit
    
    def build_pattern_library(self, save_file='toffoli_pattern_library.pkl'):
        """
        Build and save a complete pattern library.
        
        Args:
            save_file: Filename to save the library
        
        Returns:
            dict: The complete pattern library
        """
        # Generate all patterns if not already done
        if not self.patterns:
            self.generate_all_patterns()
        
        # Simplify all patterns using the best available method
        if self.optimizer_available:
            print("Using ToffoliDepthOptimizer for pattern simplification...")
            self.simplify_patterns_with_optimizer()
        else:
            print("Using Qiskit transpiler for pattern simplification...")
            self.simplify_patterns()
        
        # Analyze the simplifications
        self.analyze_simplifications()
        
        # Create the complete library
        library = {
            'patterns': self.patterns,
            'simplified_circuits': self.simplified_circuits,
            'simplification_metrics': self.simplification_metrics,
            'num_qubits': self.num_qubits,
            'max_toffolis': self.max_toffolis
        }
        
        # Save the library to file
        with open(save_file, 'wb') as f:
            pickle.dump(library, f)
        
        print(f"Pattern library saved to {save_file}")
        return library
    
    @staticmethod
    def load_pattern_library(save_file='toffoli_pattern_library.pkl'):
        """
        Load a pattern library from file.
        
        Args:
            save_file: Filename to load the library from
        
        Returns:
            dict: The loaded pattern library
        """
        if not os.path.exists(save_file):
            print(f"Pattern library file {save_file} not found")
            return None
        
        with open(save_file, 'rb') as f:
            library = pickle.load(f)
        
        print(f"Loaded pattern library from {save_file}")
        return library


class ToffoliPatternOptimizer:
    """Optimizer that uses the pattern library to optimize Toffoli circuits."""
    
    def __init__(self, library_file='toffoli_pattern_library.pkl', debug_mode=False):
        """
        Initialize the optimizer with a pattern library.
        
        Args:
            library_file: File containing the pattern library
            debug_mode: Whether to print debug information
        """
        self.debug_mode = debug_mode
        
        # Load the pattern library
        self.library = ToffoliPatternGenerator.load_pattern_library(library_file)
        
        if self.library is None:
            print("Warning: Pattern library not found. Generating a new library...")
            generator = ToffoliPatternGenerator(debug_mode=debug_mode)
            self.library = generator.build_pattern_library(library_file)
        
        # Initialize pattern generator for finding patterns
        self.pattern_generator = ToffoliPatternGenerator(debug_mode=debug_mode)
        self.pattern_generator.patterns = self.library['patterns']
        self.pattern_generator.simplified_circuits = self.library['simplified_circuits']
        self.pattern_generator.simplification_metrics = self.library['simplification_metrics']
    
    def optimize_circuit(self, circuit, threshold=10.0, use_ancilla=True):
        """
        Optimize a circuit using pattern-based replacements.
        
        Args:
            circuit: The circuit to optimize
            threshold: Minimum depth reduction percentage to apply a pattern
            use_ancilla: Whether to allow solutions with ancilla qubits
            
        Returns:
            QuantumCircuit: The optimized circuit
        """
        return self.pattern_generator.optimize_circuit(circuit, threshold, use_ancilla)
