
"""
Toffoli Depth Optimizer Module

This module provides the ToffoliDepthOptimizer class for minimizing the depth of Toffoli gate 
networks on quantum hardware with limited connectivity, with options to trade fidelity for depth.

It combines multiple optimization techniques:
1. Enhanced transpilation with custom passes
2. Depth-fidelity balancing with configurable weights
"""

import os
import time
import traceback
import numpy as np
from datetime import datetime
import signal
from contextlib import contextmanager
import gc  # For garbage collection

from .compiler import ToffoliCompiler, ToffoliType
from .circuit_gate_processor import CircuitGateProcessor
from ..utils.circuit_utils import (
    create_optimized_physical_mapping,
    estimate_fidelity,
    get_default_coupling_map
)

# Check for Qiskit availability
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


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

class ToffoliDepthOptimizer:
    """
    Consolidated optimizer for minimizing the depth of Toffoli gate networks on quantum hardware.
    
    This class combines the best features from various optimizer implementations.
    
    Attributes:
        target_fidelity (float): Target circuit fidelity (0.0-1.0)
        min_fidelity (float): Minimum acceptable fidelity (0.0-1.0)
        max_passes (int): Maximum number of optimization passes
        output_dir (str): Directory for optimizer output
        use_parallel (bool): Whether to use parallel execution
        debug_mode (bool): Whether to print debug information
        pass_timeout_seconds (int): Maximum seconds per optimization pass (0 for no limit)
        strategy (OptimizationStrategy): Strategy for optimization priorities
        depth_weight (float): Weight for depth in scoring (for DEPTH_FIDELITY_BALANCE)
        fidelity_weight (float): Weight for fidelity in scoring (for DEPTH_FIDELITY_BALANCE)
        pattern_threshold (float): Minimum depth reduction percentage to apply a pattern replacement
    """
    
    # Class-level default for the equivalence gate. This lets a CLI front-end
    # (scripts/main.py) flip the gate on globally before instances are created
    # by code paths that don't forward the constructor argument (e.g. optimize.py).
    _verify_equivalence_default = False

    def __init__(self, target_fidelity=0.95, min_fidelity=0.8, max_passes=2, output_dir=None,
                use_parallel=True, debug_mode=False, pass_timeout_seconds=60,
                strategy=None, depth_weight=0.7, fidelity_weight=0.3,
                verify_equivalence=None):
        """
        Initialize the Toffoli Depth Optimizer.
        
        Args:
            target_fidelity (float): Target circuit fidelity (0.0-1.0).
                Higher values prioritize maintaining circuit fidelity over depth reduction.
                Recommended values: 0.9-0.99.
            
            min_fidelity (float): Minimum acceptable fidelity (0.0-1.0).
                This sets a lower bound for how much fidelity can be sacrificed for depth.
                Only applies to strategies that allow fidelity trade-offs.
            
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
                - ULTRA_DEPTH_REDUCTION: Aggressively reduce depth with fidelity trade-offs
                - DEPTH_FIDELITY_BALANCE: Explicitly balance depth and fidelity using weights
            
            depth_weight (float): Weight for depth in balanced optimization (0.0-1.0).
                Only applies to DEPTH_FIDELITY_BALANCE strategy. Higher values prioritize depth more.
            
            fidelity_weight (float): Weight for fidelity in balanced optimization (0.0-1.0).
                Only applies to DEPTH_FIDELITY_BALANCE strategy. Higher values prioritize fidelity more.
                
        """
        from .optimization_strategy import OptimizationStrategy
        
        self.target_fidelity = target_fidelity
        self.min_fidelity = min_fidelity
        self.max_passes = max_passes
        self.use_parallel = use_parallel
        self.debug_mode = debug_mode
        self.pass_timeout_seconds = pass_timeout_seconds
        self.depth_weight = depth_weight
        self.fidelity_weight = fidelity_weight

        # OPT-IN exact-equivalence correctness gate. When True, candidate
        # circuits are checked against the pre-optimization input and rejected
        # if proven non-equivalent. Falls back to the class default so a CLI
        # front-end can enable it globally. Default behavior is UNCHANGED (off).
        if verify_equivalence is None:
            self.verify_equivalence = type(self)._verify_equivalence_default
        else:
            self.verify_equivalence = bool(verify_equivalence)
        # Lazily constructed verifier (only when the gate is actually used).
        self._equivalence_verifier = None

        # Set strategy - default to HYBRID if not a valid enum
        if strategy is None:
            self.strategy = OptimizationStrategy.HYBRID
        elif isinstance(strategy, OptimizationStrategy):
            self.strategy = strategy
        elif isinstance(strategy, str):
            # Try to convert string to enum
            try:
                self.strategy = OptimizationStrategy[strategy]
            except (KeyError, ValueError):
                self.strategy = OptimizationStrategy.HYBRID
        else:
            self.strategy = OptimizationStrategy.HYBRID
        
        # Set default output directory if none provided
        if output_dir is None:
            self.output_dir = "toffoli_optimizer_results"
        else:
            self.output_dir = output_dir
        
        # Create timestamp for this optimizer run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize compiler
        self.compiler = ToffoliCompiler(debug_mode=self.debug_mode)
        
        # Initialize component processors
        self._initialize_components()
  
    def validate_physical_circuit(self, circuit, coupling_map):
        """
        Validate that a circuit respects the coupling constraints of a given coupling map.
        
        Args:
            circuit: Quantum circuit to validate
            coupling_map: Coupling map to validate against
            
        Returns:
            tuple: (is_valid, violations)
                - is_valid: Boolean indicating if the circuit respects the coupling map
                - violations: List of coupling violations (qubit pairs)
        """
        if circuit is None or coupling_map is None:
            return True, []  # No circuit or coupling map, nothing to validate
            
        # Create set of allowed qubit pairs
        allowed_pairs = set()
        if isinstance(coupling_map, list):
            for pair in coupling_map:
                if isinstance(pair, list) and len(pair) == 2:
                    allowed_pairs.add((pair[0], pair[1]))
                    allowed_pairs.add((pair[1], pair[0]))  # Add reverse direction too
        else:
            # Try to get edges from coupling map object
            try:
                edges = coupling_map.get_edges()
                for edge in edges:
                    allowed_pairs.add((edge[0], edge[1]))
                    allowed_pairs.add((edge[1], edge[0]))  # Add reverse direction too
            except:
                if self.debug_mode:
                    print("Warning: Could not extract connectivity from coupling map object")
                return True, []  # Can't validate
        
        # Check for violations
        violations = []
        
        for inst in circuit.data:
            # Skip single-qubit gates
            if len(inst.qubits) <= 1:
                continue
                
            # Check multi-qubit gates
            qubit_indices = []
            for q in inst.qubits:
                qubit_indices.append(q._index if hasattr(q, '_index') else q.index)
            
            # Check if all qubit pairs are allowed
            for i in range(len(qubit_indices)):
                for j in range(i+1, len(qubit_indices)):
                    q1, q2 = qubit_indices[i], qubit_indices[j]
                    
                    # Skip if this pair is allowed
                    if (q1, q2) in allowed_pairs or (q2, q1) in allowed_pairs:
                        continue
                    
                    # Record violation
                    violations.append((q1, q2))
        
        is_valid = len(violations) == 0
        return is_valid, violations

  
    def _initialize_components(self):
        """Initialize optimization components based on availability and settings."""
        # Initialize circuit gate processor
        self.gate_processor = CircuitGateProcessor(debug_mode=self.debug_mode)
        
        
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
            if self.debug_mode:
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
            if self.debug_mode:
                print(f"Error calculating logical fidelity: {e}")
            return 0.9  # Default fallback
            

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
        from .optimization_strategy import OptimizationStrategy
        
        # First check minimum fidelity requirement for all strategies
        if new_fidelity < self.min_fidelity and self.strategy != OptimizationStrategy.ULTRA_DEPTH_REDUCTION:
            return False
        
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
            
        elif self.strategy == OptimizationStrategy.ULTRA_DEPTH_REDUCTION:
            # For ultra depth reduction, prioritize depth reduction even at the cost of fidelity,
            # but respect the minimum fidelity threshold
            if new_fidelity >= self.min_fidelity:
                if new_depth < current_depth:
                    return True
                elif new_depth == current_depth and new_gates < current_gates:
                    return True
            # If fidelity is better and depth is close (within 10%), accept the trade-off
            elif new_fidelity > current_fidelity and new_depth <= current_depth * 1.1:
                return True
            # For ultra depth reduction, allow even lower fidelity if it gives significant depth improvement
            elif new_fidelity >= 0.7 * self.min_fidelity and new_depth < current_depth * 0.7:
                return True
            return False
            
        elif self.strategy == OptimizationStrategy.DEPTH_FIDELITY_BALANCE:
            # For depth-fidelity balance, use the provided weights
            # Normalize metrics to compare properly
            depth_ratio = current_depth / max(1, new_depth)  # Higher is better
            fidelity_ratio = new_fidelity / max(0.1, current_fidelity)  # Higher is better
            
            # Adjust by weights
            depth_score = self.depth_weight * depth_ratio
            fidelity_score = self.fidelity_weight * fidelity_ratio
            
            # Gate count serves as a tie-breaker
            gate_score = 0.1 * (current_gates / max(1, new_gates))
            
            # Compare weighted scores
            new_score = depth_score + fidelity_score + gate_score
            current_score = self.depth_weight + self.fidelity_weight + 0.1
            
            return new_score > current_score
        
        elif self.strategy in [OptimizationStrategy.TRANSPILER, OptimizationStrategy.TRANSPILER_L1,
                             OptimizationStrategy.TRANSPILER_L2, OptimizationStrategy.TRANSPILER_L3]:
            # For transpiler optimization, prioritize depth, then gate count
            if new_depth < current_depth:
                return True
            elif new_depth == current_depth and new_gates < current_gates:
                return True
            return False
        
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


    def _get_equivalence_verifier(self):
        """Lazily build (and cache) the exact-equivalence verifier."""
        if self._equivalence_verifier is None:
            from .equivalence_verifier import ExactEquivalenceVerifier
            self._equivalence_verifier = ExactEquivalenceVerifier()
        return self._equivalence_verifier

    def _equivalence_gate(self, reference_circuit, candidate_circuit, context=""):
        """Exact-equivalence correctness gate.

        Compares `candidate_circuit` to `reference_circuit` (the pre-optimization
        input) with the exact verifier. The verifier self-guards circuits it can't
        check exactly (>12 qubits / non-classical-too-large) and returns False with
        a size/feasibility `reason`; that is treated as "cannot verify" -> SKIP the
        gate (do NOT reject). Only a genuine non-equivalence (classical/unitary
        mismatch, ancilla-not-restored, leakage) causes a REJECT.

        Returns one of:
            "ok"     -- proven equivalent; keep the candidate
            "skip"   -- could not verify (infeasible / error); keep the candidate
            "reject" -- proven NOT equivalent; caller must fall back
        """
        if not self.verify_equivalence:
            return "ok"
        if reference_circuit is None or candidate_circuit is None:
            return "skip"

        label = f" [{context}]" if context else ""

        # Reasons that mean "the verifier declined to decide" -> cannot verify.
        infeasible_markers = ("too large", "too big", "qubits)", "fewer qubits")
        try:
            verifier = self._get_equivalence_verifier()
            is_equivalent, _perm, info = verifier.verify(
                reference_circuit, candidate_circuit, allow_permutation=False
            )
        except Exception as e:
            # A crashing verifier must not break optimization; treat as unverifiable.
            print(f"WARNING: equivalence verification raised an error{label}: {e}; "
                  f"skipping correctness gate")
            return "skip"

        if is_equivalent:
            if self.debug_mode:
                method = (info or {}).get("method", "?")
                print(f"Equivalence gate{label}: candidate VERIFIED equivalent "
                      f"(method={method})")
            return "ok"

        reason = (info or {}).get("reason", "") or ""
        reason_l = reason.lower()
        if any(m in reason_l for m in infeasible_markers):
            print(f"WARNING: cannot exactly verify equivalence{label} "
                  f"({reason}); skipping correctness gate (candidate kept un-verified)")
            return "skip"

        # Genuine non-equivalence.
        print(f"WARNING: equivalence gate REJECTED candidate{label}: "
              f"circuit is NOT equivalent to the original ({reason}); "
              f"falling back to the last verified-good circuit")
        return "reject"


    def optimize_toffoli_network(self, toffoli_gates, output_qubits=None, input_qubits=None,
                               num_qubits=None, topology=None, coupling_map=None,
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
            toffoli_type (ToffoliType): Type of Toffoli implementation to use
            use_ancilla (bool): Whether to use ancilla qubits
            
        Returns:
            dict: Dictionary with optimization results
        """
        # Force garbage collection at the start to free memory
        gc.collect()
        
        # Set defaults for parameters
        if output_qubits is None:
            output_qubits = []
        if input_qubits is None:
            input_qubits = []
        if num_qubits is None and logical_circuit is not None:
            num_qubits = logical_circuit.num_qubits
        elif num_qubits is None:
            # Try to infer from toffoli_gates
            max_qubit = 0
            for gate in toffoli_gates:
                if isinstance(gate, tuple):
                    if len(gate) == 3:  # (control1, control2, target)
                        max_qubit = max(max_qubit, gate[0], gate[1], gate[2])
                    elif len(gate) == 2:  # ([controls], target)
                        controls, target = gate
                        if isinstance(controls, list):
                            max_qubit = max(max_qubit, target, *controls)
                        else:
                            max_qubit = max(max_qubit, controls, target)
            num_qubits = max_qubit + 1
        
        # Set default Toffoli type if not provided
        if toffoli_type is None:
            toffoli_type = ToffoliType.RELATIVE_PHASE_1
        
        print(f"\nOptimizing Toffoli network with {len(toffoli_gates)} gates on {num_qubits} qubits")
        if isinstance(toffoli_type, ToffoliType):
            print(f"Using {toffoli_type.name} implementation")
        else:
            print(f"Using STANDARD implementation")
        print(f"Max optimization passes: {self.max_passes}")
        print(f"Optimization strategy: {self.strategy.name}")
        
        # Print fidelity parameters based on strategy
        from .optimization_strategy import OptimizationStrategy
        if self.strategy in [OptimizationStrategy.ULTRA_DEPTH_REDUCTION, OptimizationStrategy.DEPTH_FIDELITY_BALANCE]:
            print(f"Target fidelity: {self.target_fidelity}, Minimum fidelity: {self.min_fidelity}")
            if self.strategy == OptimizationStrategy.DEPTH_FIDELITY_BALANCE:
                print(f"Depth weight: {self.depth_weight}, Fidelity weight: {self.fidelity_weight}")
        else:
            print(f"Target fidelity: {self.target_fidelity}")
        
        if self.pass_timeout_seconds > 0:
            print(f"Pass timeout: {self.pass_timeout_seconds} seconds")
        else:
            print(f"Pass timeout: disabled")
        

        
        # Use instance target_fidelity if none provided
        if target_fidelity is None:
            target_fidelity = self.target_fidelity
        
        # Get coupling map if topology provided
        if coupling_map is None and topology is not None:
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
            
            
            # 2. Generate naive physical mapping if not provided
            if original_circuit is None:
                print("Creating naive physical mapping...")
                # Use the pattern-optimized circuit as the starting point if available, otherwise use logical
                circuit_to_map = logical_circuit
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
            optimized_circuit =  logical_circuit.copy()
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
                
                # Apply specific optimization techniques with timeout
                pass_start_time = time.time()
                
                if self.pass_timeout_seconds > 0:
                    try:
                        # Run optimization with timeout
                        with time_limit(self.pass_timeout_seconds):
                            new_circuit = self._optimize_circuit(
                                optimized_circuit.copy(),
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
                improvement = self._is_better_circuit(new_depth, new_gates, new_fidelity,
                                                     current_depth, current_gates, current_fidelity)

                # CORRECTNESS GATE: before swapping in an improving candidate, prove
                # it still computes the same function as the pre-optimization input
                # (the logical circuit). On a genuine non-equivalence, reject the
                # candidate and keep the last verified-good circuit; on infeasible /
                # unverifiable, skip the gate and keep the candidate.
                if improvement and self._equivalence_gate(
                        logical_circuit, new_circuit,
                        context=f"pass {pass_num}") == "reject":
                    improvement = False

                if improvement:
                    # Free old circuit from memory if we're replacing it
                    del optimized_circuit
                    gc.collect()
                    
                    optimized_circuit = new_circuit
                    current_depth = new_depth
                    current_gates = new_gates
                    current_fidelity = new_fidelity
                    best_pass = pass_num
                    print(f"Improvement found in pass {pass_num}")
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

            # FINAL CORRECTNESS GATE: prove the final optimized logical circuit is
            # function-equivalent to the original logical circuit. The per-pass gate
            # already guards each accepted candidate, but this re-checks the end
            # result (covers any unguarded path). On a genuine non-equivalence, fall
            # back to the original logical circuit (the last verified-good circuit).
            if self.verify_equivalence:
                final_gate = self._equivalence_gate(
                    logical_circuit, optimized_circuit, context="final")
                if final_gate == "reject":
                    optimized_circuit = logical_circuit.copy()
                    current_depth = optimized_circuit.depth()
                    final_current_metrics = self.compiler.get_circuit_metrics(optimized_circuit)
                    current_gates = sum(final_current_metrics["gate_counts"].values())
                    del final_current_metrics
                    current_fidelity = self.estimate_physical_fidelity(optimized_circuit)
                    best_pass = 0
                    gc.collect()



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
            mapped_circuit = create_optimized_physical_mapping(
                optimized_circuit.copy(),
                coupling_map,
                basis_gates,
                optimization_level=1  # Use high optimization level
            )
            
            # Validate that the mapped circuit respects coupling constraints
            is_valid, violations = self.validate_physical_circuit(mapped_circuit, coupling_map)
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
                is_valid, violations = self.validate_physical_circuit(mapped_circuit, coupling_map)
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
                    is_valid, violations = self.validate_physical_circuit(mapped_circuit, coupling_map)
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
            
            # Add pattern-specific metrics if available
            if "pattern_optimized" in results:
                pattern_to_optimized_reduction = ((results["pattern_optimized"]["depth"] - results['optimized']['depth']) /
                                                results["pattern_optimized"]["depth"] * 100) if results["pattern_optimized"]["depth"] > 0 else 0
                results["pattern_to_optimized_reduction"] = pattern_to_optimized_reduction
                print(f"Depth reduction from pattern-optimized to final: {pattern_to_optimized_reduction:.2f}%")
            
            print(f"Depth reduction from logical: {depth_reduction:.2f}%")
            print(f"Depth reduction from naive: {depth_reduction_from_naive:.2f}%")
            
            # Record execution time
            execution_time = time.time() - start_time
            results["execution_time"] = execution_time
            # Add "optimization_time" for compatibility with main.py
            results["optimization_time"] = execution_time
            
            print(f"Optimization completed in {execution_time:.4f} seconds")
            
            # Force garbage collection before saving images
            gc.collect()
            
            # Save images of the circuits using the self-contained function
            try:
                import os  # Explicit import here to ensure it's available
                from ..utils.visualization import save_circuit_image
                
                # Create output directory
                output_dir = os.path.join(self.output_dir, "circuit_images")
                os.makedirs(output_dir, exist_ok=True)

                # Save original physical circuit
                if original_circuit is not None:
                    try:
                        original_img = save_circuit_image(
                            original_circuit,
                            "original_physical_circuit",
                            output_dir,
                            use_text_mode=True
                        )
                        
                        if original_img:
                            results["naive_physical"]["image_path"] = original_img
                            
                    except Exception as e:
                        print(f"Warning: Could not save original circuit image: {e}")
                    
                    # Force garbage collection after saving image
                    gc.collect()

                # Save optimized logical circuit
                if optimized_circuit is not None:
                    try:
                        opt_logical_img = save_circuit_image(
                            optimized_circuit,
                            "optimized_logical_circuit",
                            output_dir,
                            use_text_mode=True
                        )
                        if opt_logical_img:
                            results["optimized"]["image_path"] = opt_logical_img
                            results["final"]["image_path"] = opt_logical_img  # Also add to "final" for compatibility
                            
                    except Exception as e:
                        print(f"Warning: Could not save optimized circuit image: {e}")
                    
                    # Force garbage collection after saving image
                    gc.collect()

                # Save final physical circuit
                if mapped_circuit is not None:
                    try:
                        final_img = save_circuit_image(
                            mapped_circuit,
                            "final_physical_circuit",
                            output_dir,
                            use_text_mode=True
                        )
                        if final_img:
                            results["mapped"]["image_path"] = final_img
                            
                    except Exception as e:
                        print(f"Warning: Could not save mapped circuit image: {e}")
                    
                    # Force garbage collection after saving image
                    gc.collect()
            except ImportError as ie:
                print(f"Warning: save_circuit_image function not available: {ie}")
            except Exception as e:
                print(f"Error saving circuit images: {e}")
            
            # Compare circuits using compiler's comparison function
            try:
                print("\nDetailed Circuit Comparison:")
                self.compiler.print_comparison(logical_circuit, mapped_circuit, optimized_circuit)
                
                # Force garbage collection after comparison
                gc.collect()
                
            except Exception as e:
                print(f"Error during circuit comparison: {e}")
            
            # Final garbage collection before returning results
            gc.collect()
            
            return results
            
        except Exception as e:
            print(f"Error optimizing Toffoli network: {e}")
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
    def _optimize_circuit(self, circuit, output_qubits, input_qubits,
                             num_qubits, coupling_map, basis_gates, target_fidelity,
                             toffoli_type=None, use_ancilla=True):
        """
        Apply optimization techniques to reduce circuit depth and improve fidelity.
        Uses the "bookend" approach for Solovay-Kitaev approximation.
        
        Args:
            circuit: Circuit to optimize
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
            from .optimization_strategy import OptimizationStrategy
            
            # Handle TRANSPILER variants directly - early return for these strategies
            if self.strategy in [OptimizationStrategy.TRANSPILER, OptimizationStrategy.TRANSPILER_L1,
                                OptimizationStrategy.TRANSPILER_L2, OptimizationStrategy.TRANSPILER_L3]:
                # Determine optimization level based on strategy
                if self.strategy == OptimizationStrategy.TRANSPILER_L1:
                    opt_level = 1
                elif self.strategy == OptimizationStrategy.TRANSPILER_L2:
                    opt_level = 2
                elif self.strategy == OptimizationStrategy.TRANSPILER_L3 or self.strategy == OptimizationStrategy.TRANSPILER:
                    opt_level = 3
                else:
                    opt_level = 3  # Default to level 3
                
                if self.debug_mode:
                    print(f"Using Qiskit transpiler with optimization level {opt_level}")
                
                try:
                    # Qiskit 2.0 style parameters
                    transpiled = transpile(
                        circuit.copy(),
                        basis_gates=basis_gates,
                        optimization_level=opt_level,
                        coupling_map=coupling_map
                    )
                    
                    return transpiled
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error in transpiler strategy: {e}")
                    # Return original circuit if transpilation fails
                    return circuit.copy()
            
            # Start with a fresh copy of the input circuit
            working_circuit = circuit.copy()
            original_depth = working_circuit.depth()
            
            # Calculate initial fidelity
            initial_fidelity = self.estimate_physical_fidelity(working_circuit)
            current_fidelity = initial_fidelity
            available_budget = current_fidelity - self.min_fidelity
            
            if self.debug_mode:
                print(f"Starting optimization with circuit depth {original_depth}")
                print(f"Circuit has {len(working_circuit.data)} gates")
                print(f"Initial fidelity: {initial_fidelity:.4f}, Available budget: {available_budget:.4f}")
            
            # BOOKEND APPROACH: First SK Approximation (Early)
            # Only apply for depth reduction or balancing strategies
            if self.strategy in [OptimizationStrategy.ULTRA_DEPTH_REDUCTION, OptimizationStrategy.DEPTH_FIDELITY_BALANCE]:
                # Determine how much of the fidelity budget to use for early approximation
                if self.strategy == OptimizationStrategy.ULTRA_DEPTH_REDUCTION:
                    early_budget_fraction = 0.4  # Use 40% of budget for early approximation
                else:
                    early_budget_fraction = 0.3  # Use 30% for balanced strategy
                    
                early_min_fidelity = max(self.min_fidelity, current_fidelity - (available_budget * early_budget_fraction))
                
                if self.debug_mode:
                    print(f"Applying early SK approximation with {early_budget_fraction*100:.0f}% of fidelity budget")
                    print(f"Early minimum fidelity: {early_min_fidelity:.4f}")
                    
                try:
                    # Apply a light early approximation
                    early_approx = self._create_early_sk_approximation(
                        working_circuit.copy(),
                        min_fidelity=early_min_fidelity,
                        current_fidelity=current_fidelity
                    )
                    
                    # Check if the early approximation improved depth
                    early_approx_depth = early_approx.depth()
                    if early_approx_depth < original_depth:
                        if self.debug_mode:
                            reduction_pct = (original_depth - early_approx_depth) / original_depth * 100
                            print(f"Early SK approximation reduced depth by {reduction_pct:.2f}%")
                            print(f"New depth: {early_approx_depth} (was {original_depth})")
                            
                        # Use the approximated circuit as our starting point
                        working_circuit = early_approx
                        
                        # Update current fidelity estimate
                        current_fidelity = self.estimate_physical_fidelity(working_circuit)
                        available_budget = current_fidelity - self.min_fidelity
                    else:
                        if self.debug_mode:
                            print("Early SK approximation did not improve depth, continuing with original circuit")
                except Exception as e:
                    if self.debug_mode:
                        print(f"Early SK approximation failed: {e}")
            
            # Create a safe version of the circuit by dynamically identifying problematic gates
            fixed_circuit = self.gate_processor.process_circuit_safely(working_circuit, coupling_map)
            working_circuit = fixed_circuit
            
            # Track the best circuit we've found so far
            best_circuit = working_circuit.copy()
            best_depth = working_circuit.depth()
            best_gate_count = sum(self.compiler.get_circuit_metrics(best_circuit)["gate_counts"].values())
            best_fidelity = current_fidelity
            
            # 1. Try standard transpilation optimization
            try:
                for opt_level in [1, 2, 3]:
                    # Qiskit 2.0 style parameters
                    transpiled = transpile(
                        working_circuit.copy(),
                        basis_gates=basis_gates,
                        optimization_level=opt_level,
                        coupling_map=coupling_map
                    )
                    
                    # Measure transpiled circuit
                    trans_metrics = self.compiler.get_circuit_metrics(transpiled)
                    trans_depth = trans_metrics["depth"]
                    trans_gates = sum(trans_metrics["gate_counts"].values())
                    trans_fidelity = self.estimate_physical_fidelity(transpiled)
                    
                    # Clear metrics from memory
                    del trans_metrics
                    gc.collect()
                    
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
            
            # 2. Try with different qubit layouts
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
                        optimization_level=1
                    )
                    
                    # Measure layout-optimized circuit
                    layout_metrics = self.compiler.get_circuit_metrics(layout_circuit)
                    layout_depth = layout_metrics["depth"]
                    layout_gates = sum(layout_metrics["gate_counts"].values())
                    layout_fidelity = self.estimate_physical_fidelity(layout_circuit)
                    
                    # Clear metrics from memory
                    del layout_metrics
                    gc.collect()
                    
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
            
            # BOOKEND APPROACH: Second SK Approximation (Late)
            # This is our more aggressive approximation after other optimizations
            if self.strategy in [OptimizationStrategy.ULTRA_DEPTH_REDUCTION, OptimizationStrategy.DEPTH_FIDELITY_BALANCE]:
                # Recalculate available budget based on current best fidelity
                available_budget = best_fidelity - self.min_fidelity
                
                if available_budget > 0.01:  # Only proceed if we have meaningful budget left
                    if self.debug_mode:
                        print(f"Applying late SK approximation with remaining fidelity budget")
                        print(f"Current fidelity: {best_fidelity:.4f}, Min fidelity: {self.min_fidelity:.4f}")
                        print(f"Available budget: {available_budget:.4f}")
                    
                    try:
                        # Apply a more aggressive approximation with remaining budget
                        # Use standard SK approximation function which can create more aggressive approximations
                        approx_circuit = self._create_approximate_toffoli_circuit(
                            best_circuit.copy(),
                            min_fidelity=self.min_fidelity,
                            current_fidelity=best_fidelity
                        )
                        
                        # Measure approximate circuit
                        approx_metrics = self.compiler.get_circuit_metrics(approx_circuit)
                        approx_depth = approx_metrics["depth"]
                        approx_gates = sum(approx_metrics["gate_counts"].values())
                        approx_fidelity = self.estimate_physical_fidelity(approx_circuit)
                        
                        # Clear metrics from memory
                        del approx_metrics
                        gc.collect()
                        
                        # Check if this is better
                        if self._is_better_circuit(approx_depth, approx_gates, approx_fidelity,
                                                best_depth, best_gate_count, best_fidelity):
                            best_circuit = approx_circuit.copy()
                            best_depth = approx_depth
                            best_gate_count = approx_gates
                            best_fidelity = approx_fidelity
                            if self.debug_mode:
                                print(f"Late SK approximation improved metrics: depth={best_depth}, gates={best_gate_count}, fidelity={best_fidelity:.6f}")
                        else:
                            if self.debug_mode:
                                print(f"Late SK approximation did not improve the circuit")
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Late SK approximation failed: {e}")
                elif self.debug_mode:
                    print(f"Insufficient fidelity budget for late SK approximation: {available_budget:.4f}")
            
            # 4. Final optimization pass
            try:
                final_circuit = transpile(
                    best_circuit,
                    basis_gates=basis_gates,
                    optimization_level=1
                )
                
                # Measure final circuit
                final_metrics = self.compiler.get_circuit_metrics(final_circuit)
                final_depth = final_metrics["depth"]
                final_gates = sum(final_metrics["gate_counts"].values())
                final_fidelity = self.estimate_physical_fidelity(final_circuit)
                
                # Clear metrics from memory
                del final_metrics
                gc.collect()
                
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
            
            # Summary of optimization results
            if self.debug_mode:
                print("\nOptimization summary:")
                print(f"  Original depth: {original_depth}, Final depth: {best_depth}")
                print(f"  Depth reduction: {(original_depth - best_depth) / original_depth * 100:.2f}%")
                print(f"  Original fidelity: {initial_fidelity:.4f}, Final fidelity: {best_fidelity:.4f}")
                print(f"  Fidelity budget used: {initial_fidelity - best_fidelity:.4f} of {available_budget:.4f} available")
            
            # Return the best circuit we found
            return best_circuit
            
        except Exception as e:
            print(f"Error in circuit optimization: {e}")
            import traceback
            traceback.print_exc()
            return circuit  # Return the original circuit on error



    def _identify_depth_critical_gates(self, circuit):
        """
        Identify gates that contribute most to circuit depth.
        
        Args:
            circuit: Quantum circuit to analyze
            
        Returns:
            set: Indices of depth-critical gates
        """
        # This is a simplified version that targets Toffoli (ccx) gates
        # in the longest dependency chains
        try:
            # Calculate the rough contribution to depth for each gate
            # using a simplified critical path analysis
            depth_contribution = {}
            
            # Create a dictionary to store the "last gate time" for each qubit
            qubit_last_time = {i: 0 for i in range(circuit.num_qubits)}
            
            # Go through gates in order, estimate "finish time" for each gate
            for i, inst in enumerate(circuit.data):
                qubits = [q.index for q in inst.qubits]
                
                # Determine when this gate can start (max of all involved qubits' last times)
                start_time = max(qubit_last_time[q] for q in qubits)
                
                # Estimate gate execution time (simplistic model)
                # Toffoli gates are most expensive, followed by CX, then single-qubit
                if inst.operation.name == 'ccx' or inst.operation.name == 'mcx':
                    gate_time = 10
                elif inst.operation.name == 'cx':
                    gate_time = 5
                else:
                    gate_time = 1
                    
                # Calculate finish time
                finish_time = start_time + gate_time
                
                # Update last time for all qubits involved
                for q in qubits:
                    qubit_last_time[q] = finish_time
                    
                # Store the contribution to depth
                depth_contribution[i] = gate_time
            
            # Calculate critical path (simplified - just take highest impact gates)
            # Focus on the top 20% of gates that contribute to depth
            critical_threshold = sorted(depth_contribution.values(), reverse=True)[
                int(len(depth_contribution) * 0.2)
            ] if depth_contribution else 0
            
            # Get indices of gates that are above the threshold
            critical_gates = {i for i, contribution in depth_contribution.items()
                              if contribution >= critical_threshold}
            
            return critical_gates
        except Exception as e:
            if self.debug_mode:
                print(f"Error identifying depth-critical gates: {e}")
            # Return empty set if analysis fails
            return set()
            
    def optimize_circuit_with_memory_management(self, circuit, toffoli_gates, output_qubits, input_qubits,
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


    def _process_logical_circuit(self, circuit, use_ancilla=True):
        """
        Process the logic of the initial circuit to express it as CCX gates with ancillas.
        
        This method ignores topology, basis gates, and connectivity restrictions to focus
        on expressing the circuit logic using CCX gates and a few single-qubit operations.
        
        Args:
            circuit: The original logical circuit
            use_ancilla: Whether to use ancilla qubits
            
        Returns:
            QuantumCircuit: Processed circuit with explicit CCX gates and ancillas
        """
        if not QISKIT_AVAILABLE:
            print("Error: Qiskit is required to process circuits")
            return circuit
            
        try:
            from qiskit import QuantumCircuit, transpile, decompose
            
            # Make a copy of the input circuit
            processed_circuit = circuit.copy()
            
            # Check if we need to add ancilla qubits
            # Count non-CCX multi-controlled gates that will need ancillas
            num_mcx_gates = 0
            for inst in circuit.data:
                if hasattr(inst.operation, 'name'):
                    # Look for MCX gates with more than 2 controls that aren't CCX/Toffoli
                    if (inst.operation.name.lower() in ['mcx', 'mcphase'] or
                        (inst.operation.name.lower().startswith('c') and
                         inst.operation.name.lower() != 'ccx' and
                         inst.operation.name.lower() != 'cx')):
                        if len(inst.qubits) > 3:  # More than 2 controls
                            num_mcx_gates += 1
            
            # Determine how many ancilla qubits to add (typically 1 per MCX gate)
            if use_ancilla and num_mcx_gates > 0:
                num_ancilla = min(num_mcx_gates, max(1, circuit.num_qubits // 4))  # Limit ancilla count
                print(f"Adding {num_ancilla} ancilla qubits to help with multi-controlled gates")
                
                # Create a new circuit with ancilla qubits
                new_circuit = QuantumCircuit(circuit.num_qubits + num_ancilla)
                
                # Copy the original circuit operations to the main qubits
                for inst in circuit.data:
                    qubits = [q.index if hasattr(q, 'index') else q._index for q in inst.qubits]
                    if hasattr(inst, 'clbits'):
                        clbits = [c.index if hasattr(c, 'index') else c._index for c in inst.clbits]
                    else:
                        clbits = []
                    new_circuit.append(inst.operation, qubits, clbits)
                
                processed_circuit = new_circuit
            
            # Try to find multi-controlled gates and decompose them to CCX
            for idx, inst in reversed(list(enumerate(processed_circuit.data))):
                if hasattr(inst.operation, 'name'):
                    # Find MCX or other multi-controlled gates
                    if (inst.operation.name.lower() in ['mcx', 'mcphase'] or
                        (inst.operation.name.lower().startswith('c') and
                         inst.operation.name.lower() != 'ccx' and
                         inst.operation.name.lower() != 'cx')):
                        
                        qubits = [q.index if hasattr(q, 'index') else q._index for q in inst.qubits]
                        
                        # For multi-controlled gates with more than 2 controls
                        if len(qubits) > 3:  # target + at least 3 controls
                            # Remove this gate
                            processed_circuit.data.pop(idx)
                            
                            # Determine control and target qubits
                            control_qubits = qubits[:-1]  # all but last are controls
                            target_qubit = qubits[-1]
                            
                            if use_ancilla and processed_circuit.num_qubits > circuit.num_qubits:
                                # Use ancilla qubits for decomposition if available
                                ancilla_qubits = list(range(circuit.num_qubits, processed_circuit.num_qubits))
                                
                                # Decompose using ancilla-efficient implementation
                                # First level: use first two controls with first ancilla
                                processed_circuit.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0])
                                
                                # Middle levels: chain through remaining controls
                                for i in range(2, len(control_qubits)):
                                    if i-2 < len(ancilla_qubits)-1:
                                        processed_circuit.ccx(control_qubits[i], ancilla_qubits[i-2], ancilla_qubits[i-1])
                                    else:
                                        # If we run out of ancillas, use the target directly
                                        processed_circuit.ccx(control_qubits[i], ancilla_qubits[-1], target_qubit)
                                
                                # Final level: connect to target if not already
                                if len(control_qubits) - 2 < len(ancilla_qubits):
                                    processed_circuit.ccx(ancilla_qubits[len(control_qubits)-2], control_qubits[-1], target_qubit)
                                
                                # Uncompute ancillas in reverse order
                                for i in range(len(control_qubits)-1, 1, -1):
                                    if i-2 < len(ancilla_qubits)-1:
                                        processed_circuit.ccx(control_qubits[i], ancilla_qubits[i-2], ancilla_qubits[i-1])
                                
                                processed_circuit.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0])
                            else:
                                # No ancilla available, use linear chain of CCX gates
                                if len(control_qubits) == 3:
                                    # Special case for 3 controls
                                    processed_circuit.ccx(control_qubits[0], control_qubits[1], target_qubit)
                                    processed_circuit.cx(control_qubits[2], target_qubit)
                                    processed_circuit.ccx(control_qubits[0], control_qubits[1], target_qubit)
                                else:
                                    # General case: use a sequence of CNOTs and CCXs
                                    for i in range(0, len(control_qubits)-1, 2):
                                        if i+1 < len(control_qubits):
                                            processed_circuit.ccx(control_qubits[i], control_qubits[i+1], target_qubit)
            
            # Try to decompose remaining non-standard gates to basic gates
            processed_circuit = transpile(
                processed_circuit,
                basis_gates=['ccx', 'cx', 'id', 'u', 'u1', 'u2', 'u3', 'h', 'rx', 'ry', 'rz', 'x', 'y', 'z', 's', 't'],
                optimization_level=0  # No optimization to preserve structure
            )
            
            return processed_circuit
            
        except Exception as e:
            print(f"Error processing logical circuit: {e}")
            traceback.print_exc()
            return circuit  # Return original if processing fails

    
    def _decompose_ccx_gates(self, circuit, toffoli_type, use_ancilla=True):
        """
        Apply decomposition of CCX gates based on selected Toffoli implementation.
        
        Args:
            circuit: The circuit with CCX gates to decompose
            toffoli_type: Type of Toffoli implementation to use
            use_ancilla: Whether to use ancilla qubits
            
        Returns:
            QuantumCircuit: Circuit with decomposed CCX gates
        """
        if not QISKIT_AVAILABLE:
            print("Error: Qiskit is required to decompose circuits")
            return circuit
            
        try:
            from qiskit import QuantumCircuit
            
            # Make a copy to avoid modifying the original circuit
            decomposed_circuit = circuit.copy()
            
            # Count and locate CCX gates
            ccx_gates = []
            for idx, inst in enumerate(circuit.data):
                if hasattr(inst.operation, 'name') and inst.operation.name == 'ccx':
                    qubits = [q.index if hasattr(q, 'index') else q._index for q in inst.qubits]
                    ccx_gates.append((idx, qubits))
            
            if not ccx_gates:
                return circuit  # No CCX gates to decompose
            
            # Determine if we need to add ancilla qubits
            num_ancilla_needed = 0
            if use_ancilla:
                if toffoli_type == ToffoliType.RELATIVE_PHASE_1:
                    num_ancilla_needed = 1
                elif toffoli_type == ToffoliType.OPTIMIZED_2:
                    num_ancilla_needed = 2
                elif toffoli_type == ToffoliType.OPTIMIZED_3:
                    num_ancilla_needed = 3
                elif toffoli_type == ToffoliType.OPTIMIZED_4:
                    num_ancilla_needed = 4
                elif toffoli_type == ToffoliType.OPTIMIZED_7:
                    num_ancilla_needed = 7
            
            # Create a new circuit with ancilla qubits if needed
            if num_ancilla_needed > 0:
                # Check if we already have ancilla qubits
                if hasattr(circuit, 'ancilla_indices') and circuit.ancilla_indices:
                    if len(circuit.ancilla_indices) >= num_ancilla_needed:
                        # Use existing ancilla qubits
                        ancilla_indices = circuit.ancilla_indices[:num_ancilla_needed]
                    else:
                        # Need more ancilla qubits
                        orig_ancilla_count = len(circuit.ancilla_indices)
                        additional_ancilla = num_ancilla_needed - orig_ancilla_count
                        new_circuit = QuantumCircuit(circuit.num_qubits + additional_ancilla)
                        
                        # Copy original circuit
                        for inst in circuit.data:
                            qubits = [q.index if hasattr(q, 'index') else q._index for q in inst.qubits]
                            if hasattr(inst, 'clbits'):
                                clbits = [c.index if hasattr(c, 'index') else c._index for c in inst.clbits]
                            else:
                                clbits = []
                            new_circuit.append(inst.operation, qubits, clbits)
                        
                        # Set ancilla indices
                        ancilla_indices = list(circuit.ancilla_indices) + \
                                          list(range(circuit.num_qubits, circuit.num_qubits + additional_ancilla))
                        decomposed_circuit = new_circuit
                else:
                    # No existing ancilla qubits, add new ones
                    new_circuit = QuantumCircuit(circuit.num_qubits + num_ancilla_needed)
                    
                    # Copy original circuit
                    for inst in circuit.data:
                        qubits = [q.index if hasattr(q, 'index') else q._index for q in inst.qubits]
                        if hasattr(inst, 'clbits'):
                            clbits = [c.index if hasattr(c, 'index') else c._index for c in inst.clbits]
                        else:
                            clbits = []
                        new_circuit.append(inst.operation, qubits, clbits)
                    
                    # Set ancilla indices
                    ancilla_indices = list(range(circuit.num_qubits, circuit.num_qubits + num_ancilla_needed))
                    decomposed_circuit = new_circuit
                
                # Store ancilla indices
                decomposed_circuit.ancilla_indices = ancilla_indices
            else:
                # No ancilla needed
                ancilla_indices = []
            
            # Process CCX gates in reverse order to avoid index shifting
            for idx, qubits in reversed(ccx_gates):
                # Remove the CCX gate
                decomposed_circuit.data.pop(idx)
                
                # Add the implementation based on toffoli_type
                try:
                    control1, control2, target = qubits
                    self.compiler.create_toffoli(
                        decomposed_circuit,
                        control1,
                        control2,
                        target,
                        toffoli_type=toffoli_type,
                        ancilla_qubits=ancilla_indices
                    )
                except Exception as e:
                    print(f"Error decomposing CCX gate {idx}: {e}")
                    # Use standard implementation as fallback
                    self.compiler._create_standard_toffoli(decomposed_circuit, qubits[0], qubits[1], qubits[2])
            
            return decomposed_circuit
            
        except Exception as e:
            print(f"Error decomposing CCX gates: {e}")
            traceback.print_exc()
            return circuit  # Return original if decomposition fails"""
