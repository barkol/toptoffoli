"""
Enhanced Toffoli Depth Optimizer

This module provides advanced optimization functionality to minimize the depth of Toffoli gate 
networks on quantum hardware with limited connectivity, with options to trade fidelity for depth.
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

# Import necessary components
from ..utils.circuit_utils import (
    create_optimized_physical_mapping, 
    QISKIT_AVAILABLE, 
    estimate_fidelity, 
    validate_physical_circuit
)
from .compiler import ToffoliCompiler, ToffoliType
from .zx_optimizer import ZXOptimizer  # Import the new ZX calculus optimizer

# Check for Qiskit availability
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator
except ImportError:
    pass

# Try to import RL optimizer 
try:
    from .rl_optimizer import RLToffoliOptimizer
    RL_OPTIMIZER_AVAILABLE = True
except ImportError:
    RL_OPTIMIZER_AVAILABLE = False
    print("Warning: RL-based optimizer not available. Will use standard optimization only.")

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

# Enhanced optimization strategies enum
class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies."""
    STANDARD = auto()                # Prioritize depth, then gates, then fidelity
    DEPTH_REDUCTION = auto()         # Only care about depth
    GATE_REDUCTION = auto()          # Only care about gate count
    FIDELITY = auto()                # Only care about fidelity
    HYBRID = auto()                  # Use a weighted score of all metrics
    ULTRA_DEPTH_REDUCTION = auto()   # Aggressively reduce depth with fidelity trade-offs
    DEPTH_FIDELITY_BALANCE = auto()  # Explicitly balance depth and fidelity with configurable weights

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


def validate_physical_circuit(circuit, coupling_map):
    """
    Validate that a physical circuit respects the coupling map constraints.
    
    Args:
        circuit: Quantum circuit to validate
        coupling_map: List of qubit pairs representing valid connections
        
    Returns:
        tuple: (is_valid, violations) where is_valid is a boolean and violations is a list of violations
    """
    if circuit is None or coupling_map is None:
        return False, ["Circuit or coupling map is None"]
    
    # Convert coupling map to a set of tuples for faster lookup
    if isinstance(coupling_map, list):
        # Convert list of lists to set of tuples
        coupling_set = set()
        for pair in coupling_map:
            if isinstance(pair, list) and len(pair) == 2:
                coupling_set.add((pair[0], pair[1]))
                coupling_set.add((pair[1], pair[0]))  # Both directions are valid
    else:
        # Assume it's already a set-like object
        coupling_set = coupling_map
    
    violations = []
    
    try:
        # Check each two-qubit gate
        for idx, instruction in enumerate(circuit.data):
            qubits = instruction.qubits
            
            # Only check multi-qubit gates
            if len(qubits) >= 2:
                # Get qubit indices
                try:
                    if hasattr(qubits[0], '_index'):
                        # Newer Qiskit
                        qubit_indices = [q._index for q in qubits]
                    else:
                        # Older Qiskit
                        qubit_indices = [q.index for q in qubits]
                except Exception:
                    # Skip if we can't determine qubit indices
                    continue
                
                # For CX gates, check if the control and target are connected
                if instruction.operation.name in ['cx', 'CX', 'cnot']:
                    # For two-qubit gates, check if the qubits are connected
                    if (qubit_indices[0], qubit_indices[1]) not in coupling_set:
                        violations.append(f"Gate {instruction.operation.name} at index {idx} "
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
                                violations.append(f"Multi-qubit gate {operation_name} at index {idx} "
                                                f"between qubits {qubit_indices[i]} and {qubit_indices[j]} "
                                                f"violates coupling map")
    
    except Exception as e:
        violations.append(f"Error during validation: {e}")
    
    is_valid = len(violations) == 0
    return is_valid, violations


class EnhancedToffoliDepthOptimizer:
    """
    Enhanced optimizer for minimizing the depth of Toffoli gate networks on quantum hardware.
    
    This class extends the original ToffoliDepthOptimizer with additional strategies
    that allow for controlled fidelity sacrifices to achieve better depth reduction.
    It also integrates ZX calculus for advanced circuit transformations.
    
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
        use_zx_optimization (bool): Whether to use ZX calculus optimization
        use_rl_optimization (bool): Whether to use reinforcement learning optimization
    """
    
    def __init__(self, target_fidelity=0.95, min_fidelity=0.8, max_passes=2, output_dir=None,
                use_parallel=True, debug_mode=False, pass_timeout_seconds=60,
                strategy=OptimizationStrategy.HYBRID, depth_weight=0.7, fidelity_weight=0.3,
                use_zx_optimization=True, use_rl_optimization=True):
        """
        Initialize the Enhanced Toffoli Depth Optimizer.
        
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
                New options include:
                - ULTRA_DEPTH_REDUCTION: Aggressively reduces depth while allowing fidelity to drop
                  to min_fidelity
                - DEPTH_FIDELITY_BALANCE: Explicitly balances depth and fidelity using the provided
                  depth_weight and fidelity_weight parameters
            
            depth_weight (float): Weight for depth in balanced optimization (0.0-1.0).
                Only applies to DEPTH_FIDELITY_BALANCE strategy. Higher values prioritize depth more.
            
            fidelity_weight (float): Weight for fidelity in balanced optimization (0.0-1.0).
                Only applies to DEPTH_FIDELITY_BALANCE strategy. Higher values prioritize fidelity more.
            
            use_zx_optimization (bool): Whether to use ZX calculus optimization.
                ZX calculus can provide additional depth reductions but may increase runtime.
                
            use_rl_optimization (bool): Whether to use reinforcement learning optimization.
                RL optimization can find better depth reductions but requires more memory and time.
        """
        self.target_fidelity = target_fidelity
        self.min_fidelity = min_fidelity
        self.max_passes = max_passes
        self.use_parallel = use_parallel
        self.debug_mode = debug_mode
        self.pass_timeout_seconds = pass_timeout_seconds
        self.depth_weight = depth_weight
        self.fidelity_weight = fidelity_weight
        self.use_zx_optimization = use_zx_optimization
        self.use_rl_optimization = use_rl_optimization and RL_OPTIMIZER_AVAILABLE
        
        # Set strategy - default to HYBRID if not a valid enum
        if isinstance(strategy, OptimizationStrategy):
            self.strategy = strategy
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
        
        # Initialize ZX optimizer if enabled
        if self.use_zx_optimization:
            # For aggressive mode, use aggressive=True if the strategy is ULTRA_DEPTH_REDUCTION
            aggressive_mode = (self.strategy == OptimizationStrategy.ULTRA_DEPTH_REDUCTION)
            self.zx_optimizer = ZXOptimizer(
                aggressive_mode=aggressive_mode,
                target_fidelity=self.min_fidelity if aggressive_mode else self.target_fidelity,
                debug_mode=self.debug_mode
            )
            
        # Initialize RL optimizer if enabled
        if self.use_rl_optimization:
            try:
                q_table_file = None
                # Try to find a pre-trained Q-table if available
                if os.path.exists("rl_toffoli_qtable.pkl"):
                    q_table_file = "rl_toffoli_qtable.pkl"
                elif os.path.exists(os.path.join(self.output_dir, "rl_toffoli_qtable.pkl")):
                    q_table_file = os.path.join(self.output_dir, "rl_toffoli_qtable.pkl")
                    
                self.rl_optimizer = RLToffoliOptimizer(
                    q_table_file=q_table_file,
                    target_fidelity=self.min_fidelity if (self.strategy == OptimizationStrategy.ULTRA_DEPTH_REDUCTION) else self.target_fidelity,
                    max_steps=30,
                    debug_mode=self.debug_mode
                )
            except Exception as e:
                print(f"Failed to initialize RL optimizer: {e}")
                self.use_rl_optimization = False
    
    def optimize_toffoli_network(self, toffoli_gates, output_qubits, input_qubits,
                               num_qubits, topology=None, coupling_map=None,
                               basis_gates=None, target_fidelity=None,
                               original_circuit=None, logical_circuit=None,
                               toffoli_type=ToffoliType.RELATIVE_PHASE_1,
                               use_ancilla=True):
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
        
        print(f"\nOptimizing Toffoli network with {len(toffoli_gates)} gates on {num_qubits} qubits")
        if isinstance(toffoli_type, ToffoliType):
            print(f"Using {toffoli_type.name} implementation")
        else:
            print(f"Using STANDARD implementation")
        print(f"Max optimization passes: {self.max_passes}")
        if isinstance(self.strategy, OptimizationStrategy):
            print(f"Optimization strategy: {self.strategy.name}")
        else:
            print(f"Optimization strategy: STANDARD")
        
        # Print fidelity parameters based on strategy
        if self.strategy in [OptimizationStrategy.ULTRA_DEPTH_REDUCTION, OptimizationStrategy.DEPTH_FIDELITY_BALANCE]:
            print(f"Target fidelity: {self.target_fidelity}, Minimum fidelity: {self.min_fidelity}")
            if self.strategy == OptimizationStrategy.DEPTH_FIDELITY_BALANCE:
                print(f"Depth weight: {self.depth_weight}, Fidelity weight: {self.fidelity_weight}")
        else:
            print(f"Target fidelity: {self.target_fidelity}")
        
        if self.pass_timeout_seconds > 0:
            print(f"Pass timeout: {self.pass_timeout_seconds} seconds")
        else:
            print("Pass timeout: disabled")
        
        if self.use_zx_optimization:
            print("ZX calculus optimization: Enabled")
        else:
            print("ZX calculus optimization: Disabled")
            
        if self.use_rl_optimization:
            print("Reinforcement learning optimization: Enabled")
        else:
            print("Reinforcement learning optimization: Disabled")
        
        # Use instance target_fidelity if none provided
        if target_fidelity is None:
            target_fidelity = self.target_fidelity
        
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
            
            # 2. Generate naive physical mapping if not provided
            if original_circuit is None:
                print("Creating naive physical mapping...")
                original_circuit = create_optimized_physical_mapping(
                    logical_circuit.copy(),
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
            
            # Start with the original circuit
            optimized_circuit = logical_circuit.copy()
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
                
                # Apply your specific optimization techniques with timeout
                pass_start_time = time.time()
                
                if self.pass_timeout_seconds > 0:
                    try:
                        # Run optimization with timeout
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
                improvement = self._is_better_circuit(new_depth, new_gates, new_fidelity,
                                                     current_depth, current_gates, current_fidelity)
                
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
            
            # Apply ZX calculus optimization if enabled
            if self.use_zx_optimization:
                print("\nApplying ZX calculus optimization...")
                
                try:
                    zx_optimized_circuit = self.zx_optimizer.optimize_circuit(
                        optimized_circuit.copy(),
                        coupling_map=coupling_map
                    )
                    
                    # Check if ZX optimization helped
                    zx_metrics = self.compiler.get_circuit_metrics(zx_optimized_circuit)
                    zx_depth = zx_metrics["depth"]
                    zx_gates = sum(zx_metrics["gate_counts"].values())
                    
                    # Clear metrics from memory
                    del zx_metrics
                    gc.collect()
                    
                    zx_fidelity = self.estimate_physical_fidelity(zx_optimized_circuit)
                    
                    # Determine if ZX optimized circuit is better
                    if self._is_better_circuit(zx_depth, zx_gates, zx_fidelity,
                                             current_depth, current_gates, current_fidelity):
                        print(f"ZX optimization improved the circuit:")
                        print(f"  Depth: {current_depth} -> {zx_depth}")
                        print(f"  Gates: {current_gates} -> {zx_gates}")
                        print(f"  Fidelity: {current_fidelity:.6f} -> {zx_fidelity:.6f}")
                        
                        # Free old circuit from memory
                        del optimized_circuit
                        gc.collect()
                        
                        optimized_circuit = zx_optimized_circuit
                        current_depth = zx_depth
                        current_gates = zx_gates
                        current_fidelity = zx_fidelity
                    else:
                        print("ZX optimization did not improve the circuit")
                        
                        # Free the unused ZX circuit from memory
                        del zx_optimized_circuit
                        gc.collect()
                    
                except Exception as e:
                    print(f"ZX optimization failed: {e}")
                    if self.debug_mode:
                        traceback.print_exc()
                        
            # Apply RL optimization if enabled
            if self.use_rl_optimization:
                print("\nApplying RL-based optimization...")
                
                try:
                    # Use the RL optimizer to optimize the circuit
                    rl_optimized_circuit = self.rl_optimizer.optimize_circuit(
                        optimized_circuit.copy(),
                        coupling_map=coupling_map
                    )
                    
                    # Check if RL optimization helped
                    rl_metrics = self.compiler.get_circuit_metrics(rl_optimized_circuit)
                    rl_depth = rl_metrics["depth"]
                    rl_gates = sum(rl_metrics["gate_counts"].values())
                    
                    # Clear metrics from memory
                    del rl_metrics
                    gc.collect()
                    
                    rl_fidelity = self.estimate_physical_fidelity(rl_optimized_circuit)
                    
                    # Determine if RL optimized circuit is better
                    if self._is_better_circuit(rl_depth, rl_gates, rl_fidelity,
                                             current_depth, current_gates, current_fidelity):
                        print(f"RL optimization improved the circuit:")
                        print(f"  Depth: {current_depth} -> {rl_depth}")
                        print(f"  Gates: {current_gates} -> {rl_gates}")
                        print(f"  Fidelity: {current_fidelity:.6f} -> {rl_fidelity:.6f}")
                        
                        # Free old circuit from memory
                        del optimized_circuit
                        gc.collect()
                        
                        optimized_circuit = rl_optimized_circuit
                        current_depth = rl_depth
                        current_gates = rl_gates
                        current_fidelity = rl_fidelity
                    else:
                        print("RL optimization did not improve the circuit")
                        
                        # Free the unused RL circuit from memory
                        del rl_optimized_circuit
                        gc.collect()
                    
                except Exception as e:
                    print(f"RL optimization failed: {e}")
                    if self.debug_mode:
                        traceback.print_exc()
            
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
                optimization_level=3  # Use high optimization level
            )
            
            # Validate that the mapped circuit respects coupling constraints
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
    
    def _optimize_circuit(self, circuit, toffoli_gates, output_qubits, input_qubits,
                         num_qubits, coupling_map, basis_gates, target_fidelity,
                         toffoli_type=None, use_ancilla=True):
        """
        Apply optimization techniques to reduce circuit depth and improve fidelity.
        
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
            gate_processor = CircuitGateProcessor(debug_mode=self.debug_mode)
            fixed_circuit = gate_processor._process_circuit_safely(working_circuit, coupling_map)
            working_circuit = fixed_circuit
            
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
            
            # 2. Apply ZX calculus optimization if enabled
            if self.use_zx_optimization:
                try:
                    # Determine target fidelity based on strategy
                    zx_target_fidelity = self.min_fidelity if self.strategy == OptimizationStrategy.ULTRA_DEPTH_REDUCTION else self.target_fidelity
                    
                    zx_circuit = self.zx_optimizer.optimize_with_zx(
                        working_circuit.copy()
                    )
                    
                    # Measure ZX-optimized circuit
                    zx_metrics = self.compiler.get_circuit_metrics(zx_circuit)
                    zx_depth = zx_metrics["depth"]
                    zx_gates = sum(zx_metrics["gate_counts"].values())
                    zx_fidelity = self.estimate_physical_fidelity(zx_circuit)
                    
                    # Clear metrics from memory
                    del zx_metrics
                    gc.collect()
                    
                    # Check if this is better
                    if self._is_better_circuit(zx_depth, zx_gates, zx_fidelity,
                                            best_depth, best_gate_count, best_fidelity):
                        best_circuit = zx_circuit.copy()
                        best_depth = zx_depth
                        best_gate_count = zx_gates
                        best_fidelity = zx_fidelity
                        if self.debug_mode:
                            print(f"ZX calculus optimization improved metrics: depth={best_depth}, gates={best_gate_count}, fidelity={best_fidelity:.6f}")
                except Exception as e:
                    if self.debug_mode:
                        print(f"ZX calculus optimization failed: {e}")
            
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
            
            # 4. Try approximate Toffoli implementations for depth reduction
            # This is especially useful for ULTRA_DEPTH_REDUCTION strategy
            if self.strategy == OptimizationStrategy.ULTRA_DEPTH_REDUCTION or self.strategy == OptimizationStrategy.DEPTH_FIDELITY_BALANCE:
                try:
                    # If using ZX optimizer, apply approximate Toffoli implementations
                    if self.use_zx_optimization:
                        # Set approximation level based on strategy
                        if self.strategy == OptimizationStrategy.ULTRA_DEPTH_REDUCTION:
                            approximation_level = 0.8  # More aggressive approximation
                        else:  # DEPTH_FIDELITY_BALANCE
                            approximation_level = 0.4 * (self.depth_weight / self.fidelity_weight)
                            approximation_level = min(0.8, max(0.2, approximation_level))  # Limit between 0.2-0.8
                        
                        approx_circuit = self.zx_optimizer.approximate_toffoli(
                            best_circuit.copy(),
                            approximation_level=approximation_level
                        )
                    else:
                        # Create a circuit with approximate Toffoli gates using our basic method
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
                            print(f"Approximate Toffoli implementation improved metrics: depth={best_depth}, gates={best_gate_count}, fidelity={best_fidelity:.6f}")
                except Exception as e:
                    if self.debug_mode:
                        print(f"Approximate Toffoli optimization failed: {e}")
            
            # 5. Use RL-based optimization if enabled
            if self.use_rl_optimization:
                try:
                    rl_circuit = self.rl_optimizer.optimize_circuit(
                        best_circuit.copy(),
                        coupling_map=coupling_map
                    )
                    
                    # Measure RL-optimized circuit
                    rl_metrics = self.compiler.get_circuit_metrics(rl_circuit)
                    rl_depth = rl_metrics["depth"]
                    rl_gates = sum(rl_metrics["gate_counts"].values())
                    rl_fidelity = self.estimate_physical_fidelity(rl_circuit)
                    
                    # Clear metrics from memory
                    del rl_metrics
                    gc.collect()
                    
                    # Check if this is better
                    if self._is_better_circuit(rl_depth, rl_gates, rl_fidelity,
                                            best_depth, best_gate_count, best_fidelity):
                        best_circuit = rl_circuit.copy()
                        best_depth = rl_depth
                        best_gate_count = rl_gates
                        best_fidelity = rl_fidelity
                        if self.debug_mode:
                            print(f"RL optimization improved metrics: depth={best_depth}, gates={best_gate_count}, fidelity={best_fidelity:.6f}")
                except Exception as e:
                    if self.debug_mode:
                        print(f"RL optimization failed: {e}")
            
            # 6. Final optimization pass
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
            
            # Return the best circuit we found
            return best_circuit
            
        except Exception as e:
            print(f"Error in circuit optimization: {e}")
            import traceback
            traceback.print_exc()
            return circuit  # Return the original circuit on error