"""
Reinforcement Learning Based Toffoli Optimizer Integration

This module provides the integration layer for using the reinforcement learning approach 
to circuit optimization, focusing on depth reduction while accepting controlled 
fidelity trade-offs.
"""

import os
import time
import numpy as np
import pickle
import gc
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum, auto
import json

# Import from parent module if available
try:
    from ..utils.circuit_utils import validate_physical_circuit, estimate_fidelity, get_default_coupling_map
    from ..core.optimizer import CircuitGateProcessor
except ImportError:
    # Direct imports for standalone usage
    try:
        from toffoli_optimizer.utils.circuit_utils import validate_physical_circuit, estimate_fidelity, get_default_coupling_map
        from toffoli_optimizer.core.optimizer import CircuitGateProcessor
    except ImportError:
        print("Warning: Could not import required utilities")

# Import Qiskit with version compatibility
try:
    from qiskit import QuantumCircuit, transpile
    # Try imports for Qiskit 2.0+
    try:
        from qiskit.qasm2 import dumps, loads
        from qiskit.transpiler import PassManager, CouplingMap
        QISKIT_2_AVAILABLE = True
    except ImportError:
        QISKIT_2_AVAILABLE = False
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QISKIT_2_AVAILABLE = False

# Try to import the RL optimizer
try:
    from ..core.rl_optimizer import RLToffoliOptimizer, GateTransformation, QLearningState
    RL_OPTIMIZER_AVAILABLE = True
except ImportError:
    # Define fallback classes if the import fails
    class GateTransformation(Enum):
        """Types of gate transformations for optimization"""
        DECOMPOSE_TOFFOLI = auto()
        REDUCE_T_COUNT = auto()
        CANCEL_ADJACENT_GATES = auto()
        COMMUTE_GATES = auto()
        APPROXIMATE_TOFFOLI = auto()
        MERGE_SINGLE_QUBIT = auto()
        REASSIGN_QUBITS = auto()
        SWAP_CONTROLS = auto()
    
    class QLearningState:
        """Placeholder for QLearningState"""
        @staticmethod
        def from_circuit(circuit, coupling_map=None):
            return None
            
    class RLToffoliOptimizer:
        """Placeholder for RLToffoliOptimizer"""
        def __init__(self, q_table_file=None, target_fidelity=0.95, max_steps=20, debug_mode=False):
            self.debug_mode = debug_mode
            
        def optimize_circuit(self, circuit, coupling_map=None, max_steps=None):
            return circuit
            
    RL_OPTIMIZER_AVAILABLE = False
    print("Warning: RL optimizer not available. Using fallback methods.")

class RLOptimizationWrapper:
    """
    Wrapper for the RL-based optimizer to provide consistent interface with other optimizers.
    
    This class provides methods to use the RLToffoliOptimizer with the rest of the
    Toffoli Optimizer framework, adapting the interface to match other optimization approaches.
    """
    
    def __init__(self, q_table_file: Optional[str] = None, 
                 target_fidelity: float = 0.95,
                 max_steps: int = 30, 
                 debug_mode: bool = False):
        """
        Initialize the RL optimization wrapper.
        
        Args:
            q_table_file: Path to a pre-trained Q-table file
            target_fidelity: Target circuit fidelity
            max_steps: Maximum optimization steps
            debug_mode: Whether to print debug information
        """
        self.debug_mode = debug_mode
        self.target_fidelity = target_fidelity
        self.max_steps = max_steps
        
        if not RL_OPTIMIZER_AVAILABLE:
            self.optimizer = None
            if debug_mode:
                print("RL optimizer not available. Using fallback methods.")
            return
        
        # Initialize the RL optimizer
        self.optimizer = RLToffoliOptimizer(
            q_table_file=q_table_file,
            learning_rate=0.1,
            discount_factor=0.9,
            exploration_rate=0.1,  # Lower exploration rate for inference
            max_steps=max_steps,
            target_fidelity=target_fidelity,
            debug_mode=debug_mode
        )
        
        if debug_mode:
            print(f"Initialized RL optimizer with {max_steps} max steps and {target_fidelity} target fidelity")
    
    def optimize_circuit(self, circuit: QuantumCircuit,
                        coupling_map: Optional[List] = None,
                        max_steps: Optional[int] = None) -> QuantumCircuit:
        """
        Optimize a circuit using the RL-based optimizer.
        
        Args:
            circuit: Quantum circuit to optimize
            coupling_map: Optional coupling map for hardware constraints
            max_steps: Maximum number of optimization steps (overrides instance value)
            
        Returns:
            QuantumCircuit: Optimized circuit
        """
        if not RL_OPTIMIZER_AVAILABLE or self.optimizer is None:
            return self._fallback_optimize(circuit, coupling_map)
        
        if circuit is None:
            return None
        
        if not QISKIT_AVAILABLE:
            return circuit
        
        try:
            # Use provided max_steps or instance default
            steps = max_steps if max_steps is not None else self.max_steps
            
            # Optimize the circuit using the RL optimizer
            start_time = time.time()
            
            # Process the circuit to make it safe for the RL optimizer
            gate_processor = CircuitGateProcessor(debug_mode=self.debug_mode)
            safe_circuit = gate_processor._process_circuit_safely(circuit, coupling_map)
            
            # Run the RL optimization
            optimized_circuit = self.optimizer.optimize_circuit(
                safe_circuit, 
                coupling_map=coupling_map,
                max_steps=steps
            )
            
            if self.debug_mode:
                duration = time.time() - start_time
                print(f"RL optimization completed in {duration:.2f} seconds")
                print(f"Original depth: {circuit.depth()}, Optimized depth: {optimized_circuit.depth()}")
                print(f"Depth reduction: {(1 - optimized_circuit.depth() / circuit.depth()) * 100:.2f}%")
            
            return optimized_circuit
        
        except Exception as e:
            if self.debug_mode:
                print(f"Error in RL optimization: {e}")
                import traceback
                traceback.print_exc()
            
            # Fallback to standard optimization
            return self._fallback_optimize(circuit, coupling_map)
    
    def _fallback_optimize(self, circuit: QuantumCircuit, 
                          coupling_map: Optional[List] = None) -> QuantumCircuit:
        """
        Fallback optimization method when the RL optimizer is not available or fails.
        
        Args:
            circuit: Quantum circuit to optimize
            coupling_map: Optional coupling map for hardware constraints
            
        Returns:
            QuantumCircuit: Optimized circuit
        """
        if not QISKIT_AVAILABLE:
            return circuit
        
        try:
            # Use Qiskit's transpiler as a fallback
            optimized = transpile(
                circuit,
                coupling_map=coupling_map,
                optimization_level=3,
                layout_method='sabre' if coupling_map is not None else None
            )
            
            if self.debug_mode:
                print(f"Fallback optimization: Depth {circuit.depth()} -> {optimized.depth()} "
                     f"({(1 - optimized.depth() / circuit.depth()) * 100:.2f}% reduction)")
            
            return optimized
        
        except Exception as e:
            if self.debug_mode:
                print(f"Error in fallback optimization: {e}")
            
            # Return the original circuit as last resort
            return circuit
    
    def get_optimization_trace(self, circuit: QuantumCircuit,
                             coupling_map: Optional[List] = None) -> Dict:
        """
        Get a detailed trace of the optimization process for analysis.
        
        Args:
            circuit: Quantum circuit to optimize
            coupling_map: Optional coupling map for hardware constraints
            
        Returns:
            dict: Detailed optimization trace and metrics
        """
        if not RL_OPTIMIZER_AVAILABLE or self.optimizer is None:
            return {"available": False, "message": "RL optimizer not available"}
        
        if circuit is None:
            return {"error": "No circuit provided"}
        
        if not QISKIT_AVAILABLE:
            return {"error": "Qiskit not available"}
        
        try:
            # Get the optimization trace from the RL optimizer
            trace = self.optimizer.get_optimization_trace(
                circuit, 
                coupling_map=coupling_map,
                max_steps=self.max_steps
            )
            
            return trace
        
        except Exception as e:
            if self.debug_mode:
                print(f"Error getting optimization trace: {e}")
                import traceback
                traceback.print_exc()
            
            return {"error": str(e)}
    
    def train_optimizer(self, circuits: List[QuantumCircuit], 
                       coupling_map: Optional[List] = None,
                       num_episodes: int = 100,
                       save_file: Optional[str] = "rl_toffoli_qtable.pkl") -> None:
        """
        Train the RL optimizer on a set of circuits to improve its performance.
        
        Args:
            circuits: List of circuits for training
            coupling_map: Optional coupling map for hardware constraints
            num_episodes: Number of training episodes
            save_file: File to save the trained Q-table to
        """
        if not RL_OPTIMIZER_AVAILABLE or self.optimizer is None:
            print("RL optimizer not available. Cannot train.")
            return
        
        if not circuits:
            print("No training circuits provided.")
            return
        
        print(f"Training RL optimizer on {len(circuits)} circuits for {num_episodes} episodes")
        
        try:
            # Train the RL optimizer
            self.optimizer.train(
                training_circuits=circuits,
                coupling_map=coupling_map,
                num_episodes=num_episodes,
                save_interval=max(1, num_episodes // 10)  # Save every ~10% of episodes
            )
            
            print(f"Training completed. Q-table saved to {save_file}")
            
        except Exception as e:
            print(f"Error training RL optimizer: {e}")
            import traceback
            traceback.print_exc()
