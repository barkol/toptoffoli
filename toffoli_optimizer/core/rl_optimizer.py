"""
Reinforcement Learning Based Toffoli Optimizer

This module provides optimization using reinforcement learning techniques
for quantum circuit depth reduction with controlled fidelity trade-offs.
"""

import os
import time
import numpy as np
import pickle
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any, Union

# Try to import Qiskit
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Instruction, Parameter
    from qiskit.converters import circuit_to_dag, dag_to_circuit
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Unroller, Optimize1qGates
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. RL optimization will be limited.")

class GateTransformation(Enum):
    """Types of gate transformations that can be applied during RL optimization"""
    DECOMPOSE_TOFFOLI = auto()        # Decompose Toffoli into basic gates
    REDUCE_T_COUNT = auto()           # Reduce T gate count
    CANCEL_ADJACENT_GATES = auto()    # Cancel adjacent gates that simplify to identity
    COMMUTE_GATES = auto()            # Commute gates to enable other optimizations
    APPROXIMATE_TOFFOLI = auto()      # Use approximate Toffoli implementations
    MERGE_SINGLE_QUBIT = auto()       # Merge adjacent single-qubit gates
    REASSIGN_QUBITS = auto()          # Reassign qubit indices for better depth
    SWAP_CONTROLS = auto()            # Swap control qubits for better routing

class QLearningState:
    """
    Represents the state of a quantum circuit for RL optimization.
    
    This provides a tensor representation of a circuit that can be used
    as input to a Q-learning or policy-based RL algorithm.
    """
    
    def __init__(self, depth: int, gate_count: int, cx_count: int,
                 t_count: int, fidelity: float):
        """
        Initialize the QLearningState with circuit metrics.
        
        Args:
            depth: Circuit depth
            gate_count: Total gate count
            cx_count: Number of CNOT gates
            t_count: Number of T gates
            fidelity: Estimated circuit fidelity
        """
        self.depth = depth
        self.gate_count = gate_count
        self.cx_count = cx_count
        self.t_count = t_count
        self.fidelity = fidelity
        
        # Create a tensor representation of the state
        self.tensor = np.array([
            depth,
            gate_count,
            cx_count,
            t_count,
            fidelity
        ])
    
    @staticmethod
    def from_circuit(circuit: QuantumCircuit, coupling_map: Optional[List] = None) -> 'QLearningState':
        """
        Extract a state representation from a quantum circuit.
        
        Args:
            circuit: Quantum circuit to extract state from
            coupling_map: Optional coupling map for fidelity estimation
            
        Returns:
            QLearningState: State representation
        """
        if circuit is None:
            return QLearningState(0, 0, 0, 0, 0.0)
            
        try:
            # Extract basic metrics
            depth = circuit.depth()
            gate_count = len(circuit.data)
            
            # Count gate types
            gate_counts = {}
            try:
                gate_counts = circuit.count_ops()
            except:
                # Manually count if method not available
                gate_counts = {}
                for inst in circuit.data:
                    name = inst.operation.name if hasattr(inst.operation, 'name') else 'unknown'
                    gate_counts[name] = gate_counts.get(name, 0) + 1
            
            # Extract specific gate counts
            cx_count = gate_counts.get('cx', 0)
            t_count = gate_counts.get('t', 0) + gate_counts.get('tdg', 0)
            
            # Estimate fidelity
            fidelity = 1.0
            # Simple fidelity model: each CNOT reduces by 1%, each T by 0.1%
            fidelity = max(0.0, fidelity - (0.01 * cx_count) - (0.001 * t_count))
            
            return QLearningState(depth, gate_count, cx_count, t_count, fidelity)
        except Exception as e:
            print(f"Error creating circuit state: {e}")
            return QLearningState(0, 0, 0, 0, 0.0)

class RLToffoliOptimizer:
    """
    Optimizer using reinforcement learning techniques for quantum circuit optimization.
    
    This optimizer applies a sequence of transformations learned through reinforcement
    learning to reduce circuit depth while maintaining fidelity above a target threshold.
    """
    
    def __init__(self, q_table_file: Optional[str] = None,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 exploration_rate: float = 0.2,
                 target_fidelity: float = 0.95,
                 max_steps: int = 20,
                 debug_mode: bool = False):
        """
        Initialize the RL optimizer with parameters.
        
        Args:
            q_table_file: Path to a pre-trained Q-table file
            learning_rate: Learning rate for Q-learning
            discount_factor: Discount factor for future rewards
            exploration_rate: Probability of exploring vs exploiting
            target_fidelity: Target circuit fidelity (0.0-1.0)
            max_steps: Maximum optimization steps
            debug_mode: Whether to print debug information
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.target_fidelity = target_fidelity
        self.max_steps = max_steps
        self.debug_mode = debug_mode
        
        # Initialize Q-table
        self.q_table = self._load_q_table(q_table_file)
        
        # Define available actions
        self.actions = list(GateTransformation)
        
        if self.debug_mode:
            print(f"RLToffoliOptimizer initialized with {len(self.actions)} possible actions")
            print(f"Target fidelity: {self.target_fidelity}")
            
    def _load_q_table(self, q_table_file: Optional[str]) -> Dict:
        """
        Load a Q-table from a file or create a new one.
        
        Args:
            q_table_file: Path to pickle file containing Q-table
            
        Returns:
            dict: Q-table mapping states to action values
        """
        if q_table_file and os.path.exists(q_table_file):
            try:
                with open(q_table_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                if self.debug_mode:
                    print(f"Error loading Q-table: {e}")
        
        # Create a new Q-table
        return {}
        
    def _save_q_table(self, filename: str) -> bool:
        """
        Save the Q-table to a file.
        
        Args:
            filename: Path to save the Q-table to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table, f)
            return True
        except Exception as e:
            if self.debug_mode:
                print(f"Error saving Q-table: {e}")
            return False
        
    def _get_state_key(self, state: QLearningState) -> str:
        """
        Convert a state to a string key for the Q-table.
        
        Args:
            state: State to convert
            
        Returns:
            str: Key for the Q-table
        """
        # Discretize continuous values to limit state space
        depth_bin = min(20, max(0, int(state.depth / 10)))
        gate_bin = min(20, max(0, int(state.gate_count / 20)))
        cx_bin = min(10, max(0, int(state.cx_count / 10)))
        t_bin = min(10, max(0, int(state.t_count / 5)))
        fidelity_bin = min(10, max(0, int(state.fidelity * 10)))
        
        return f"{depth_bin}_{gate_bin}_{cx_bin}_{t_bin}_{fidelity_bin}"
        
    def _get_q_values(self, state: QLearningState) -> np.ndarray:
        """
        Get Q-values for a state.
        
        Args:
            state: Current state
            
        Returns:
            np.ndarray: Q-values for each action
        """
        state_key = self._get_state_key(state)
        
        if state_key not in self.q_table:
            # Initialize with zeros
            self.q_table[state_key] = np.zeros(len(self.actions))
            
        return self.q_table[state_key]
        
    def _choose_action(self, state: QLearningState) -> GateTransformation:
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            GateTransformation: Chosen action
        """
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return np.random.choice(self.actions)
            
        # Exploitation: best action from Q-table
        q_values = self._get_q_values(state)
        return self.actions[np.argmax(q_values)]
        
    def _apply_action(self, circuit: QuantumCircuit,
                      action: GateTransformation) -> QuantumCircuit:
        """
        Apply a transformation action to a circuit.
        
        Args:
            circuit: Circuit to transform
            action: Transformation to apply
            
        Returns:
            QuantumCircuit: Transformed circuit
        """
        if not QISKIT_AVAILABLE or circuit is None:
            return circuit
            
        # Create a copy of the circuit
        result = circuit.copy()
        
        try:
            if action == GateTransformation.DECOMPOSE_TOFFOLI:
                # Decompose Toffoli gates into more elementary gates
                dag = circuit_to_dag(result)
                for node in dag.op_nodes():
                    if node.name == 'ccx':
                        dag.substitute_node_with_dag(node, circuit_to_dag(self._decompose_toffoli()))
                result = dag_to_circuit(dag)
                
            elif action == GateTransformation.REDUCE_T_COUNT:
                # Apply T-count reduction optimizations
                pass_manager = PassManager()
                pass_manager.append(Unroller(['u', 'cx']))
                pass_manager.append(Optimize1qGates())
                result = pass_manager.run(result)
                
            elif action == GateTransformation.CANCEL_ADJACENT_GATES:
                # Find and cancel adjacent gates that simplify
                # For now, just use the transpiler optimization
                result = transpile(result, optimization_level=1)
                
            elif action == GateTransformation.COMMUTE_GATES:
                # Try to commute gates to enable cancellations
                # For now, just use the transpiler optimization
                result = transpile(result, optimization_level=2)
                
            elif action == GateTransformation.APPROXIMATE_TOFFOLI:
                # Replace Toffoli gates with approximate versions
                result = self._approximate_toffolis(result)
                
            elif action == GateTransformation.MERGE_SINGLE_QUBIT:
                # Merge adjacent single-qubit gates
                pass_manager = PassManager()
                pass_manager.append(Optimize1qGates())
                result = pass_manager.run(result)
                
            elif action == GateTransformation.REASSIGN_QUBITS:
                # Try different qubit assignments
                # For now, just use a simple transpile
                result = transpile(result, optimization_level=2)
                
            elif action == GateTransformation.SWAP_CONTROLS:
                # Swap control qubits where advantageous
                # This would require a custom pass in a real implementation
                pass
            
            return result
            
        except Exception as e:
            if self.debug_mode:
                print(f"Error applying action {action}: {e}")
            return circuit
    
    def _decompose_toffoli(self) -> QuantumCircuit:
        """
        Create a circuit that implements Toffoli decomposition.
        
        Returns:
            QuantumCircuit: Circuit implementing a Toffoli gate
        """
        circuit = QuantumCircuit(3)
        
        # Decomposition using standard gates
        circuit.h(2)
        circuit.cx(1, 2)
        circuit.tdg(2)
        circuit.cx(0, 2)
        circuit.t(2)
        circuit.cx(1, 2)
        circuit.tdg(2)
        circuit.cx(0, 2)
        circuit.t(1)
        circuit.t(2)
        circuit.h(2)
        circuit.cx(0, 1)
        circuit.t(0)
        circuit.tdg(1)
        circuit.cx(0, 1)
        
        return circuit
    
    def _approximate_toffolis(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Replace Toffoli gates with approximate versions.
        
        Args:
            circuit: Circuit to transform
            
        Returns:
            QuantumCircuit: Circuit with approximate Toffoli gates
        """
        if not QISKIT_AVAILABLE or circuit is None:
            return circuit
            
        # Create a new circuit
        result = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        
        # Process each gate
        for inst in circuit.data:
            if inst.operation.name == 'ccx':
                # Get qubit indices
                control1 = inst.qubits[0].index
                control2 = inst.qubits[1].index
                target = inst.qubits[2].index
                
                # Use an approximate Toffoli implementation
                result.h(target)
                result.cx(control1, target)
                result.cx(control2, target)
                result.h(target)
            else:
                # Copy the gate as-is
                result.append(inst.operation, inst.qubits, inst.clbits)
        
        return result
    
    def _calculate_reward(self, old_state: QLearningState,
                         new_state: QLearningState) -> float:
        """
        Calculate the reward for a state transition.
        
        Args:
            old_state: State before action
            new_state: State after action
            
        Returns:
            float: Reward value
        """
        # Reward for depth reduction
        depth_reward = (old_state.depth - new_state.depth) * 2.0
        
        # Reward for gate count reduction
        gate_reward = (old_state.gate_count - new_state.gate_count) * 0.5
        
        # Penalty for fidelity reduction
        fidelity_penalty = 0
        if new_state.fidelity < self.target_fidelity:
            fidelity_penalty = (self.target_fidelity - new_state.fidelity) * 20.0
        
        # Calculate total reward
        total_reward = depth_reward + gate_reward - fidelity_penalty
        
        return total_reward
        
    def _update_q_table(self, state: QLearningState,
                       action: GateTransformation,
                       reward: float,
                       next_state: QLearningState) -> None:
        """
        Update the Q-table based on the reward.
        
        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
        """
        state_key = self._get_state_key(state)
        action_idx = self.actions.index(action)
        
        # Get current Q-value
        current_q = self.q_table.get(state_key, np.zeros(len(self.actions)))[action_idx]
        
        # Get max Q-value for next state
        next_q_values = self._get_q_values(next_state)
        max_next_q = np.max(next_q_values)
        
        # Q-learning update formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Update Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        self.q_table[state_key][action_idx] = new_q
        
    def optimize_circuit(self, circuit: QuantumCircuit,
                        coupling_map: Optional[List] = None,
                        max_steps: Optional[int] = None) -> QuantumCircuit:
        """
        Optimize a circuit using RL techniques.
        
        Args:
            circuit: The circuit to optimize
            coupling_map: Optional coupling map constraints
            max_steps: Maximum optimization steps (overrides instance value)
            
        Returns:
            QuantumCircuit: Optimized circuit
        """
        if not QISKIT_AVAILABLE or circuit is None:
            return circuit
            
        # Use provided max_steps or instance default
        steps = max_steps if max_steps is not None else self.max_steps
        
        if self.debug_mode:
            print(f"RL optimization with target fidelity {self.target_fidelity}, max steps {steps}")
            print(f"Input circuit depth: {circuit.depth()}, gate count: {len(circuit.data)}")
        
        # Keep track of the best circuit found
        best_circuit = circuit.copy()
        best_depth = circuit.depth()
        
        # Current working circuit
        current_circuit = circuit.copy()
        
        # Optimization loop
        for step in range(steps):
            # Get current state
            current_state = QLearningState.from_circuit(current_circuit, coupling_map)
            
            # Choose an action
            action = self._choose_action(current_state)
            
            # Apply the action
            if self.debug_mode:
                print(f"Step {step+1}: Applying action {action.name}")
                
            new_circuit = self._apply_action(current_circuit, action)
            
            # Get new state
            new_state = QLearningState.from_circuit(new_circuit, coupling_map)
            
            # Check if the optimization improves the circuit
            if new_state.depth < current_state.depth and new_state.fidelity >= self.target_fidelity:
                if self.debug_mode:
                    print(f"  Depth improved: {current_state.depth} -> {new_state.depth}")
                    print(f"  Fidelity: {new_state.fidelity:.4f}")
                
                current_circuit = new_circuit
                
                # Update best circuit if this is better
                if new_state.depth < best_depth:
                    best_circuit = new_circuit.copy()
                    best_depth = new_state.depth
                    
                    if self.debug_mode:
                        print(f"  New best depth: {best_depth}")
            
            # Calculate reward and update Q-table (for learning mode)
            reward = self._calculate_reward(current_state, new_state)
            self._update_q_table(current_state, action, reward, new_state)
            
            # Early termination if we can't make further progress
            if step > 5 and current_state.depth == new_state.depth:
                if self.debug_mode:
                    print(f"Early termination at step {step+1}: No progress in depth")
                break
        
        # Final transpile to clean up the circuit
        final_circuit = transpile(best_circuit, optimization_level=1)
        
        if self.debug_mode:
            print(f"Optimization complete:")
            print(f"  Original depth: {circuit.depth()}")
            print(f"  Final depth: {final_circuit.depth()}")
            print(f"  Depth reduction: {(circuit.depth() - final_circuit.depth()) / circuit.depth() * 100:.2f}%")
            
        return final_circuit
            
    def train(self, training_circuits: List[QuantumCircuit],
             coupling_map: Optional[List] = None,
             num_episodes: int = 100,
             save_interval: int = 10,
             output_dir: str = "rl_training") -> Dict:
        """
        Train the RL optimizer on a set of circuits.
        
        Args:
            training_circuits: List of circuits to train on
            coupling_map: Optional coupling map constraints
            num_episodes: Number of training episodes
            save_interval: How often to save the Q-table
            output_dir: Directory to save training results
            
        Returns:
            dict: Training statistics
        """
        if not QISKIT_AVAILABLE or not training_circuits:
            return {"error": "Qiskit not available or no training circuits provided"}
            
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if self.debug_mode:
            print(f"Training RL optimizer on {len(training_circuits)} circuits for {num_episodes} episodes")
            
        # Tracking metrics
        episode_rewards = []
        depth_reductions = []
        
        # Save initial Q-table
        self._save_q_table(os.path.join(output_dir, "q_table_initial.pkl"))
        
        # Training loop
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Randomly select a training circuit
            circuit_idx = np.random.randint(0, len(training_circuits))
            circuit = training_circuits[circuit_idx].copy()
            
            if self.debug_mode:
                print(f"Episode {episode+1}/{num_episodes}, Circuit {circuit_idx+1}/{len(training_circuits)}")
                print(f"  Initial depth: {circuit.depth()}")
            
            # Track episode reward
            total_reward = 0
            
            # Initial state
            current_circuit = circuit.copy()
            current_state = QLearningState.from_circuit(current_circuit, coupling_map)
            
            # Episode loop (optimization steps)
            for step in range(self.max_steps):
                # Choose an action with higher exploration during training
                saved_exploration = self.exploration_rate
                self.exploration_rate = max(0.1, saved_exploration * (1 - episode / num_episodes))
                action = self._choose_action(current_state)
                self.exploration_rate = saved_exploration
                
                # Apply the action
                new_circuit = self._apply_action(current_circuit, action)
                
                # Get new state
                new_state = QLearningState.from_circuit(new_circuit, coupling_map)
                
                # Calculate reward
                reward = self._calculate_reward(current_state, new_state)
                total_reward += reward
                
                # Update Q-table
                self._update_q_table(current_state, action, reward, new_state)
                
                # Move to new state if it's better
                if new_state.depth <= current_state.depth and new_state.fidelity >= self.target_fidelity:
                    current_circuit = new_circuit
                    current_state = new_state
                
                # Early termination if no progress
                if step > 5 and reward <= 0:
                    break
            
            # Record metrics
            episode_rewards.append(total_reward)
            depth_reduction = (circuit.depth() - current_state.depth) / circuit.depth() * 100 if circuit.depth() > 0 else 0
            depth_reductions.append(depth_reduction)
            
            if self.debug_mode:
                print(f"  Final depth: {current_state.depth}")
                print(f"  Depth reduction: {depth_reduction:.2f}%")
                print(f"  Total reward: {total_reward:.2f}")
            
            # Save Q-table periodically
            if (episode + 1) % save_interval == 0:
                self._save_q_table(os.path.join(output_dir, f"q_table_episode_{episode+1}.pkl"))
        
        # Save final Q-table
        self._save_q_table(os.path.join(output_dir, "q_table_final.pkl"))
        
        # Calculate training stats
        training_time = time.time() - start_time
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        avg_depth_reduction = np.mean(depth_reductions) if depth_reductions else 0
        
        # Compile training results
        results = {
            "num_episodes": num_episodes,
            "num_circuits": len(training_circuits),
            "training_time": training_time,
            "avg_reward": avg_reward,
            "avg_depth_reduction": avg_depth_reduction,
            "final_q_table_size": len(self.q_table),
            "target_fidelity": self.target_fidelity
        }
        
        if self.debug_mode:
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Average depth reduction: {avg_depth_reduction:.2f}%")
            print(f"Q-table size: {len(self.q_table)} states")
        
        return results
    
    def get_optimization_trace(self, circuit: QuantumCircuit,
                             coupling_map: Optional[List] = None,
                             max_steps: Optional[int] = None) -> Dict:
        """
        Get detailed information about the optimization process for analysis.
        
        Args:
            circuit: The circuit to optimize
            coupling_map: Optional coupling map constraints
            max_steps: Maximum optimization steps
            
        Returns:
            dict: Detailed optimization trace with metrics at each step
        """
        if not QISKIT_AVAILABLE or circuit is None:
            return {"error": "Qiskit not available or invalid circuit"}
            
        # Use provided max_steps or instance default
        steps = max_steps if max_steps is not None else self.max_steps
        
        # Trace data
        trace = {
            "initial_depth": circuit.depth(),
            "initial_gate_count": len(circuit.data),
            "steps": [],
            "final_depth": 0,
            "depth_reduction": 0,
            "runtime": 0
        }
        
        start_time = time.time()
        
        # Optimization process with tracing
        current_circuit = circuit.copy()
        
        for step in range(steps):
            # Get current state
            current_state = QLearningState.from_circuit(current_circuit, coupling_map)
            
            # Choose an action
            action = self._choose_action(current_state)
            
            # Apply the action
            new_circuit = self._apply_action(current_circuit, action)
            
            # Get new state
            new_state = QLearningState.from_circuit(new_circuit, coupling_map)
            
            # Calculate reward
            reward = self._calculate_reward(current_state, new_state)
            
            # Record step data
            step_data = {
                "step": step + 1,
                "action": action.name,
                "depth_before": current_state.depth,
                "depth_after": new_state.depth,
                "fidelity": new_state.fidelity,
                "reward": reward
            }
            trace["steps"].append(step_data)
            
            # Update current circuit if it's better
            if new_state.depth <= current_state.depth and new_state.fidelity >= self.target_fidelity:
                current_circuit = new_circuit
            
            # Early termination if no progress
            if step > 5 and reward <= 0:
                break
        
        # Final circuit metrics
        final_state = QLearningState.from_circuit(current_circuit, coupling_map)
        trace["final_depth"] = final_state.depth
        trace["final_gate_count"] = final_state.gate_count
        trace["final_fidelity"] = final_state.fidelity
        
        # Calculate overall metrics
        runtime = time.time() - start_time
        trace["runtime"] = runtime
        trace["depth_reduction"] = (circuit.depth() - final_state.depth) / circuit.depth() * 100 if circuit.depth() > 0 else 0
        
        return trace
