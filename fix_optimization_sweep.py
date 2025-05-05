#!/usr/bin/env python3
"""
Fixed Toffoli Optimizer Optimization Sweep Script

This script reports only physical depth metrics for all optimizers,
ensuring consistent reporting across all optimization methods.
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

# Add current directory to path to fix import issues
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print(f"Python path: {sys.path}")

# Check if Qiskit is available
try:
    import qiskit
    print(f"Qiskit found! Version: {qiskit.__version__}")
except ImportError as e:
    print(f"Failed to import Qiskit: {e}")

# Import the ShiftExperiment_FlowControl for circuit generation
try:
    from SynchronizingCircuit import ShiftExperiment_FlowControl
    print("Successfully imported ShiftExperiment_FlowControl")
except ImportError:
    try:
        from scripts.SynchronizingCircuit import ShiftExperiment_FlowControl
        print("Successfully imported ShiftExperiment_FlowControl from scripts directory")
    except ImportError as e:
        print(f"Error importing ShiftExperiment_FlowControl: {e}")
        print("Please ensure scripts/SynchronizingCircuit.py is accessible")
        sys.exit(1)

# First, let's check what modules are available in your repository
available_modules = {}

# Try to import the compiler
try:
    from toffoli_optimizer.core.compiler import ToffoliCompiler, ToffoliType
    available_modules["compiler"] = True
    print("Successfully imported ToffoliCompiler")
except ImportError as e:
    available_modules["compiler"] = False
    print(f"Could not import ToffoliCompiler: {e}")

# Try to import the RLToffoliOptimizer
try:
    from toffoli_optimizer.core.rl_optimizer import RLToffoliOptimizer
    available_modules["rl_optimizer"] = True
    print("Successfully imported RLToffoliOptimizer")
except ImportError as e:
    available_modules["rl_optimizer"] = False
    print(f"Could not import RLToffoliOptimizer: {e}")

# Try to import the ZXOptimizer
try:
    from toffoli_optimizer.core.zx_optimizer import ZXOptimizer
    available_modules["zx_optimizer"] = True
    print("Successfully imported ZXOptimizer")
except ImportError as e:
    available_modules["zx_optimizer"] = False
    print(f"Could not import ZXOptimizer: {e}")

# Try to import the ConsolidatedToffoliDepthOptimizer
try:
    from toffoli_optimizer.core.optimizer import ConsolidatedToffoliDepthOptimizer, OptimizationStrategy
    available_modules["consolidated_optimizer"] = True
    print("Successfully imported ConsolidatedToffoliDepthOptimizer")
except ImportError as e:
    try:
        # It might be under a different name in some versions of the repo
        from toffoli_optimizer.core.optimizer import ToffoliDepthOptimizer as ConsolidatedToffoliDepthOptimizer
        from toffoli_optimizer.core.optimizer import OptimizationStrategy
        available_modules["consolidated_optimizer"] = True
        print("Successfully imported ToffoliDepthOptimizer as ConsolidatedToffoliDepthOptimizer")
    except ImportError as e2:
        available_modules["consolidated_optimizer"] = False
        print(f"Could not import ConsolidatedToffoliDepthOptimizer: {e2}")

# Try to import utility functions
try:
    from toffoli_optimizer.utils.io_utils import save_circuit_to_qasm
    available_modules["io_utils"] = True
    print("Successfully imported I/O utilities")
except ImportError as e:
    available_modules["io_utils"] = False
    print(f"Could not import I/O utilities: {e}")

try:
    from toffoli_optimizer.utils.circuit_utils import get_default_coupling_map
    available_modules["circuit_utils"] = True
    print("Successfully imported circuit utilities")
except ImportError as e:
    available_modules["circuit_utils"] = False
    print(f"Could not import circuit utilities: {e}")

try:
    from toffoli_optimizer.utils.visualization import save_circuit_image
    available_modules["visualization"] = True
    print("Successfully imported visualization utilities")
except ImportError as e:
    available_modules["visualization"] = False
    print(f"Could not import visualization utilities: {e}")

# Check for essential modules
essential_modules = ["compiler", "io_utils", "circuit_utils"]
missing_modules = [m for m in essential_modules if not available_modules.get(m, False)]

if missing_modules:
    print(f"Error: The following essential modules are missing: {', '.join(missing_modules)}")
    print("Please ensure the repository is properly installed.")
    sys.exit(1)

# Add timeout functionality for optimizers
class TimeoutException(Exception):
    """Exception raised when a function execution times out."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeouts."""
    raise TimeoutException("Function execution timed out")

# Context manager for limiting execution time
from contextlib import contextmanager
import signal

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

# Common function to extract Toffoli gates with Qiskit 2.0 compatibility
def extract_toffoli_gates_from_circuit(circuit):
    """Extract Toffoli gates from a circuit with Qiskit 2.0 compatibility"""
    toffoli_gates = []
    for instruction in circuit.data:
        if instruction.operation.name == 'ccx':
            try:
                # Get qubit indices with Qiskit 2.0 compatibility
                qubits = []
                for q in instruction.qubits:
                    if hasattr(q, '_index'):
                        qubits.append(q._index)
                    elif hasattr(q, 'index'):
                        qubits.append(q.index)
                
                if len(qubits) >= 3:
                    # Format as expected: ([control1, control2], target)
                    toffoli_gates.append(([qubits[0], qubits[1]], qubits[2]))
            except Exception as e:
                print(f"Error extracting Toffoli gate: {e}")
    return toffoli_gates

def get_io_qubits_from_toffoli_gates(toffoli_gates, num_qubits):
    """Get input and output qubits from Toffoli gates"""
    input_qubits = set()
    output_qubits = set()
    
    for controls, target in toffoli_gates:
        input_qubits.update(controls)
        output_qubits.add(target)
    
    # Convert to lists and ensure they're not empty
    input_qubits = sorted(list(input_qubits)) if input_qubits else list(range(min(3, num_qubits)))
    output_qubits = sorted(list(output_qubits)) if output_qubits else list(range(3, min(6, num_qubits)))
    
    return input_qubits, output_qubits

def generate_toffoli_network(qubits=3, ancillas=4, controlling_anc=3):
    """
    Generate a Toffoli network from ShiftExperiment_FlowControl.
    
    Args:
        qubits: Number of qubits
        ancillas: Number of ancillas
        controlling_anc: Controlling ancilla index
        
    Returns:
        tuple: (toffoli_gates, output_qubits, input_qubits, num_qubits)
    """
    print(f"Generating Toffoli network with qubits={qubits}, ancillas={ancillas}, controlling_anc={controlling_anc}")
    
    # Create the circuit
    experiment = ShiftExperiment_FlowControl(qubits=qubits, ancillas=ancillas, controlling_anc=controlling_anc)
    circuit = experiment.Part2
    
    print(f"Circuit created with {circuit.num_qubits} qubits and {circuit.num_clbits} classical bits")
    
    # Extract Toffoli gates using the common function
    toffoli_gates = extract_toffoli_gates_from_circuit(circuit)
    
    # Get input/output qubits
    input_qubits, output_qubits = get_io_qubits_from_toffoli_gates(toffoli_gates, circuit.num_qubits)
    
    # If no Toffoli gates found, create a simple example
    if not toffoli_gates:
        print("No Toffoli gates found in circuit, creating simple example")
        toffoli_gates = [
            ([0, 1], 2),  # Controls: 0,1; Target: 2
            ([0, 2], 3),  # Controls: 0,2; Target: 3
            ([1, 2], 4),  # Controls: 1,2; Target: 4
            ([3, 4], 5),  # Controls: 3,4; Target: 5
        ]
        input_qubits = [0, 1]
        output_qubits = [2, 3, 4, 5]
    
    # Save the Toffoli network for debugging
    num_qubits = circuit.num_qubits
    
    print(f"Extracted {len(toffoli_gates)} Toffoli gates from circuit")
    print(f"Input qubits: {input_qubits}")
    print(f"Output qubits: {output_qubits}")
    print(f"Total qubits: {num_qubits}")
    
    return toffoli_gates, output_qubits, input_qubits, num_qubits

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Toffoli Circuit Optimization Sweep')
    
    parser.add_argument('--input', type=str, default=None,
                       help='Input file containing the Toffoli network (without extension)')
    
    parser.add_argument('--output', type=str, default='comprehensive_results',
                       help='Output directory for optimization results')
    
    parser.add_argument('--topology', type=str, default='falcon',
                       choices=['linear', 'grid', 'falcon'],
                       help='Target topology')
    
    parser.add_argument('--num-trials', type=int, default=5,
                       help='Number of optimization trials (default: 5)')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations of results')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    # Advanced parameters for the sweep
    parser.add_argument('--target-fidelities', type=str, default='0.85,0.9,0.95,0.99',
                       help='Comma-separated list of target fidelities to sweep')
    
    return parser.parse_args()

def create_results_table(all_results, output_file):
    """Create a formatted results table as a text file"""
    with open(output_file, 'w') as f:
        f.write("Toffoli Circuit Optimization Sweep Results\n")
        f.write("=" * 80 + "\n\n")
        
        # Write a summary table with only physical depths
        f.write(f"{'Method':<25} {'Fidelity':<10} {'Original Depth':<15} ")
        f.write(f"{'Physical Depth':<15} {'Depth Reduction':<15} ")
        f.write(f"{'Gate Count':<12} {'CX Count':<10} {'Runtime (s)':<12}\n")
        f.write("-" * 120 + "\n")
        
        for method, results_by_fidelity in sorted(all_results.items()):
            for fidelity, result in sorted(results_by_fidelity.items()):
                # Extract key metrics (use physical depths only)
                original_depth = result.get("original_physical_depth", 0)
                physical_depth = result.get("physical_depth", 0)
                
                # Calculate physical depth reduction
                if original_depth > 0:
                    depth_reduction = ((original_depth - physical_depth) / original_depth * 100)
                else:
                    depth_reduction = 0
                    
                gate_count = result.get("gate_count", 0)
                cx_count = result.get("cx_count", 0)
                runtime = result.get("runtime", 0)
                
                # Format and write the row
                f.write(f"{method:<25} {float(fidelity):<10.3f} {original_depth:<15} ")
                f.write(f"{physical_depth:<15} {depth_reduction:<15.2f} ")
                f.write(f"{gate_count:<12} {cx_count:<10} {runtime:<12.2f}\n")
        
        # Add a summary of the best results
        f.write("\n\nBest Results:\n")
        f.write("-" * 60 + "\n")
        
        # Find the best physical depth reduction
        best_physical_reduction = 0
        best_method = ""
        best_fidelity = 0
        
        for method, results_by_fidelity in all_results.items():
            for fidelity, result in results_by_fidelity.items():
                if "original_physical_depth" in result and "physical_depth" in result:
                    original = result["original_physical_depth"]
                    physical = result["physical_depth"]
                    if original > 0:
                        reduction = ((original - physical) / original * 100)
                        if reduction > best_physical_reduction:
                            best_physical_reduction = reduction
                            best_method = method
                            best_fidelity = fidelity
        
        f.write(f"Best physical depth reduction: {best_physical_reduction:.2f}% achieved by:\n")
        f.write(f"  Method: {best_method}\n")
        f.write(f"  Target fidelity: {best_fidelity}\n")
        
        # Find the best tradeoff (depth reduction / runtime)
        best_tradeoff = 0
        best_tradeoff_method = ""
        best_tradeoff_fidelity = 0
        
        for method, results_by_fidelity in all_results.items():
            for fidelity, result in results_by_fidelity.items():
                if "original_physical_depth" in result and "physical_depth" in result and "runtime" in result:
                    original = result["original_physical_depth"]
                    physical = result["physical_depth"]
                    runtime = result.get("runtime", 1)  # Avoid division by zero
                    
                    if original > 0 and runtime > 0:
                        reduction = ((original - physical) / original * 100)
                        tradeoff = reduction / runtime
                        
                        if tradeoff > best_tradeoff:
                            best_tradeoff = tradeoff
                            best_tradeoff_method = method
                            best_tradeoff_fidelity = fidelity
        
        f.write(f"\nBest efficiency (physical depth reduction / runtime): {best_tradeoff:.2f} achieved by:\n")
        f.write(f"  Method: {best_tradeoff_method}\n")
        f.write(f"  Target fidelity: {best_tradeoff_fidelity}\n")
        
        # End with timestamp
        f.write(f"\n\nResults generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def plot_comparison_chart(all_results, output_file):
    """Create a comparison chart of different optimization methods"""
    # Extract data for plotting - only physical depths
    methods = []
    original_depths = []
    physical_depths = []
    depth_reductions = []
    
    # Prepare data
    for method, results_by_fidelity in sorted(all_results.items()):
        for fidelity, result in sorted(results_by_fidelity.items()):
            method_name = f"{method} (f={fidelity})"
            methods.append(method_name)
            
            original_depths.append(result.get("original_physical_depth", 0))
            physical_depths.append(result.get("physical_depth", 0))
            
            # Calculate physical depth reduction
            if "original_physical_depth" in result and "physical_depth" in result:
                original = result["original_physical_depth"]
                physical = result["physical_depth"]
                if original > 0:
                    reduction = ((original - physical) / original * 100)
                    depth_reductions.append(reduction)
                else:
                    depth_reductions.append(0)
            else:
                depth_reductions.append(0)
    
    # Create the figure
    plt.figure(figsize=(14, 10))
    
    # Plot the depths
    plt.subplot(2, 1, 1)
    x = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x - width/2, original_depths, width, label='Original Physical Depth')
    plt.bar(x + width/2, physical_depths, width, label='Optimized Physical Depth')
    
    plt.ylabel('Circuit Depth')
    plt.title('Physical Circuit Depth Comparison Across Optimization Methods')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Plot depth reduction
    plt.subplot(2, 1, 2)
    
    plt.bar(x, depth_reductions, width, label='Physical Depth Reduction (%)')
    
    plt.ylabel('Depth Reduction (%)')
    plt.title('Physical Optimization Performance')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_optimized_physical_mapping(circuit, coupling_map, basis_gates=None, optimization_level=1):
    """
    Create a physically optimized circuit respecting coupling constraints.
    
    Args:
        circuit: Quantum circuit to optimize
        coupling_map: Coupling map constraints
        basis_gates: Target basis gates
        optimization_level: Optimization level (0-3)
    
    Returns:
        QuantumCircuit: Physically optimized circuit
    """
    if circuit is None:
        return None
        
    if basis_gates is None:
        basis_gates = ['id', 'rz', 'sx', 'x', 'cx']
    
    print(f"Mapping and optimizing circuit for target hardware (optimization level: {optimization_level})...")
    
    try:
        from qiskit import transpile
        
        # Apply physical mapping with specified optimization level
        try:
            physical_circuit = transpile(
                circuit,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                optimization_level=optimization_level
            )
            return physical_circuit
        except Exception as e:
            print(f"Error applying physical mapping: {e}")
            print("Trying alternative approach...")
            
            # Fallback to simpler mapping
            try:
                physical_circuit = transpile(
                    circuit,
                    coupling_map=coupling_map,
                    basis_gates=basis_gates,
                    optimization_level=min(1, optimization_level)  # Use lower optimization level
                )
                return physical_circuit
            except:
                print("Alternative approach failed, returning original circuit")
                return circuit
                
    except ImportError:
        print("Qiskit transpiler not available, returning original circuit")
        return circuit

def optimize_with_rl(circuit, coupling_map, target_fidelity, num_trials=3, debug=False, timeout_seconds=60):
    """Optimize circuit using RL optimizer if available"""
    if not available_modules.get("rl_optimizer", False):
        print("RL optimizer not available, skipping")
        return None
    
    try:
        # Extract Toffoli gates for the RL optimizer
        toffoli_gates = extract_toffoli_gates_from_circuit(circuit)
        print(f"Found {len(toffoli_gates)} Toffoli gates for RL optimizer")
        
        # Skip if no Toffoli gates found
        if not toffoli_gates:
            print("No Toffoli gates found for RL optimization, skipping")
            return None
        
        # Create the RL optimizer
        optimizer = RLToffoliOptimizer(
            target_fidelity=target_fidelity,
            max_steps=30,  # Reasonable default
            debug_mode=debug
        )
        
        # Get original physical depth for accurate comparison
        original_physical_circuit = create_optimized_physical_mapping(
            circuit.copy(),
            coupling_map,
            basis_gates=['id', 'rz', 'sx', 'x', 'cx'],
            optimization_level=1
        )
        original_physical_depth = original_physical_circuit.depth()
        
        best_depth = float('inf')
        best_result = None
        
        for trial in range(num_trials):
            print(f"  RL Optimization trial {trial+1}/{num_trials}...")
            start_time = time.time()
            
            try:
                # Run optimization with timeout
                with time_limit(timeout_seconds):
                    # Run optimization
                    optimized_circuit = optimizer.optimize_circuit(circuit.copy(), coupling_map=coupling_map)
            except TimeoutException:
                print(f"  RL Optimization trial {trial+1} timed out after {timeout_seconds} seconds")
                continue
            except Exception as e:
                print(f"  Error in RL Optimization trial {trial+1}: {e}")
                continue
            
            # Map to physical circuit with optimization level 1
            physical_circuit = create_optimized_physical_mapping(
                optimized_circuit.copy(),
                coupling_map,
                basis_gates=['id', 'rz', 'sx', 'x', 'cx'],
                optimization_level=1
            )
            
            # Check if we got a better result
            if optimized_circuit is not None and physical_circuit.depth() < best_depth:
                best_depth = physical_circuit.depth()
                best_result = {
                    "circuit": optimized_circuit,
                    "runtime": time.time() - start_time,
                    "original_physical_depth": original_physical_depth,
                    "physical_depth": physical_circuit.depth(),
                    "gate_count": len(optimized_circuit.data),
                    "cx_count": sum(1 for g in optimized_circuit.data if g.operation.name == 'cx')
                }
        
        return best_result
    
    except Exception as e:
        print(f"Error running RL optimization: {e}")
        import traceback
        traceback.print_exc()
        return None

def optimize_with_zx(circuit, coupling_map, target_fidelity, debug=False, timeout_seconds=60):
    """Optimize circuit using ZX optimizer if available"""
    if not available_modules.get("zx_optimizer", False):
        print("ZX optimizer not available, skipping")
        return None
    
    try:
        # Extract Toffoli gates for the ZX optimizer
        toffoli_gates = extract_toffoli_gates_from_circuit(circuit)
        print(f"Found {len(toffoli_gates)} Toffoli gates for ZX optimizer")
        
        # Skip if no Toffoli gates found
        if not toffoli_gates:
            print("No Toffoli gates found for ZX optimization, skipping")
            return None
        
        # Get original physical depth for accurate comparison
        original_physical_circuit = create_optimized_physical_mapping(
            circuit.copy(),
            coupling_map,
            basis_gates=['id', 'rz', 'sx', 'x', 'cx'],
            optimization_level=1
        )
        original_physical_depth = original_physical_circuit.depth()
        
        # Create the ZX optimizer
        aggressive_mode = (target_fidelity < 0.9)
        zx_optimizer = ZXOptimizer(
            aggressive_mode=aggressive_mode,
            target_fidelity=target_fidelity,
            debug_mode=debug
        )
        
        # Run optimization with timeout
        try:
            start_time = time.time()
            with time_limit(timeout_seconds):
                optimized_circuit = zx_optimizer.optimize_circuit(circuit.copy(), coupling_map=coupling_map)
        except TimeoutException:
            print(f"ZX optimization timed out after {timeout_seconds} seconds")
            return None
        except Exception as e:
            print(f"Error in ZX optimization: {e}")
            return None
        
        # Map to physical circuit with optimization level 1
        physical_circuit = create_optimized_physical_mapping(
            optimized_circuit.copy(),
            coupling_map,
            basis_gates=['id', 'rz', 'sx', 'x', 'cx'],
            optimization_level=1
        )
        
        # Return the result
        return {
            "circuit": optimized_circuit,
            "runtime": time.time() - start_time,
            "original_physical_depth": original_physical_depth,
            "physical_depth": physical_circuit.depth(),
            "gate_count": len(optimized_circuit.data),
            "cx_count": sum(1 for g in optimized_circuit.data if g.operation.name == 'cx')
        }
    
    except Exception as e:
        print(f"Error running ZX optimization: {e}")
        import traceback
        traceback.print_exc()
        return None

def optimize_with_consolidated(circuit, coupling_map, target_fidelity, strategy, debug=False, timeout_seconds=60):
    """Optimize circuit using ConsolidatedToffoliDepthOptimizer with specified strategy"""
    if not available_modules.get("consolidated_optimizer", False):
        print("ConsolidatedToffoliDepthOptimizer not available, skipping")
        return None
    
    try:
        # Extract Toffoli gates for the consolidated optimizer
        toffoli_gates = extract_toffoli_gates_from_circuit(circuit)
        print(f"Found {len(toffoli_gates)} Toffoli gates for consolidated optimizer")
        
        # Skip if no Toffoli gates found
        if not toffoli_gates:
            print("No Toffoli gates found for consolidated optimization, skipping")
            return None
        
        # Get original physical depth for accurate comparison
        original_physical_circuit = create_optimized_physical_mapping(
            circuit.copy(),
            coupling_map,
            basis_gates=['id', 'rz', 'sx', 'x', 'cx'],
            optimization_level=1
        )
        original_physical_depth = original_physical_circuit.depth()
        
        # Get input/output qubits
        input_qubits, output_qubits = get_io_qubits_from_toffoli_gates(toffoli_gates, circuit.num_qubits)
        
        # Get the optimization strategy
        if isinstance(strategy, str):
            # Convert string to enum
            strategy_enum = None
            for strat in OptimizationStrategy:
                if strat.name == strategy:
                    strategy_enum = strat
                    break
            if strategy_enum is None:
                print(f"Unknown strategy: {strategy}, defaulting to STANDARD")
                strategy_enum = OptimizationStrategy.STANDARD
        else:
            strategy_enum = strategy
            
        # Set min_fidelity based on the target_fidelity and strategy
        min_fidelity = target_fidelity
        if strategy_enum == OptimizationStrategy.ULTRA_DEPTH_REDUCTION:
            min_fidelity = max(0.8, target_fidelity - 0.1)  # More aggressive
        
        # Create the optimizer
        optimizer = ConsolidatedToffoliDepthOptimizer(
            target_fidelity=target_fidelity,
            min_fidelity=min_fidelity,
            max_passes=2,  # Reasonable default
            debug_mode=debug,
            strategy=strategy_enum,
            use_zx_optimization=available_modules.get("zx_optimizer", False),
            use_rl_optimization=available_modules.get("rl_optimizer", False),
            pass_timeout_seconds=timeout_seconds
        )
        
        # Run optimization with timeout
        try:
            start_time = time.time()
            with time_limit(timeout_seconds * 2):  # Double the timeout for the whole process
                # Try direct circuit optimization first
                if hasattr(optimizer, "optimize_circuit"):
                    optimized_circuit = optimizer.optimize_circuit(circuit.copy(), coupling_map=coupling_map)
                    
                    # Map to physical circuit with optimization level 1
                    physical_circuit = create_optimized_physical_mapping(
                        optimized_circuit.copy(),
                        coupling_map,
                        basis_gates=['id', 'rz', 'sx', 'x', 'cx'],
                        optimization_level=1
                    )
                    
                    # Calculate metrics
                    result = {
                        "circuit": optimized_circuit,
                        "runtime": time.time() - start_time,
                        "original_physical_depth": original_physical_depth,
                        "physical_depth": physical_circuit.depth(),
                        "gate_count": len(optimized_circuit.data),
                        "cx_count": sum(1 for g in optimized_circuit.data if g.operation.name == 'cx'),
                        "strategy": strategy_enum.name
                    }
                    
                    return result
                else:
                    print("Direct circuit optimization method not available")
                    print("Falling back to optimize_toffoli_network method")
                    
                    result = optimizer.optimize_toffoli_network(
                        toffoli_gates,
                        output_qubits,
                        input_qubits,
                        circuit.num_qubits,
                        topology=None,  # Use coupling_map instead
                        coupling_map=coupling_map,
                        original_circuit=circuit.copy()  # Pass original circuit for optimization
                    )
                    
                    # Extract the optimized circuit from results
                    if "optimized" in result and "circuit" in result["optimized"]:
                        optimized_circuit = result["optimized"]["circuit"]
                        
                        # Map to physical circuit with optimization level 1
                        physical_circuit = create_optimized_physical_mapping(
                            optimized_circuit.copy(),
                            coupling_map,
                            basis_gates=['id', 'rz', 'sx', 'x', 'cx'],
                            optimization_level=1
                        )
                        
                        # Calculate metrics
                        return {
                            "circuit": optimized_circuit,
                            "runtime": time.time() - start_time,
                            "original_physical_depth": original_physical_depth,
                            "physical_depth": physical_circuit.depth(),
                            "gate_count": len(optimized_circuit.data),
                            "cx_count": sum(1 for g in optimized_circuit.data if g.operation.name == 'cx'),
                            "strategy": strategy_enum.name
                        }
                    else:
                        print("Failed to extract optimized circuit from results")
                        return None
        except TimeoutException:
            print(f"Consolidated optimization timed out after {timeout_seconds*2} seconds")
            return None
        except Exception as e:
            print(f"Error in consolidated optimization: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    except Exception as e:
        print(f"Error running ConsolidatedToffoliDepthOptimizer: {e}")
        import traceback
        traceback.print_exc()
        return None

def optimize_with_transpiler(circuit, coupling_map, optimization_level=3, timeout_seconds=30):
    """Optimize circuit using Qiskit's transpiler"""
    try:
        from qiskit import transpile
        
        # Get original physical depth for accurate comparison
        original_physical_circuit = create_optimized_physical_mapping(
            circuit.copy(),
            coupling_map,
            basis_gates=['id', 'rz', 'sx', 'x', 'cx'],
            optimization_level=1
        )
        original_physical_depth = original_physical_circuit.depth()
        
        # Run transpilation with timeout
        try:
            start_time = time.time()
            with time_limit(timeout_seconds):
                # Do optimization and physical mapping in one step
                # This ensures the circuit is expressed in physical gates
                physical_circuit = transpile(
                    circuit.copy(),
                    coupling_map=coupling_map,  # Apply coupling constraints
                    basis_gates=['id', 'rz', 'sx', 'x', 'cx'],  # Use physical basis gates
                    optimization_level=optimization_level  # Use requested optimization level
                )
            
            # Store both the optimized physical circuit and its depth
            physical_depth = physical_circuit.depth()
            
            # For completeness, calculate the gate counts
            gate_count = len(physical_circuit.data)
            cx_count = sum(1 for g in physical_circuit.data if g.operation.name == 'cx')
            
            return {
                "circuit": physical_circuit,  # This is already the physical circuit
                "runtime": time.time() - start_time,
                "original_physical_depth": original_physical_depth,
                "physical_depth": physical_depth,
                "gate_count": gate_count,
                "cx_count": cx_count,
                "optimization_level": optimization_level
            }
                
        except TimeoutException:
            print(f"Transpiler optimization timed out after {timeout_seconds} seconds")
            return None
        except Exception as e:
            print(f"Error in transpiler optimization: {e}")
            return None
    
    except Exception as e:
        print(f"Error running transpiler optimization: {e}")
        import traceback
        traceback.print_exc()
        return None
        
def run_optimization_sweep(circuit, topology, num_trials, target_fidelities, debug, output_dir):
    """
    Run a sweep of various optimization methods on the input circuit.
    
    Args:
        circuit: QuantumCircuit to optimize
        topology: Target topology ('linear', 'grid', 'falcon')
        num_trials: Number of optimization trials
        target_fidelities: List of target fidelities to try
        debug: Whether to enable debug output
        output_dir: Directory for output files
        
    Returns:
        dict: Dictionary with all optimization results
    """
    print(f"Running optimization sweep for circuit with {len(circuit.data)} gates on {circuit.num_qubits} qubits")
    print(f"Target topology: {topology}")
    print(f"Number of trials: {num_trials}")
    print(f"Target fidelities: {target_fidelities}")
    
    # Get coupling map
    coupling_map = get_default_coupling_map(topology, circuit.num_qubits)
    
    print(f"Circuit has depth {circuit.depth()}")
    
    # Debug gate types
    gate_types = set(inst.operation.name for inst in circuit.data)
    print(f"Circuit contains gate types: {gate_types}")
    toffoli_count = sum(1 for inst in circuit.data if inst.operation.name == 'ccx')
    print(f"Circuit contains {toffoli_count} Toffoli (ccx) gates")
    
    # Results dictionary
    all_results = {}
    
    # Save the original circuit details
    original_dir = os.path.join(output_dir, "original")
    os.makedirs(original_dir, exist_ok=True)
    
    # Save circuit as QASM and image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    save_circuit_to_qasm(
        circuit,
        os.path.join(original_dir, f"original_circuit_{timestamp}.txt")
    )
    
    # Get original physical circuit depth once for consistency
    original_physical_circuit = create_optimized_physical_mapping(
        circuit.copy(),
        coupling_map,
        basis_gates=['id', 'rz', 'sx', 'x', 'cx'],
        optimization_level=1
    )
    original_physical_depth = original_physical_circuit.depth()
    print(f"Original physical circuit depth: {original_physical_depth}")
    
    if available_modules.get("visualization", False):
        try:
            save_circuit_image(
                circuit,
                f"original_circuit_{timestamp}",
                output_dir=original_dir,
                use_text_mode=True
            )
        except Exception as e:
            print(f"Warning: Could not save original circuit image: {e}")
    
    # 1. Run specialized optimizers first (before transpiler decomposes Toffoli gates)
    
    # 1a. Run ConsolidatedToffoliDepthOptimizer with different strategies
    if available_modules.get("consolidated_optimizer", False):
        print("\n=== Running ConsolidatedToffoliDepthOptimizer with Different Strategies ===")
        
        # Define strategies to test
        strategies = [
            ("STANDARD", "Standard optimization balancing depth and fidelity"),
            ("ULTRA_DEPTH_REDUCTION", "Aggressive depth reduction with controlled fidelity trade-offs"),
            ("DEPTH_FIDELITY_BALANCE", "Explicit balancing of depth and fidelity"),
            ("HYBRID", "Combined approach using multiple techniques")
        ]
        
        for strategy_name, strategy_desc in strategies:
            method_name = f"Consolidated_{strategy_name}"
            all_results[method_name] = {}
            
            print(f"\nRunning ConsolidatedToffoliDepthOptimizer with strategy {strategy_name}:")
            print(f"  {strategy_desc}")
            
            for fidelity in target_fidelities:
                print(f"\n  Target fidelity {fidelity}:")
                
                result = optimize_with_consolidated(
                    circuit,
                    coupling_map,
                    fidelity,
                    strategy_name,
                    debug
                )
                
                if result:
                    result["target_fidelity"] = fidelity
                    all_results[method_name][fidelity] = result
                    
                    print(f"    Original physical depth: {result['original_physical_depth']}")
                    print(f"    Optimized physical depth: {result['physical_depth']}")
                    print(f"    Physical depth reduction: {((result['original_physical_depth'] - result['physical_depth']) / result['original_physical_depth'] * 100):.2f}%")
                    print(f"    Runtime: {result['runtime']:.2f} seconds")
                    
                    # Save the optimized circuit
                    method_dir = os.path.join(output_dir, f"{method_name}_f{fidelity}")
                    os.makedirs(method_dir, exist_ok=True)
                    
                    save_circuit_to_qasm(
                        result["circuit"],
                        os.path.join(method_dir, f"optimized_circuit_{timestamp}.txt")
                    )
                    
                    if available_modules.get("visualization", False):
                        try:
                            save_circuit_image(
                                result["circuit"],
                                f"optimized_circuit_{timestamp}",
                                output_dir=method_dir,
                                use_text_mode=True
                            )
                        except Exception as e:
                            print(f"Warning: Could not save circuit image: {e}")
    
    # 1b. Run RL optimization if available
    if available_modules.get("rl_optimizer", False):
        print("\n=== Running RL Optimization ===")
        all_results["RL_Optimizer"] = {}
        
        for fidelity in target_fidelities:
            print(f"\nRunning RL optimization with target fidelity {fidelity}...")
            
            result = optimize_with_rl(circuit, coupling_map, fidelity, num_trials, debug)
            if result:
                result["target_fidelity"] = fidelity
                all_results["RL_Optimizer"][fidelity] = result
                
                print(f"  Target fidelity {fidelity}:")
                print(f"    Original physical depth: {result['original_physical_depth']}")
                print(f"    Optimized physical depth: {result['physical_depth']}")
                print(f"    Physical depth reduction: {((result['original_physical_depth'] - result['physical_depth']) / result['original_physical_depth'] * 100):.2f}%")
                print(f"    Runtime: {result['runtime']:.2f} seconds")
                
                # Save the optimized circuit
                method_dir = os.path.join(output_dir, f"RL_Optimizer_f{fidelity}")
                os.makedirs(method_dir, exist_ok=True)
                
                save_circuit_to_qasm(
                    result["circuit"],
                    os.path.join(method_dir, f"optimized_circuit_{timestamp}.txt")
                )
                
                if available_modules.get("visualization", False):
                    try:
                        save_circuit_image(
                            result["circuit"],
                            f"optimized_circuit_{timestamp}",
                            output_dir=method_dir,
                            use_text_mode=True
                        )
                    except Exception as e:
                        print(f"Warning: Could not save circuit image: {e}")
    
    # 1c. Run ZX optimization if available
    if available_modules.get("zx_optimizer", False):
        print("\n=== Running ZX Optimization ===")
        all_results["ZX_Optimizer"] = {}
        
        for fidelity in target_fidelities:
            print(f"\nRunning ZX optimization with target fidelity {fidelity}...")
            
            result = optimize_with_zx(circuit, coupling_map, fidelity, debug)
            if result:
                result["target_fidelity"] = fidelity
                all_results["ZX_Optimizer"][fidelity] = result
                
                print(f"  Target fidelity {fidelity}:")
                print(f"    Original physical depth: {result['original_physical_depth']}")
                print(f"    Optimized physical depth: {result['physical_depth']}")
                print(f"    Physical depth reduction: {((result['original_physical_depth'] - result['physical_depth']) / result['original_physical_depth'] * 100):.2f}%")
                print(f"    Runtime: {result['runtime']:.2f} seconds")
                
                # Save the optimized circuit
                method_dir = os.path.join(output_dir, f"ZX_Optimizer_f{fidelity}")
                os.makedirs(method_dir, exist_ok=True)
                
                save_circuit_to_qasm(
                    result["circuit"],
                    os.path.join(method_dir, f"optimized_circuit_{timestamp}.txt")
                )
                
                if available_modules.get("visualization", False):
                    try:
                        save_circuit_image(
                            result["circuit"],
                            f"optimized_circuit_{timestamp}",
                            output_dir=method_dir,
                            use_text_mode=True
                        )
                    except Exception as e:
                        print(f"Warning: Could not save circuit image: {e}")
    
    # 2. Now run transpiler optimization with different levels
    all_results["Transpile_L1"] = {}
    all_results["Transpile_L2"] = {}
    all_results["Transpile_L3"] = {}
    
    print("\n=== Running Qiskit Transpiler Optimization ===")
    
    for level in [1, 2, 3]:
        method_name = f"Transpile_L{level}"
        print(f"\nRunning transpiler with optimization level {level}...")
        
        for fidelity in target_fidelities:
            # Transpiler doesn't use fidelity directly, but we track it for comparison
            result = optimize_with_transpiler(circuit, coupling_map, level)
            if result:
                result["target_fidelity"] = fidelity
                all_results[method_name][fidelity] = result
                
                print(f"  Level {level}, Fidelity {fidelity}:")
                print(f"    Original physical depth: {result['original_physical_depth']}")
                print(f"    Optimized physical depth: {result['physical_depth']}")
                print(f"    Physical depth reduction: {((result['original_physical_depth'] - result['physical_depth']) / result['original_physical_depth'] * 100):.2f}%")
                print(f"    Runtime: {result['runtime']:.2f} seconds")
                
                # Save the optimized circuit
                method_dir = os.path.join(output_dir, f"{method_name}_f{fidelity}")
                os.makedirs(method_dir, exist_ok=True)
                
                save_circuit_to_qasm(
                    result["circuit"],
                    os.path.join(method_dir, f"optimized_circuit_{timestamp}.txt")
                )
                
                if available_modules.get("visualization", False):
                    try:
                        save_circuit_image(
                            result["circuit"],
                            f"optimized_circuit_{timestamp}",
                            output_dir=method_dir,
                            use_text_mode=True
                        )
                    except Exception as e:
                        print(f"Warning: Could not save circuit image: {e}")
    
    return all_results

def save_serializable_results(results, output_file):
    """Save results to a JSON file, handling non-serializable objects"""
    serializable_results = {}
    
    for method, results_by_fidelity in results.items():
        serializable_results[method] = {}
        
        for fidelity, result in results_by_fidelity.items():
            # Create a serializable copy
            serializable_result = {}
            
            for key, value in result.items():
                # Skip non-serializable objects like circuits
                if key != "circuit":
                    serializable_result[key] = value
            
            serializable_results[method][fidelity] = serializable_result
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Parse target fidelities
    target_fidelities = [float(f.strip()) for f in args.target_fidelities.split(',')]
    
    # Create circuit from the JSON data
    json_data = {
        "num_qubits": 6,
        "num_clbits": 3,
        "instructions": [
            {"name": "ccx", "qubits": [3, 4, 5], "clbits": []},
            {"name": "x", "qubits": [5], "clbits": []},
            {"name": "ccx", "qubits": [3, 5, 4], "clbits": []},
            {"name": "x", "qubits": [5], "clbits": []},
            {"name": "x", "qubits": [5], "clbits": []},
            {"name": "x", "qubits": [4], "clbits": []},
            {"name": "ccx", "qubits": [4, 5, 3], "clbits": []},
            {"name": "x", "qubits": [4], "clbits": []},
            {"name": "x", "qubits": [5], "clbits": []},
            {"name": "x", "qubits": [5], "clbits": []},
            {"name": "ccx", "qubits": [3, 5, 4], "clbits": []},
            {"name": "x", "qubits": [5], "clbits": []},
            {"name": "ccx", "qubits": [3, 4, 5], "clbits": []},
            {"name": "ccx", "qubits": [3, 5, 4], "clbits": []},
            {"name": "x", "qubits": [4], "clbits": []},
            {"name": "ccx", "qubits": [4, 5, 3], "clbits": []},
            {"name": "x", "qubits": [4], "clbits": []},
            {"name": "ccx", "qubits": [3, 5, 4], "clbits": []},
            {"name": "ccx", "qubits": [3, 4, 5], "clbits": []},
            {"name": "x", "qubits": [5], "clbits": []},
            {"name": "ccx", "qubits": [4, 5, 3], "clbits": []},
            {"name": "x", "qubits": [5], "clbits": []},
            {"name": "ccx", "qubits": [3, 4, 5], "clbits": []},
            {"name": "ccx", "qubits": [4, 5, 3], "clbits": []},
            {"name": "ccx", "qubits": [3, 4, 5], "clbits": []},
            {"name": "x", "qubits": [5], "clbits": []},
            {"name": "ccx", "qubits": [3, 5, 4], "clbits": []},
            {"name": "x", "qubits": [5], "clbits": []},
            {"name": "ccx", "qubits": [3, 4, 5], "clbits": []},
            {"name": "ccx", "qubits": [3, 5, 4], "clbits": []},
            {"name": "ccx", "qubits": [3, 4, 5], "clbits": []}
        ]
    }
    
    # Create the circuit from JSON
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit(json_data["num_qubits"], json_data["num_clbits"])
    
    # Add instructions
    for inst in json_data["instructions"]:
        gate_name = inst["name"]
        qubits = inst["qubits"]
        clbits = inst.get("clbits", [])
        
        if gate_name == "ccx":
            if len(qubits) >= 3:
                circuit.ccx(qubits[0], qubits[1], qubits[2])
        elif gate_name == "cx":
            if len(qubits) >= 2:
                circuit.cx(qubits[0], qubits[1])
        elif gate_name == "x":
            if len(qubits) >= 1:
                circuit.x(qubits[0])
        elif gate_name == "h":
            if len(qubits) >= 1:
                circuit.h(qubits[0])

    
    # Count Toffoli gates for verification
    toffoli_count = sum(1 for inst in circuit.data if inst.operation.name == 'ccx')
    x_count = sum(1 for inst in circuit.data if inst.operation.name == 'x')
    
    print(f"Created circuit from JSON with {circuit.depth()} depth, {len(circuit.data)} gates")
    print(f"Circuit contains {toffoli_count} Toffoli (ccx) gates and {x_count} X gates")
    
    # Run the optimization sweep
    all_results = run_optimization_sweep(
        circuit,
        args.topology,
        args.num_trials,
        target_fidelities,
        args.debug,
        args.output
    )
    
    # Save results
    results_file = os.path.join(args.output, "results.json")
    save_serializable_results(all_results, results_file)
    print(f"Results saved to {results_file}")
    
    # Create results table
    table_file = os.path.join(args.output, "results_table.txt")
    create_results_table(all_results, table_file)
    print(f"Results table saved to {table_file}")
    
    # Generate visualizations if requested
    if args.visualize:
        chart_file = os.path.join(args.output, "optimization_comparison.png")
        plot_comparison_chart(all_results, chart_file)
        print(f"Comparison chart saved to {chart_file}")
    
    print("\nOptimization sweep completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
