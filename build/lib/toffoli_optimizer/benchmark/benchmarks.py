"""
Benchmark module for comparing Toffoli optimization approaches.

This module provides benchmark functionality for evaluating the performance
of the Toffoli Depth Optimizer against other approaches like the Qiskit transpiler.
"""

import os
import json
import time
import traceback
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

# Import from core
from ..core.compiler import ToffoliCompiler
from ..core.optimizer import ToffoliDepthOptimizer, OptimizationStrategy

# Import from utils
from ..utils.circuit_utils import create_optimized_physical_mapping
from ..utils.io_utils import save_circuit_to_qasm as save_circuit
from ..utils.visualization import save_benchmark_circuits


class ToffoliBenchmark:
    """
    Base benchmark class for Toffoli optimization.
    
    This class provides the foundational functionality for benchmarking
    Toffoli network optimization approaches.
    """
    
    def __init__(self, config=None):
        """
        Initialize the benchmark with configuration.
        
        Args:
            config: Dictionary with benchmark configuration
        """
        # Default configuration
        default_config = {
            "output_dir": "benchmark_results",
            "repetitions": 3,
            "max_passes": 2,
            "target_fidelity": 0.95,
            "use_parallel": True,
            "debug_mode": False
        }
        
        # Use default config if none provided
        self.config = default_config
        
        # Update with provided config if any
        if config:
            for key, value in config.items():
                self.config[key] = value
        
        # Create timestamp for this benchmark run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize results dictionary
        self.results = {}
        
        # Initialize compiler and optimizer
        self.compiler = ToffoliCompiler(debug_mode=self.config["debug_mode"])
        self.optimizer = ToffoliDepthOptimizer(
            target_fidelity=self.config["target_fidelity"],
            max_passes=self.config["max_passes"],
            output_dir=self.config["output_dir"],
            use_parallel=self.config["use_parallel"],
            debug_mode=self.config["debug_mode"]
        )
        
class ComparisonBenchmark(ToffoliBenchmark):
    """
    Benchmark class for comparing Toffoli Depth Optimizer with Qiskit transpiler.
    
    This class provides functionality to run benchmarks comparing different
    optimization approaches on Toffoli networks.
    """
    
    def __init__(self, config=None):
        """Initialize with parent class configuration"""
        super().__init__(config)
    
    def run_benchmark(self, network_name, network_config, topology,
                      optimization_level, target_fidelity):
        """
        Run a specific benchmark configuration.
        
        Args:
            network_name: Name of the Toffoli network
            network_config: Configuration for the network
            topology: Target topology ('linear', 'grid', 'falcon')
            optimization_level: Qiskit optimization level (0-3)
            target_fidelity: Target fidelity for optimization
            
        Returns:
            dict: Benchmark results
        """
        # Create a unique key for this benchmark
        config_key = f"{network_name}_{topology}_opt{optimization_level}_fid{target_fidelity:.2f}"
        
        print(f"\nRunning benchmark: {config_key}")
        
        # Initialize optimizer with target fidelity
        self.optimizer.target_fidelity = target_fidelity
        
        # Call the define function to get the Toffoli network
        try:
            define_func = network_config["define_func"]
            toffoli_network = define_func()
            if isinstance(toffoli_network, tuple) and len(toffoli_network) >= 3:
                toffoli_gates, output_qubits, input_qubits = toffoli_network
            else:
                raise ValueError("Invalid Toffoli network format returned by define_func")
        except Exception as e:
            print(f"Error defining Toffoli network: {e}")
            traceback.print_exc()
            return None
        
        # Get the number of qubits
        num_qubits = network_config.get("num_qubits", 8)
            
        # Get coupling map for the selected topology
        from ..utils.circuit_utils import get_default_coupling_map
        coupling_map = get_default_coupling_map(topology, num_qubits)
        
        basis_gates = ['id', 'rz', 'sx', 'x', 'cx']  # Common basis gates
        
        # Create a fresh circuit with the original Toffoli network (logical circuit)
        logical_circuit, _ = self.compiler.create_toffoli_network(
            toffoli_gates,
            num_qubits,
            use_ancilla=True,
            target_fidelity=1.0  # Use perfect fidelity for logical circuit
        )
        
        if logical_circuit is None:
            print("Error: Failed to create logical circuit")
            return None
        
        # Get logical circuit metrics
        logical_metrics = self.compiler.get_circuit_metrics(logical_circuit)
        logical_depth = logical_metrics["depth"]
        
        print(f"Logical circuit depth: {logical_depth}")
        
        # Create naive physical mapping
        naive_physical_circuit = create_optimized_physical_mapping(
            logical_circuit.copy(),
            coupling_map,
            basis_gates,
            optimization_level=1  # Lower optimization level for naive mapping
        )
        
        # Get naive physical circuit metrics
        naive_metrics = self.compiler.get_circuit_metrics(naive_physical_circuit)
        naive_depth = naive_metrics["depth"]
        naive_gate_count = sum(naive_metrics["gate_counts"].values())
        naive_cx_count = naive_metrics["cx_count"]
        
        print(f"Naive physical mapping depth: {naive_depth}")
        
        # Run Toffoli Depth Optimizer benchmark
        toffoli_optimizer_results = self.run_toffoli_optimizer(
            toffoli_gates=toffoli_gates,
            output_qubits=output_qubits,
            input_qubits=input_qubits,
            num_qubits=num_qubits,
            topology=topology,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            target_fidelity=target_fidelity,
            original_circuit=naive_physical_circuit,
            logical_circuit=logical_circuit
        )
        
        # Run Qiskit transpiler benchmark
        qiskit_transpiler_results = self.run_qiskit_transpiler(
            circuit=logical_circuit.copy(),
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            optimization_level=optimization_level
        )
        
        # Store the results
        self.results[config_key] = {
            "config": {
                "network": network_name,
                "topology": topology,
                "optimization_level": optimization_level,
                "target_fidelity": target_fidelity,
                "num_qubits": num_qubits,
                "num_toffoli_gates": len(toffoli_gates)
            },
            "logical": {
                "depth": logical_metrics["depth"],
                "gate_count": sum(logical_metrics["gate_counts"].values()),
                "cx_count": logical_metrics["cx_count"],
                "circuit": logical_circuit
            },
            "original": {
                "depth": naive_depth,
                "gate_count": naive_gate_count,
                "cx_count": naive_cx_count,
                "description": "Naive mapping to physical topology",
                "circuit": naive_physical_circuit
            },
            "toffoli_optimizer": toffoli_optimizer_results,
            "qiskit_transpiler": qiskit_transpiler_results
        }
        
        # Save circuit images for this benchmark configuration
        print(f"Saving circuit images for {config_key}...")
        image_paths = save_benchmark_circuits(self.results[config_key],
                                            os.path.join(self.config["output_dir"], config_key))
        # Add image paths to results
        if image_paths:
            self.results[config_key]["image_paths"] = image_paths
        
        # Print comparison summary
        self.print_comparison_summary(config_key)
        
        # Save interim results
        self._save_results()
        
        return self.results[config_key]

    def run_toffoli_optimizer(self, toffoli_gates, output_qubits, input_qubits,
                            num_qubits, topology, coupling_map, basis_gates,
                            target_fidelity, original_circuit, logical_circuit):
        """
        Run the Toffoli Depth Optimizer benchmark.
        
        Args:
            toffoli_gates: List of Toffoli gates
            output_qubits: List of output qubit indices
            input_qubits: List of input qubit indices
            num_qubits: Number of qubits
            topology: Target topology
            coupling_map: Coupling map for the topology
            basis_gates: Basis gates for the device
            target_fidelity: Target fidelity
            original_circuit: Original circuit for comparison
            logical_circuit: Logical circuit representation
        
        Returns:
            dict: Benchmark results
        """
        print("\nRunning Toffoli Depth Optimizer benchmark...")
        
        # Measure execution time
        start_time = time.time()
        
        try:
            # Run the optimizer
            optimizer_results = self.optimizer.optimize_toffoli_network(
                toffoli_gates,
                output_qubits,
                input_qubits,
                num_qubits,
                topology=topology,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                target_fidelity=target_fidelity,
                original_circuit=original_circuit,
                logical_circuit=logical_circuit
            )
            

            execution_time = time.time() - start_time
            
            print(f"Toffoli Optimizer benchmark completed in {execution_time:.4f} seconds")
            
            # Get the depth reduction percentage
            logical_depth = logical_circuit.depth()
            optimized_depth = optimizer_results["optimized"]["depth"]
            naive_depth = original_circuit.depth()
            physical_depth = optimizer_results["mapped"]["depth"]
            
            # Calculate reductions
            depth_reduction = ((logical_depth - optimized_depth) / logical_depth * 100) if logical_depth > 0 else 0
            depth_reduction_from_naive = ((naive_depth - physical_depth) / naive_depth * 100) if naive_depth > 0 else 0
            
            print(f"Depth reduction: {depth_reduction:.2f}%")
            print(f"Depth reduction from naive mapping: {depth_reduction_from_naive:.2f}%")
            
            # Add execution time and depth reductions to results
            optimizer_results["execution_time"] = execution_time
            optimizer_results["depth_reduction"] = depth_reduction
            optimizer_results["depth_reduction_from_naive"] = depth_reduction_from_naive
            
            # Preserve the circuits for image saving
            optimizer_results["logical_circuit"] = logical_circuit
            optimizer_results["naive_physical_circuit"] = original_circuit
            
            return optimizer_results
            
        except Exception as e:
            print(f"Error running Toffoli Optimizer benchmark: {str(e)}")
            traceback.print_exc()
            
            # Return a minimal result dictionary with error information
            return {
                "error": str(e),
                "execution_time": time.time() - start_time,
                "depth_reduction": 0,
                "depth_reduction_from_naive": 0
            }
    
    def run_qiskit_transpiler(self, circuit, coupling_map, basis_gates, optimization_level):
        """
        Run the Qiskit transpiler benchmark.
        
        Args:
            circuit: The circuit to transpile
            coupling_map: Coupling map for the topology
            basis_gates: Basis gates for the device
            optimization_level: Optimization level (0-3)
        
        Returns:
            dict: Benchmark results
        """
        print(f"\nRunning Qiskit transpiler benchmark (optimization level: {optimization_level})...")
        
        # Measure execution time
        start_time = time.time()
        
        try:
            # Import Qiskit transpiler
            from qiskit import transpile
            
            # Run the transpiler
            original_depth = circuit.depth()
            naive_depth = create_optimized_physical_mapping(
                circuit.copy(),
                coupling_map,
                basis_gates
            ).depth()
            
            # Transpile with specified optimization level
            transpiled_circuit = transpile(
                circuit,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                optimization_level=optimization_level
            )
            
            transpiled_depth = transpiled_circuit.depth()
            
            execution_time = time.time() - start_time
            print(f"Qiskit transpiler benchmark completed in {execution_time:.4f} seconds")
            
            # Calculate depth reduction
            depth_reduction = ((original_depth - transpiled_depth) / original_depth * 100) if original_depth > 0 else 0
            depth_reduction_from_naive = ((naive_depth - transpiled_depth) / naive_depth * 100) if naive_depth > 0 else 0
            
            print(f"Depth reduction: {depth_reduction:.2f}%")
            print(f"Depth reduction from naive mapping: {depth_reduction_from_naive:.2f}%")
            
            # Get circuit metrics
            metrics = self.compiler.get_circuit_metrics(transpiled_circuit)
            
            # Create results dictionary
            results = {
                "depth": transpiled_depth,
                "gate_count": sum(metrics["gate_counts"].values()),
                "cx_count": metrics["cx_count"],
                "t_gates": metrics["t_gates"],
                "execution_time": execution_time,
                "depth_reduction": depth_reduction,
                "depth_reduction_from_naive": depth_reduction_from_naive,
                "fidelity": metrics.get("fidelity", None),
                "optimization_level": optimization_level,
                "circuit": transpiled_circuit
            }
            
            return results
            
        except Exception as e:
            print(f"Error running Qiskit transpiler benchmark: {str(e)}")
            traceback.print_exc()
            
            # Return a minimal result dictionary with error information
            return {
                "error": str(e),
                "execution_time": time.time() - start_time,
                "depth_reduction": 0,
                "depth_reduction_from_naive": 0,
                "optimization_level": optimization_level
            }
    
    def print_comparison_summary(self, config_key):
        """
        Print a summary of the benchmark comparison.
        
        Args:
            config_key: Key for the benchmark configuration
        """
        if config_key not in self.results:
            print(f"No results found for {config_key}")
            return
        
        results = self.results[config_key]
        
        print("\n" + "="*80)
        print(f"BENCHMARK COMPARISON: {config_key}")
        print("="*80 + "\n")
        
        # Print config information
        config = results["config"]
        
        # Handle both the original benchmark and comprehensive benchmark format
        if "network" in config:
            print(f"Network: {config['network']}")
        elif "network_name" in config:
            print(f"Network: {config['network_name']}")
            
        print(f"Topology: {config['topology']}")
        
        # Safely access optimization_level which might not be present in comprehensive benchmark
        if "optimization_level" in config:
            print(f"Qiskit optimization level: {config['optimization_level']}")
        else:
            print(f"Qiskit optimization level: 3 (default)")
            
        print(f"Toffoli optimizer target fidelity: {config.get('target_fidelity', 0.95)}")
        print()
        
        # Print Toffoli Optimizer results
        if "toffoli_optimizer" in results and "error" not in results["toffoli_optimizer"]:
            toffoli_results = results["toffoli_optimizer"]
            print("TOFFOLI DEPTH OPTIMIZER:")
            print(f"  Execution time: {toffoli_results.get('execution_time', 0):.4f} seconds")
            
            # Handle different result structure between original and comprehensive benchmark
            if "logical" in results:
                print(f"  Logical depth: {results['logical']['depth']}")
            
            if "original" in results:
                print(f"  Naive physical depth: {results['original']['depth']}")
            
            # In comprehensive benchmark, optimized result might be accessed differently
            if "optimized" in toffoli_results:
                print(f"  Optimized logical depth: {toffoli_results['optimized']['depth']}")
            
            # The physical depth might be under 'mapped' or directly as 'physical_depth'
            if "mapped" in toffoli_results:
                print(f"  Final physical depth: {toffoli_results['mapped']['depth']}")
                gate_count = toffoli_results['mapped'].get('gate_count', 0)
                cx_count = toffoli_results['mapped'].get('cx_count', 0)
            else:
                physical_depth = toffoli_results.get('physical_depth', 0)
                print(f"  Final physical depth: {physical_depth}")
                gate_count = toffoli_results.get('physical_gate_count', 0)
                cx_count = toffoli_results.get('cx_count', 0)
            
            print(f"  Depth reduction: {toffoli_results.get('depth_reduction', 0):.2f}%")
            print(f"  Depth reduction from naive: {toffoli_results.get('depth_reduction_from_naive', 0):.2f}%")
            print(f"  Gate count: {gate_count}")
            print(f"  CNOT count: {cx_count}")
            
            # Add safe access to fidelity field - check if it exists first
            if "mapped" in toffoli_results and "fidelity" in toffoli_results["mapped"] and toffoli_results["mapped"]["fidelity"] is not None:
                print(f"  Estimated fidelity: {toffoli_results['mapped']['fidelity']:.6f}")
            elif "fidelity" in toffoli_results and toffoli_results["fidelity"] is not None:
                print(f"  Estimated fidelity: {toffoli_results['fidelity']:.6f}")
            else:
                print(f"  Estimated fidelity: Not available")
            
            print()
        else:
            print("TOFFOLI DEPTH OPTIMIZER: Error or no results available")
            print()
        
        # Print Qiskit transpiler results
        if "qiskit_transpiler" in results and "error" not in results["qiskit_transpiler"]:
            qiskit_results = results["qiskit_transpiler"]
            print("QISKIT TRANSPILER:")
            print(f"  Execution time: {qiskit_results.get('execution_time', 0):.4f} seconds")
            
            if "logical" in results:
                print(f"  Original depth: {results['logical']['depth']}")
            
            if "original" in results:
                print(f"  Naive physical depth: {results['original']['depth']}")
            
            print(f"  Transpiled depth: {qiskit_results.get('depth', 0)}")
            print(f"  Depth reduction: {qiskit_results.get('depth_reduction', 0):.2f}%")
            print(f"  Depth reduction from naive: {qiskit_results.get('depth_reduction_from_naive', 0):.2f}%")
            print(f"  Gate count: {qiskit_results.get('gate_count', 0)}")
            print(f"  CNOT count: {qiskit_results.get('cx_count', 0)}")
            
            # Add safe access to fidelity field - check if it exists first
            if "fidelity" in qiskit_results and qiskit_results["fidelity"] is not None:
                print(f"  Estimated fidelity: {qiskit_results['fidelity']:.6f}")
            else:
                print(f"  Estimated fidelity: Not available")
            
            print()
        else:
            print("QISKIT TRANSPILER: Error or no results available")
            print()
        
        # Print comparison metrics
        if ("toffoli_optimizer" in results and "error" not in results["toffoli_optimizer"] and
            "qiskit_transpiler" in results and "error" not in results["qiskit_transpiler"]):
            
            # Get depth values, handling different result structures
            if "mapped" in results["toffoli_optimizer"]:
                toffoli_depth = results["toffoli_optimizer"]["mapped"]["depth"]
            else:
                toffoli_depth = results["toffoli_optimizer"].get("physical_depth", 0)
            
            qiskit_depth = results["qiskit_transpiler"].get("depth", 0)
            
            # Get gate counts, handling different result structures
            if "mapped" in results["toffoli_optimizer"]:
                toffoli_gates = results["toffoli_optimizer"]["mapped"].get("gate_count", 0)
            else:
                toffoli_gates = results["toffoli_optimizer"].get("physical_gate_count", 0)
            
            qiskit_gates = results["qiskit_transpiler"].get("gate_count", 0)
            
            # Calculate improvements
            if qiskit_depth > 0:
                depth_improvement = ((qiskit_depth - toffoli_depth) / qiskit_depth * 100)
            else:
                depth_improvement = 0
                
            if qiskit_gates > 0:
                gate_improvement = ((qiskit_gates - toffoli_gates) / qiskit_gates * 100)
            else:
                gate_improvement = 0
            
            print("COMPARISON:")
            print(f"  Depth improvement of Toffoli Optimizer over Qiskit: {depth_improvement:.2f}%")
            print(f"  Gate count {'reduction' if gate_improvement > 0 else 'increase'}: {abs(gate_improvement):.2f}%")
        
        print("="*80)
    
    def _save_results(self):
        """Save the benchmark results to disk"""
        results_dir = os.path.join(self.config["output_dir"], f"benchmark_{self.timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        serializable_results = {}
        
        # Create a serializable version of the results (without circuit objects)
        for key, value in self.results.items():
            serializable_results[key] = self._make_serializable(value)
            
        with open(os.path.join(results_dir, "all_results.json"), "w") as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"Results saved to {os.path.join(results_dir, 'all_results.json')}")
    
    def _make_serializable(self, obj):
        """Make an object JSON serializable by removing non-serializable components"""
        if isinstance(obj, dict):
            serializable_dict = {}
            for key, value in obj.items():
                # Skip circuit objects and other non-serializable things
                if key in ["circuit", "logical_circuit", "naive_physical_circuit", "mapped_circuit",
                          "optimized_circuit", "coupling_map"]:
                    continue
                serializable_dict[key] = self._make_serializable(value)
            return serializable_dict
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Convert other objects to string representation
            return str(obj)


class ComprehensiveBenchmark(ComparisonBenchmark):
    """
    Extended benchmark class for comprehensive parameter sweeps.
    
    This class provides functionality for running comprehensive benchmarks
    across various topologies, fidelities, and network sizes.
    """
    
    def __init__(self, config=None):
        """Initialize with parent class configuration"""
        super().__init__(config)
        
        # Add default configurations for the parameter sweep
        sweep_defaults = {
            "topologies": ['linear', 'grid', 'falcon'],
            "fidelities": [0.90, 0.95, 0.99],
            "network_sizes": {
                "small": {"gates": 5, "qubits": 8},
                "medium": {"gates": 10, "qubits": 12},
                "large": {"gates": 20, "qubits": 20}
            },
            "networks_per_size": 5,
            "loaded_networks": []  # Empty by default
        }
        
        # Update with user configuration if provided
        self.sweep_config = sweep_defaults
        if config and "sweep" in config:
            for key, value in config["sweep"].items():
                self.sweep_config[key] = value
        
        # Mark as comprehensive mode to handle results properly
        self.comprehensive_mode = True
        
        # Initialize network cache to store generated networks
        self.network_cache = {}


def run_simple_benchmark(toffoli_gates, output_qubits, input_qubits, num_qubits,
                      topology='linear', target_fidelity=0.95, output_dir=None,
                      debug_mode=False):
    """
    Run a simple benchmark of the Toffoli Depth Optimizer.
    
    This function provides a simpler interface to run a benchmark without needing
    to create and configure benchmark classes directly.
    
    Args:
        toffoli_gates (list): List of Toffoli gates to optimize
        output_qubits (list): List of output qubit indices
        input_qubits (list): List of input qubit indices
        num_qubits (int): Number of qubits in the circuit
        topology (str): Hardware topology ('linear', 'grid', 'falcon')
        target_fidelity (float): Target fidelity for optimization
        output_dir (str): Directory for benchmark output
        debug_mode (bool): Whether to print debug information
        
    Returns:
        dict: Dictionary with benchmark results
    """
    # Configure the benchmark
    config = {
        "output_dir": output_dir or "benchmark_results",
        "repetitions": 1,
        "max_passes": 2,
        "target_fidelity": target_fidelity,
        "use_parallel": False,
        "debug_mode": debug_mode
    }
    
    # Create the benchmark
    benchmark = ComparisonBenchmark(config)
    
    # Create a network config function
    def network_func():
        return (toffoli_gates, output_qubits, input_qubits)
    
    network_config = {
        "define_func": network_func,
        "num_qubits": num_qubits
    }
    
    # Run the benchmark
    print(f"Running benchmark for {len(toffoli_gates)} Toffoli gates on {num_qubits} qubits...")
    
    results = benchmark.run_benchmark(
        network_name="custom_network",
        network_config=network_config,
        topology=topology,
        optimization_level=3,
        target_fidelity=target_fidelity
    )
    
    # Print a summary of the results
    if results:
        print("\nBenchmark Results:")
        print("=" * 60)
        
        # Toffoli Optimizer results
        if "toffoli_optimizer" in results and "error" not in results["toffoli_optimizer"]:
            toffoli = results["toffoli_optimizer"]
            print("Toffoli Optimizer:")
            print(f"  Logical depth: {results['logical']['depth']}")
            print(f"  Naive physical depth: {results['original']['depth']}")
            
            # The optimized depth might be under different keys
            if "optimized" in toffoli:
                print(f"  Optimized logical depth: {toffoli['optimized']['depth']}")
            
            # The physical depth might be under 'mapped' or directly
            if "mapped" in toffoli:
                print(f"  Final physical depth: {toffoli['mapped']['depth']}")
                print(f"  Final gate count: {toffoli['mapped']['gate_count']}")
                print(f"  Final CNOT count: {toffoli['mapped']['cx_count']}")
            
            print(f"  Depth reduction: {toffoli.get('depth_reduction', 0):.2f}%")
            print(f"  Depth reduction from naive mapping: {toffoli.get('depth_reduction_from_naive', 0):.2f}%")
        
        # Qiskit Transpiler results
        if "qiskit_transpiler" in results and "error" not in results["qiskit_transpiler"]:
            qiskit = results["qiskit_transpiler"]
            print("\nQiskit Transpiler:")
            print(f"  Transpiled depth: {qiskit.get('depth', 0)}")
            print(f"  Gate count: {qiskit.get('gate_count', 0)}")
            print(f"  CNOT count: {qiskit.get('cx_count', 0)}")
            print(f"  Depth reduction: {qiskit.get('depth_reduction', 0):.2f}%")
        
        print("=" * 60)
    
    return results
