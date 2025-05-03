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
from ..utils.io_utils import save_circuit
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
    
    def run_full_benchmark(self, coupling_map_provider=None):
        """
        Run the full benchmark with all configurations.
        
        Args:
            coupling_map_provider: Function that returns a coupling map for a topology
        
        Returns:
            dict: Dictionary with benchmark results
        """
        print("\n" + "="*80)
        print("RUNNING FULL BENCHMARK")
        print("="*80)
        
        # Define networks to benchmark
        networks = {
            "adder_2bit": {
                "define_func": "define_toffoli_network",  # 2-bit adder
                "num_qubits": 8
            },
            "variable_small": {
                "define_func": "define_variable_toffoli_network",
                "args": {"num_gates": 5, "num_qubits": 8},
                "num_qubits": 8
            },
            "variable_medium": {
                "define_func": "define_variable_toffoli_network",
                "args": {"num_gates": 10, "num_qubits": 12},
                "num_qubits": 12
            }
        }
        
        # Define topologies to benchmark
        topologies = ["linear", "grid", "falcon"]
        
        # Define Qiskit optimization levels
        optimization_levels = [1, 3]
        
        # Define target fidelities
        target_fidelities = [0.9, 0.95, 0.99]
        
        # Import network generators
        from ..utils.network_generators import define_toffoli_network, define_variable_toffoli_network
        
        # Run benchmarks for all combinations
        for network_name, network_config in networks.items():
            for topology in topologies:
                for opt_level in optimization_levels:
                    for fidelity in target_fidelities:
                        try:
                            # Get the appropriate define function
                            if network_config["define_func"] == "define_toffoli_network":
                                define_func = define_toffoli_network
                            elif network_config["define_func"] == "define_variable_toffoli_network":
                                args = network_config.get("args", {})
                                define_func = lambda: define_variable_toffoli_network(**args)
                            else:
                                raise ValueError(f"Unknown define function: {network_config['define_func']}")
                                
                            self.run_benchmark(
                                network_name,
                                define_func,
                                network_config["num_qubits"],
                                topology,
                                opt_level,
                                fidelity,
                                coupling_map_provider
                            )
                        except Exception as e:
                            print(f"Error running benchmark for {network_name} on {topology}: {e}")
                            traceback.print_exc()
        
        # Generate comprehensive report
        self.generate_report()
        
        return self.results
    
    def run_benchmark(self, network_name, define_func, num_qubits, topology,
                      optimization_level, target_fidelity, coupling_map_provider=None):
        """
        Run a specific benchmark configuration.
        
        Args:
            network_name: Name of the Toffoli network
            define_func: Function that defines the Toffoli network
            num_qubits: Number of qubits in the network
            topology: Target topology ('linear', 'grid', 'falcon')
            optimization_level: Qiskit optimization level (0-3)
            target_fidelity: Target fidelity for optimization
            coupling_map_provider: Function that returns a coupling map for a topology
        
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
            toffoli_network = define_func()
            if isinstance(toffoli_network, tuple) and len(toffoli_network) >= 3:
                toffoli_gates, output_qubits, input_qubits = toffoli_network
            else:
                raise ValueError("Invalid Toffoli network format returned by define_func")
        except Exception as e:
            print(f"Error defining Toffoli network: {e}")
            traceback.print_exc()
            return None
            
        # Get coupling map for the selected topology
        if coupling_map_provider:
            coupling_map = coupling_map_provider(topology, num_qubits)
        else:
            # Import default provider if none specified
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
            depth_improvement = ((qiskit_depth - toffoli_depth) / qiskit_depth * 100) if qiskit_depth > 0 else 0
            gate_improvement = ((qiskit_gates - toffoli_gates) / qiskit_gates * 100) if qiskit_gates > 0 else 0
            
            print("COMPARISON:")
            print(f"  Depth improvement of Toffoli Optimizer over Qiskit: {depth_improvement:.2f}%")
            print(f"  Gate count {'reduction' if gate_improvement > 0 else 'increase'}: {abs(gate_improvement):.2f}%")
        
        print("="*80)
    
    def generate_report(self):
        """Generate a comprehensive report of all benchmark results"""
        print("\nGenerating benchmark report...")
        
        # Create a directory for the report
        report_dir = os.path.join(self.config["output_dir"], f"report_{self.timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Create visualizations directory
        vis_dir = os.path.join(report_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate summary report
        self._generate_summary_report(report_dir)
        
        # Generate visualizations
        self._generate_visualizations(vis_dir)
        
        # Save the full results
        self._save_results()
        
        print(f"Benchmark report generated at {report_dir}")
        
    def _generate_summary_report(self, report_dir):
        """Generate a summary report of the benchmark results"""
        with open(os.path.join(report_dir, "summary.txt"), "w") as f:
            f.write("="*80 + "\n")
            f.write("TOFFOLI DEPTH OPTIMIZER BENCHMARK SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration:\n")
            for key, value in self.config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Aggregate results by topology and network
            topologies = set()
            networks = set()
            
            for config_key in self.results:
                config = self.results[config_key]["config"]
                topologies.add(config["topology"])
                networks.add(config["network"])
            
            # Summary by topology
            f.write("="*80 + "\n")
            f.write("SUMMARY BY TOPOLOGY\n")
            f.write("="*80 + "\n\n")
            
            for topology in sorted(topologies):
                f.write(f"Topology: {topology}\n")
                f.write("-"*80 + "\n")
                
                # Calculate average metrics for this topology
                toffoli_depths = []
                qiskit_depths = []
                toffoli_improvements = []
                
                for config_key, result in self.results.items():
                    if result["config"]["topology"] == topology:
                        if ("toffoli_optimizer" in result and "error" not in result["toffoli_optimizer"] and
                            "qiskit_transpiler" in result and "error" not in result["qiskit_transpiler"]):
                            
                            # Get toffoli depth (account for different result structures)
                            if "mapped" in result["toffoli_optimizer"]:
                                toffoli_depth = result["toffoli_optimizer"]["mapped"]["depth"]
                            else:
                                toffoli_depth = result["toffoli_optimizer"].get("physical_depth", 0)
                            
                            qiskit_depth = result["qiskit_transpiler"]["depth"]
                            toffoli_depths.append(toffoli_depth)
                            qiskit_depths.append(qiskit_depth)
                            
                            # Calculate improvement
                            improvement = ((qiskit_depth - toffoli_depth) / qiskit_depth * 100) if qiskit_depth > 0 else 0
                            toffoli_improvements.append(improvement)
                
                if toffoli_depths and qiskit_depths:
                    avg_toffoli_depth = sum(toffoli_depths) / len(toffoli_depths)
                    avg_qiskit_depth = sum(qiskit_depths) / len(qiskit_depths)
                    avg_improvement = sum(toffoli_improvements) / len(toffoli_improvements)
                    
                    f.write(f"Average Toffoli Optimizer depth: {avg_toffoli_depth:.2f}\n")
                    f.write(f"Average Qiskit transpiler depth: {avg_qiskit_depth:.2f}\n")
                    f.write(f"Average improvement: {avg_improvement:.2f}%\n")
                else:
                    f.write("No valid results for this topology\n")
                
                f.write("\n")
            
            # Summary by network
            f.write("="*80 + "\n")
            f.write("SUMMARY BY NETWORK\n")
            f.write("="*80 + "\n\n")
            
            # Similar implementation for network summary as for topology
            for network in sorted(networks):
                f.write(f"Network: {network}\n")
                f.write("-"*80 + "\n")
                
                # Calculate metrics for each network...
                # (similar to topology metrics calculation)
                
            # Overall summary
            f.write("="*80 + "\n")
            f.write("OVERALL SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Calculate overall metrics across all benchmarks...
    
    def _generate_visualizations(self, vis_dir):

        """Generate visualizations of the benchmark results"""
        # Prepare data for visualization
        data = defaultdict(list)
        
        for config_key, result in self.results.items():
            if ("toffoli_optimizer" in result and "error" not in result["toffoli_optimizer"] and
                "qiskit_transpiler" in result and "error" not in result["qiskit_transpiler"]):
                
                config = result["config"]
                topology = config["topology"]
                network = config["network"]
                opt_level = config["optimization_level"]
                
                # Extract metrics (handling different result structures)
                if "mapped" in result["toffoli_optimizer"]:
                    toffoli_depth = result["toffoli_optimizer"]["mapped"]["depth"]
                else:
                    toffoli_depth = result["toffoli_optimizer"].get("physical_depth", 0)
                
                qiskit_depth = result["qiskit_transpiler"]["depth"]
                
                # Calculate improvement percentage
                improvement = ((qiskit_depth - toffoli_depth) / qiskit_depth * 100) if qiskit_depth > 0 else 0
                
                # Store data for visualization
                data["topologies"].append(topology)
                data["networks"].append(network)
                data["opt_levels"].append(opt_level)
                data["toffoli_depths"].append(toffoli_depth)
                data["qiskit_depths"].append(qiskit_depth)
                data["improvements"].append(improvement)
        
        # Create visualizations if we have data
        if data["topologies"]:
            # Topology comparisons
            self._create_topology_comparison_plots(data, vis_dir)
            
            # Network comparisons
            self._create_network_comparison_plots(data, vis_dir)
            
            # Overall comparison
            self._create_overall_comparison_plots(data, vis_dir)
    
    def _create_topology_comparison_plots(self, data, vis_dir):

        """Create topology comparison visualizations"""
        # Implementation for topology visualization plots
        pass
    
    def _create_network_comparison_plots(self, data, vis_dir):

        """Create network comparison visualizations"""
        # Implementation for network visualization plots
        pass
    

    def _create_overall_comparison_plots(self, data, vis_dir):

        """Create overall benchmark comparison visualizations"""
        # Implementation for overall visualization plots
        pass
    

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


# Include comprehensive benchmark class
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
    
    def run_comprehensive_benchmark(self):
        """
        Run a comprehensive benchmark with parameter sweep.
        
        This method runs benchmarks across all combinations of topologies,
        fidelities, and network sizes specified in the configuration.
        
        Returns:
            dict: Dictionary with all benchmark results
        """
        # Implementation for comprehensive benchmark
        pass


    def _create_topology_comparison_plots(self, data, vis_dir):
        """Create topology comparison visualizations"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Get unique topologies
        topologies = sorted(list(set(data["topologies"])))

        # Create depth comparison plot for different topologies
        plt.figure(figsize=(12, 8))

        # Group data by topology
        topo_depths = {topo: {'toffoli': [], 'qiskit': []} for topo in topologies}

        for i in range(len(data["topologies"])):
            topo = data["topologies"][i]
            topo_depths[topo]['toffoli'].append(data["toffoli_depths"][i])
            topo_depths[topo]['qiskit'].append(data["qiskit_depths"][i])

        # Get average depths
        avg_toffoli = [np.mean(topo_depths[topo]['toffoli']) for topo in topologies]
        avg_qiskit = [np.mean(topo_depths[topo]['qiskit']) for topo in topologies]

        # Set up bar chart
        x = np.arange(len(topologies))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        toffoli_bars = ax.bar(x - width/2, avg_toffoli, width, label='Toffoli Optimizer')
        qiskit_bars = ax.bar(x + width/2, avg_qiskit, width, label='Qiskit Transpiler')

        # Add labels and title
        ax.set_xlabel('Topology')
        ax.set_ylabel('Average Circuit Depth')
        ax.set_title('Circuit Depth by Topology')
        ax.set_xticks(x)
        ax.set_xticklabels(topologies)
        ax.legend()

        # Add value labels on bars
        for bar in toffoli_bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')

        for bar in qiskit_bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{vis_dir}/topology_depth_comparison.png")
        plt.close()

        # Create improvement percentage plot
        improvements = []
        for topo in topologies:
            toffoli_avg = np.mean(topo_depths[topo]['toffoli'])
            qiskit_avg = np.mean(topo_depths[topo]['qiskit'])
            if qiskit_avg > 0:
                improvement = ((qiskit_avg - toffoli_avg) / qiskit_avg) * 100
            else:
                improvement = 0
            improvements.append(improvement)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(topologies, improvements, color='green')

        # Add labels and title
        plt.xlabel('Topology')
        plt.ylabel('Depth Improvement (%)')
        plt.title('Toffoli Optimizer Improvement Over Qiskit by Topology')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{vis_dir}/topology_improvement.png")
        plt.close()

    def _create_network_comparison_plots(self, data, vis_dir):
        """Create network comparison visualizations"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Get unique networks
        networks = sorted(list(set(data["networks"])))

        # Create depth comparison plot for different networks
        plt.figure(figsize=(12, 8))

        # Group data by network
        network_depths = {net: {'toffoli': [], 'qiskit': []} for net in networks}

        for i in range(len(data["networks"])):
            net = data["networks"][i]
            network_depths[net]['toffoli'].append(data["toffoli_depths"][i])
            network_depths[net]['qiskit'].append(data["qiskit_depths"][i])

        # Get average depths
        avg_toffoli = [np.mean(network_depths[net]['toffoli']) for net in networks]
        avg_qiskit = [np.mean(network_depths[net]['qiskit']) for net in networks]

        # Set up bar chart
        x = np.arange(len(networks))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        toffoli_bars = ax.bar(x - width/2, avg_toffoli, width, label='Toffoli Optimizer')
        qiskit_bars = ax.bar(x + width/2, avg_qiskit, width, label='Qiskit Transpiler')

        # Add labels and title
        ax.set_xlabel('Network Type')
        ax.set_ylabel('Average Circuit Depth')
        ax.set_title('Circuit Depth by Network Type')
        ax.set_xticks(x)
        ax.set_xticklabels(networks)
        ax.legend()

        # Add value labels on bars
        for bar in toffoli_bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')

        for bar in qiskit_bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{vis_dir}/network_depth_comparison.png")
        plt.close()

        # Create improvement percentage plot
        improvements = []
        for net in networks:
            toffoli_avg = np.mean(network_depths[net]['toffoli'])
            qiskit_avg = np.mean(network_depths[net]['qiskit'])
            if qiskit_avg > 0:
                improvement = ((qiskit_avg - toffoli_avg) / qiskit_avg) * 100
            else:
                improvement = 0
            improvements.append(improvement)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(networks, improvements, color='green')

        # Add labels and title
        plt.xlabel('Network Type')
        plt.ylabel('Depth Improvement (%)')
        plt.title('Toffoli Optimizer Improvement Over Qiskit by Network Type')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{vis_dir}/network_improvement.png")
        plt.close()

    def _create_overall_comparison_plots(self, data, vis_dir):
        """Create overall benchmark comparison visualizations"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Calculate overall statistics
        toffoli_depths = np.array(data["toffoli_depths"])
        qiskit_depths = np.array(data["qiskit_depths"])

        # Create boxplot comparison
        plt.figure(figsize=(10, 6))

        box_data = [toffoli_depths, qiskit_depths]
        labels = ['Toffoli Optimizer', 'Qiskit Transpiler']

        box = plt.boxplot(box_data, patch_artist=True, labels=labels)

        # Fill boxes with colors
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)

        # Add labels and title
        plt.ylabel('Circuit Depth')
        plt.title('Overall Circuit Depth Comparison')

        # Add statistical annotations
        toffoli_mean = np.mean(toffoli_depths)
        qiskit_mean = np.mean(qiskit_depths)

        plt.figtext(0.15, 0.01, f'Toffoli: Mean={toffoli_mean:.2f}, Median={np.median(toffoli_depths):.2f}',
                   horizontalalignment='left', fontsize=10)
        plt.figtext(0.65, 0.01, f'Qiskit: Mean={qiskit_mean:.2f}, Median={np.median(qiskit_depths):.2f}',
                   horizontalalignment='left', fontsize=10)

        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust for text at bottom
        plt.savefig(f"{vis_dir}/overall_depth_comparison.png")
        plt.close()

        # Create overall improvement chart
        improvements = []
        for i in range(len(toffoli_depths)):
            if qiskit_depths[i] > 0:
                imp = ((qiskit_depths[i] - toffoli_depths[i]) / qiskit_depths[i]) * 100
            else:
                imp = 0
            improvements.append(imp)

        # Create histogram of improvements
        plt.figure(figsize=(10, 6))
        plt.hist(improvements, bins=10, color='green', alpha=0.7, edgecolor='black')

        # Add mean line
        mean_imp = np.mean(improvements)
        plt.axvline(mean_imp, color='red', linestyle='dashed', linewidth=2, 
                   label=f'Mean: {mean_imp:.2f}%')

        # Add labels and title
        plt.xlabel('Depth Improvement (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Depth Improvements')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{vis_dir}/improvement_distribution.png")
        plt.close()

        # Create summary pie chart for improvements
        positive_imps = sum(1 for imp in improvements if imp > 0)
        neutral_imps = sum(1 for imp in improvements if imp == 0)
        negative_imps = sum(1 for imp in improvements if imp < 0)

        if positive_imps + neutral_imps + negative_imps > 0:  # Ensure we have data
            plt.figure(figsize=(10, 6))

            sizes = [positive_imps, neutral_imps, negative_imps]
            labels = ['Improvement', 'No Change', 'Regression']
            colors = ['green', 'gray', 'red']
            explode = (0.1, 0, 0)  # explode the 'Improvement' slice

            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

            plt.title('Comparison Results Overview')
            plt.tight_layout()
            plt.savefig(f"{vis_dir}/results_overview.png")
            plt.close()


# Export benchmark classes
__all__ = ['ToffoliBenchmark', 'ComparisonBenchmark', 'ComprehensiveBenchmark']
