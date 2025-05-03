#!/usr/bin/env python3
"""
Toffoli Optimizer Main Script

This is the entry point for the Toffoli Optimizer project, providing command line
functionality for optimizing Toffoli networks and running benchmarks.
"""

import sys
import argparse
import os
import time
import json

from toffoli_optimizer.core.compiler import ToffoliCompiler
from toffoli_optimizer.core.optimizer import ToffoliDepthOptimizer
from toffoli_optimizer.utils.io_utils import load_toffoli_network, save_circuit_to_qasm
from toffoli_optimizer.utils.circuit_utils import get_default_coupling_map


def main():
    """Main function when script is run directly"""
    parser = argparse.ArgumentParser(description='Toffoli Depth Optimizer')
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Optimizer command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize a Toffoli network')
    optimize_parser.add_argument('--input', type=str, required=True,
                               help='Input file containing the Toffoli network (without extension)')
    optimize_parser.add_argument('--output', type=str, default='optimized_circuit.qasm',
                               help='Output file for the optimized circuit (default: optimized_circuit.qasm)')
    optimize_parser.add_argument('--topology', type=str, default='linear',
                               choices=['linear', 'grid', 'falcon'], help='Target topology')
    optimize_parser.add_argument('--num_qubits', type=int, default=None,
                               help='Number of qubits (overrides the value from input file if specified)')
    optimize_parser.add_argument('--fidelity', type=float, default=0.95,
                               help='Target fidelity for the Toffoli Depth Optimizer')
    optimize_parser.add_argument('--max_passes', type=int, default=2,
                               help='Maximum number of optimization passes for Toffoli Depth Optimizer (default: 2)')
    optimize_parser.add_argument('--debug', action='store_true',
                               help='Enable debug mode')
    optimize_parser.add_argument('--visualize', action='store_true',
                               help='Generate visualization of the optimization results')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run a benchmark')
    benchmark_parser.add_argument('--output_dir', type=str, default='toffoli_benchmark_results',
                                help='Output directory for benchmark results')
    benchmark_parser.add_argument('--network', type=str, default='adder_2bit',
                                choices=['adder_2bit', 'variable', 'loaded'], help='Toffoli network to benchmark')
    benchmark_parser.add_argument('--loaded_network_file', type=str, default='toffoli_network_circuit',
                                help='Filename (without extension) for loaded network')
    benchmark_parser.add_argument('--num_gates', type=int, default=5,
                                help='Number of gates for variable network')
    benchmark_parser.add_argument('--num_qubits', type=int, default=8,
                                help='Number of qubits for variable network')
    benchmark_parser.add_argument('--topology', type=str, default='linear',
                                choices=['linear', 'grid', 'falcon'], help='Target topology')
    benchmark_parser.add_argument('--fidelity', type=float, default=0.95,
                                help='Target fidelity for the Toffoli Depth Optimizer')
    benchmark_parser.add_argument('--debug', action='store_true',
                                help='Enable debug mode')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle the case when no command is specified
    if args.command is None:
        parser.print_help()
        return 0
    
    # Execute the chosen command
    if args.command == 'optimize':
        return run_optimizer(args)
    elif args.command == 'benchmark':
        return run_benchmark(args)
    else:
        parser.print_help()
        return 0

def run_optimizer(args):
    """Run the Toffoli Depth Optimizer on the specified input file"""
    print(f"Running Toffoli Depth Optimizer on {args.input}")
    start_time = time.time()
    
    # Initialize the optimizer
    optimizer = ToffoliDepthOptimizer(
        target_fidelity=args.fidelity,
        max_passes=args.max_passes,
        debug_mode=args.debug
    )
    
    # Load the Toffoli network
    toffoli_gates, output_qubits, input_qubits, num_qubits = load_toffoli_network(args.input)
    
    if toffoli_gates is None:
        print(f"Error: Failed to load Toffoli network from {args.input}")
        return 1
    
    # Override the number of qubits if specified
    if args.num_qubits is not None:
        if args.num_qubits < num_qubits:
            print(f"Warning: Specified number of qubits ({args.num_qubits}) is less than the number of qubits in the input file ({num_qubits})")
            print("Using the larger value to ensure all qubits are included")
            num_qubits = max(args.num_qubits, num_qubits)
        else:
            num_qubits = args.num_qubits
    
    # Get the coupling map for the specified topology
    coupling_map = get_default_coupling_map(args.topology, num_qubits)
    
    # Run the optimization
    results = optimizer.optimize_toffoli_network(
        toffoli_gates,
        output_qubits,
        input_qubits,
        num_qubits,
        topology=args.topology,
        coupling_map=coupling_map
    )
    
    if "error" in results:
        print(f"Error during optimization: {results['error']}")
        return 1
    
    # Get the optimized circuit
    optimized_circuit = results["mapped"]["circuit"]
    
    # Save the optimized circuit
    if save_circuit_to_qasm(optimized_circuit, args.output):
        print(f"Optimized circuit saved to {args.output}")
    else:
        print(f"Error: Failed to save optimized circuit to {args.output}")
        return 1
    
    # Print optimization summary
    logical_depth = results["logical"]["depth"] if "logical" in results else 0
    naive_depth = results["naive_physical"]["depth"] if "naive_physical" in results else 0
    optimized_depth = results["optimized"]["depth"] if "optimized" in results else 0
    physical_depth = results["mapped"]["depth"] if "mapped" in results else 0
    
    print("\nOptimization Summary:")
    print("-" * 80)
    print(f"Network: {args.input}")
    print(f"Topology: {args.topology}")
    print(f"Number of Toffoli gates: {len(toffoli_gates)}")
    print(f"Number of qubits: {num_qubits}")
    print(f"Target fidelity: {args.fidelity}")
    print(f"Maximum optimization passes: {args.max_passes}")
    
    print("\nDepth Metrics:")
    print(f"Logical circuit depth: {logical_depth}")
    print(f"Naive physical mapping depth: {naive_depth}")
    print(f"Optimized logical depth: {optimized_depth}")
    print(f"Final physical depth: {physical_depth}")
    
    # Calculate reductions
    depth_reduction = ((logical_depth - optimized_depth) / logical_depth * 100) if logical_depth > 0 else 0
    depth_reduction_from_naive = ((naive_depth - physical_depth) / naive_depth * 100) if naive_depth > 0 else 0
    
    print("\nOptimization Results:")
    print(f"Reduction from logical circuit: {depth_reduction:.2f}%")
    print(f"Reduction from naive mapping: {depth_reduction_from_naive:.2f}%")
    print(f"Optimization time: {results['optimization_time']:.4f} seconds")
    print("-" * 80)
    
    # Generate visualization if requested
    if args.visualize:
        try:
            from toffoli_optimizer.utils.visualization import visualize_optimization
            output_base = os.path.splitext(args.output)[0]
            visualize_optimization(optimized_circuit, results, filename=output_base)
            print(f"Visualization saved to {output_base}_*.png")
        except Exception as e:
            print(f"Error generating visualization: {e}")
    
    # Save optimization report
    report_file = os.path.splitext(args.output)[0] + "_report.json"
    try:
        # Create a serializable report
        serializable_report = {}
        for key, value in results.items():
            if key not in ["logical", "naive_physical", "optimized", "mapped", "final"]:
                serializable_report[key] = value
        
        # Add key metrics
        serializable_report["logical_depth"] = logical_depth
        serializable_report["naive_depth"] = naive_depth
        serializable_report["optimized_depth"] = optimized_depth
        serializable_report["physical_depth"] = physical_depth
        serializable_report["depth_reduction"] = depth_reduction
        serializable_report["depth_reduction_from_naive"] = depth_reduction_from_naive
        
        with open(report_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        print(f"Optimization report saved to {report_file}")
    except Exception as e:
        print(f"Error saving optimization report: {e}")
    
    return 0

def run_benchmark(args):
    """Run a benchmark with the specified configuration"""
    print("Running benchmark")
    
    # Import benchmark functionality
    try:
        from toffoli_optimizer.benchmark.benchmarks import run_simple_benchmark
        from toffoli_optimizer.utils.io_utils import define_loaded_toffoli_network
        from toffoli_optimizer.benchmark.network_generators import define_toffoli_network, define_variable_toffoli_network
    except ImportError as e:
        print(f"Error importing benchmark modules: {e}")
        print("Make sure the toffoli_optimizer package is installed correctly")
        return 1
    
    # Define the network configuration
    if args.network == 'adder_2bit':
        # 2-bit adder network
        toffoli_gates, output_qubits, input_qubits = define_toffoli_network()
        num_qubits = args.num_qubits
    elif args.network == 'variable':
        # Variable network with specified number of gates and qubits
        toffoli_gates, output_qubits, input_qubits = define_variable_toffoli_network(
            num_gates=args.num_gates,
            num_qubits=args.num_qubits
        )
        num_qubits = args.num_qubits
    elif args.network == 'loaded':
        # Load the network from a file
        toffoli_gates, output_qubits, input_qubits = define_loaded_toffoli_network(
            filename=args.loaded_network_file
        )
        num_qubits = args.num_qubits  # Use the specified number of qubits
    else:
        print(f"Error: Unknown network type '{args.network}'")
        return 1
    
    # Run the benchmark
    results = run_simple_benchmark(
        toffoli_gates=toffoli_gates,
        output_qubits=output_qubits,
        input_qubits=input_qubits,
        num_qubits=num_qubits,
        topology=args.topology,
        target_fidelity=args.fidelity,
        output_dir=args.output_dir,
        debug_mode=args.debug
    )
    
    # Save benchmark results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"benchmark_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "benchmark_results.json")
    
    try:
        # Make the results serializable
        serializable_results = {}
        for key, value in results.items():
            if not key.endswith("circuit"):  # Skip circuit objects
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Benchmark results saved to {results_file}")
    except Exception as e:
        print(f"Error saving benchmark results: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
