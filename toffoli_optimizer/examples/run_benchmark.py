#!/usr/bin/env python3
"""
Run Benchmark Example

This script demonstrates how to run benchmarks with the Toffoli Depth Optimizer,
comparing its performance against Qiskit's transpiler for various circuit configurations.
"""

import os
import argparse
import sys
import time
from datetime import datetime

# Add parent directory to path for imports when run as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from the toffoli_optimizer package
from toffoli_optimizer.benchmark.benchmarks import ComparisonBenchmark
from toffoli_optimizer.utils.circuit_utils import generate_random_toffoli_network


def run_simple_benchmark():
    """Run a simple benchmark with default parameters"""
    print("\n=== Running Simple Benchmark ===")
    
    # Configure the benchmark
    config = {
        "output_dir": "benchmark_results",
        "repetitions": 3,
        "max_passes": 2,
        "target_fidelity": 0.95,
        "use_parallel": True,
        "debug_mode": False
    }
    
    # Initialize the benchmark
    benchmark = ComparisonBenchmark(config)
    
    # Generate a small random Toffoli network
    print("Generating random Toffoli network...")
    toffoli_gates, output_qubits, input_qubits, num_qubits = generate_random_toffoli_network(
        num_gates=5,
        target_logical_depth=3,
        num_qubits=8
    )
    
    print(f"Created network with {len(toffoli_gates)} Toffoli gates on {num_qubits} qubits")
    
    # Define the network
    network_config = {
        "define_func": lambda: (toffoli_gates, output_qubits, input_qubits),
        "num_qubits": num_qubits
    }
    
    # Run the benchmark for a single configuration
    benchmark.run_benchmark(
        network_name="random_network",
        network_config=network_config,
        topology="linear",
        optimization_level=3,
        target_fidelity=0.95
    )
    
    # Generate a report
    benchmark.generate_report()
    
    return benchmark.results


def run_comparative_benchmark(topologies=None, target_fidelities=None, output_dir=None):
    """
    Run a more comprehensive benchmark comparing multiple configurations.
    
    Args:
        topologies: List of topologies to benchmark (default: ["linear", "grid"])
        target_fidelities: List of target fidelities (default: [0.9, 0.95])
        output_dir: Directory to save results (default: "comparative_benchmark_results")
    """
    if topologies is None:
        topologies = ["linear", "grid"]
    
    if target_fidelities is None:
        target_fidelities = [0.9, 0.95]
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"comparative_benchmark_results_{timestamp}"
    
    print(f"\n=== Running Comparative Benchmark ===")
    print(f"Topologies: {', '.join(topologies)}")
    print(f"Target fidelities: {', '.join([str(f) for f in target_fidelities])}")
    print(f"Output directory: {output_dir}")
    
    # Configure the benchmark
    config = {
        "output_dir": output_dir,
        "repetitions": 2,
        "max_passes": 2,
        "target_fidelity": 0.95,  # Default, will be overridden
        "use_parallel": True,
        "debug_mode": False
    }
    
    # Initialize the benchmark
    benchmark = ComparisonBenchmark(config)
    
    # Generate a small random Toffoli network
    print("Generating random Toffoli network...")
    toffoli_gates, output_qubits, input_qubits, num_qubits = generate_random_toffoli_network(
        num_gates=5,
        target_logical_depth=3,
        num_qubits=8
    )
    
    # Define the network
    network_config = {
        "define_func": lambda: (toffoli_gates, output_qubits, input_qubits),
        "num_qubits": num_qubits
    }
    
    # Run benchmarks for each configuration
    start_time = time.time()
    
    for topology in topologies:
        for fidelity in target_fidelities:
            try:
                benchmark.run_benchmark(
                    network_name="random_network",
                    network_config=network_config,
                    topology=topology,
                    optimization_level=3,
                    target_fidelity=fidelity
                )
            except Exception as e:
                print(f"Error running benchmark for topology={topology}, fidelity={fidelity}: {e}")
    
    total_time = time.time() - start_time
    print(f"\nCompleted all benchmarks in {total_time:.2f} seconds")
    
    # Generate a report
    benchmark.generate_report()
    
    return benchmark.results


def main():
    """Main function when script is run directly"""
    parser = argparse.ArgumentParser(description='Run Toffoli Optimizer Benchmarks')
    parser.add_argument('--mode', type=str, default='simple',
                       choices=['simple', 'comparative'],
                       help='Benchmark mode: simple or comparative')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for benchmark results')
    parser.add_argument('--topologies', type=str, default='linear,grid',
                       help='Comma-separated list of topologies to benchmark')
    parser.add_argument('--fidelities', type=str, default='0.9,0.95',
                       help='Comma-separated list of target fidelities')
    
    args = parser.parse_args()
    
    if args.mode == 'simple':
        run_simple_benchmark()
    elif args.mode == 'comparative':
        # Parse topologies and fidelities
        topologies = args.topologies.split(',')
        target_fidelities = [float(f) for f in args.fidelities.split(',')]
        
        run_comparative_benchmark(
            topologies=topologies,
            target_fidelities=target_fidelities,
            output_dir=args.output_dir
        )
    else:
        print(f"Unknown mode: {args.mode}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
