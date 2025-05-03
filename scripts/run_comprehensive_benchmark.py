#!/usr/bin/env python3
"""
Comprehensive Benchmark for Toffoli Circuit Optimization

This script runs a comprehensive benchmark of the Toffoli Depth Optimizer
against Qiskit transpiler across different topologies, fidelities, and
circuit sizes. The benchmark results are saved as a detailed report.

Usage:
    python run_comprehensive_benchmark.py [options]

Options:
    --output-dir DIR          Output directory for benchmark results
    --repetitions N           Number of repetitions for each configuration
    --sweep-topologies        Comma-separated list of topologies to test
    --sweep-fidelities        Comma-separated list of fidelities to test
    --sweep-sizes             Comma-separated list of size categories to test
    --loaded-networks FILES   Comma-separated list of loaded network files
    --debug                   Enable debug output
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add the parent directory to system path to enable imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import from the reorganized project structure
from toffoli_optimizer.benchmark.benchmarks import ComprehensiveBenchmark
from toffoli_optimizer.utils.io_utils import ensure_directory_exists


def parse_arguments():
    """Parse command line arguments for the benchmark."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive benchmark of Toffoli Depth Optimizer'
    )
    
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for benchmark results')
    
    parser.add_argument('--repetitions', type=int, default=3,
                       help='Number of repetitions for timing measurements (minimum 2)')
    
    parser.add_argument('--sweep-topologies', type=str, default='linear,grid,falcon',
                       help='Topologies to benchmark (comma-separated)')
    
    parser.add_argument('--sweep-fidelities', type=str, default='0.90,0.95,0.99',
                       help='Target fidelities to benchmark (comma-separated)')
    
    parser.add_argument('--sweep-sizes', type=str, default='small,medium,large',
                       help='Network sizes to benchmark (comma-separated)')
    
    parser.add_argument('--loaded-networks', type=str, default='',
                       help='Filenames of loaded networks (comma-separated, without extension)')
    
    parser.add_argument('--max-passes', type=int, default=2,
                       help='Maximum number of optimization passes (default: 2)')
    
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with more detailed output')
    
    return parser.parse_args()


def setup_benchmark_config(args):
    """Set up the benchmark configuration from parsed arguments."""
    # Parse topologies
    topologies = [t.strip() for t in args.sweep_topologies.split(',') if t.strip()]
    
    # Parse fidelities
    fidelities = []
    for f in args.sweep_fidelities.split(','):
        try:
            fidelities.append(float(f.strip()))
        except ValueError:
            continue
    
    # Parse network sizes
    sizes = [s.strip() for s in args.sweep_sizes.split(',') if s.strip()]
    
    # Parse loaded networks
    loaded_networks = [n.strip() for n in args.loaded_networks.split(',') if n.strip()]
    
    # Define network sizes
    network_sizes = {
        "small": {"gates": 5, "qubits": 8},
        "medium": {"gates": 10, "qubits": 12},
        "large": {"gates": 20, "qubits": 20}
    }
    
    # Create benchmark configuration
    config = {
        "output_dir": args.output_dir,
        "repetitions": max(2, args.repetitions),  # Ensure at least 2 repetitions
        "max_passes": max(1, args.max_passes),    # Ensure at least 1 pass
        "target_fidelity": 0.95,  # Default value
        "use_parallel": True,
        "debug_mode": args.debug,
        "sweep": {
            "topologies": topologies,
            "fidelities": fidelities,
            "network_sizes": {k: v for k, v in network_sizes.items() if k in sizes},
            "networks_per_size": 3,  # Default number of networks per size
            "loaded_networks": loaded_networks
        }
    }
    
    return config


def validate_config(config):
    """Validate the benchmark configuration."""
    errors = []
    
    # Check topologies
    if not config["sweep"]["topologies"]:
        errors.append("No topologies specified for benchmarking")
    
    # Check fidelities
    if not config["sweep"]["fidelities"]:
        errors.append("No target fidelities specified for benchmarking")
    
    # Check network sizes or loaded networks
    if not config["sweep"]["network_sizes"] and not config["sweep"]["loaded_networks"]:
        errors.append("No network sizes or loaded networks specified for benchmarking")
    
    # Verify loaded networks existence
    for network_file in config["sweep"]["loaded_networks"]:
        file_found = False
        for ext in ['.json', '.pickle', '.qasm']:
            if os.path.exists(network_file + ext):
                file_found = True
                break
        
        if not file_found:
            errors.append(f"Loaded network file not found: {network_file}.*")
    
    return errors


def print_benchmark_config(config):
    """Print the benchmark configuration."""
    print("\nBenchmark Configuration:")
    print("=" * 80)
    print(f"Output Directory: {config['output_dir']}")
    print(f"Repetitions: {config['repetitions']}")
    print(f"Maximum Optimization Passes: {config['max_passes']}")
    print(f"Debug Mode: {'Enabled' if config['debug_mode'] else 'Disabled'}")
    
    print("\nSweep Parameters:")
    print(f"  Topologies: {', '.join(config['sweep']['topologies'])}")
    print(f"  Target Fidelities: {', '.join([str(f) for f in config['sweep']['fidelities']])}")
    
    if config['sweep']['network_sizes']:
        print("\nNetwork Sizes:")
        for size, params in config['sweep']['network_sizes'].items():
            print(f"  {size.capitalize()}: {params['gates']} gates, {params['qubits']} qubits")
    
    if config['sweep']['loaded_networks']:
        print("\nLoaded Networks:")
        for network in config['sweep']['loaded_networks']:
            print(f"  {network}")
    
    print(f"\nNetworks per Size: {config['sweep']['networks_per_size']}")
    print("=" * 80)


def run_comprehensive_benchmark(config):
    """Run the comprehensive benchmark using the provided configuration."""
    print("\n" + "=" * 80)
    print("STARTING COMPREHENSIVE BENCHMARK")
    print("=" * 80)
    
    start_time = datetime.now()
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set up the benchmark
    benchmark = ComprehensiveBenchmark(config)
    
    # Run the comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Print completion message
    end_time = datetime.now()
    duration = end_time - start_time
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED")
    print(f"Total Duration: {duration}")
    print(f"Results saved to: {config['output_dir']}")
    print("=" * 80)
    
    return results


def main():
    """Main function to run the comprehensive benchmark."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up benchmark configuration
    config = setup_benchmark_config(args)
    
    # Validate configuration
    errors = validate_config(config)
    if errors:
        print("\nConfiguration Errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nBenchmark cannot proceed with invalid configuration.")
        return 1
    
    # Create output directory
    ensure_directory_exists(config["output_dir"])
    
    # Print configuration
    print_benchmark_config(config)
    
    # Confirm benchmark execution
    if not args.debug:  # Skip confirmation in debug mode
        confirmation = input("\nProceed with benchmark? This may take a long time. (y/n): ")
        if confirmation.lower() != 'y':
            print("Benchmark cancelled.")
            return 0
    
    # Run the benchmark
    try:
        run_comprehensive_benchmark(config)
        return 0
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        if config["debug_mode"]:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
