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
