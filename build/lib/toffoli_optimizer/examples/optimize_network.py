#!/usr/bin/env python3
"""
Toffoli Network Optimization Example

This example demonstrates how to optimize a Toffoli network using the ToffoliDepthOptimizer.
"""

import os
import time
from qiskit import QuantumCircuit

# Import from the reorganized project
from toffoli_optimizer.core import ToffoliCompiler, ToffoliDepthOptimizer, ToffoliType
from toffoli_optimizer.utils import save_circuit_to_qasm, save_circuit_image

def run_optimization_example():
    """Run a simple optimization example"""
    print("=== Toffoli Network Optimization Example ===")
    
    # Define a simple Toffoli network (2-bit adder)
    toffoli_gates = [
        ([0, 1], 2),  # Controls: 0,1; Target: 2
        ([0, 1], 3),  # Controls: 0,1; Target: 3
        ([0, 2], 4),  # Controls: 0,2; Target: 4
        ([1, 2], 5),  # Controls: 1,2; Target: 5
        ([2, 3], 6),  # Controls: 2,3; Target: 6
        ([4, 5], 7),  # Controls: 4,5; Target: 7
    ]
    num_qubits = 8
    output_qubits = [3, 6, 7]  # Sum bits and carry
    input_qubits = [0, 1, 2]   # Input bits
    
    # Create a compiler and optimizer
    compiler = ToffoliCompiler()
    optimizer = ToffoliDepthOptimizer(
        target_fidelity=0.95,
        max_passes=2,
        output_dir="example_results",
        debug_mode=False
    )
    optimizer.compiler = compiler
    
    print(f"Optimizing a 2-bit adder with {len(toffoli_gates)} Toffoli gates and {num_qubits} qubits...")
    
    # Run optimization for different topologies
    topologies = ["linear", "grid", "falcon"]
    
    for topology in topologies:
        print(f"\nRunning optimization for {topology} topology...")
        start_time = time.time()
        
        # Run the optimization
        results = optimizer.optimize_toffoli_network(
            toffoli_gates,
            output_qubits,
            input_qubits,
            num_qubits,
            topology=topology
        )
        
        # Extract results
        logical_depth = results["logical"]["depth"]
        naive_depth = results["naive_physical"]["depth"]
        optimized_depth = results["optimized"]["depth"]
        physical_depth = results["mapped"]["depth"]
        
        # Calculate reductions
        depth_reduction = ((logical_depth - optimized_depth) / logical_depth * 100) if logical_depth > 0 else 0
        depth_reduction_from_naive = ((naive_depth - physical_depth) / naive_depth * 100) if naive_depth > 0 else 0
        
        # Print results
        print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
        print(f"Logical depth: {logical_depth}")
        print(f"Naive physical depth: {naive_depth}")
        print(f"Optimized logical depth: {optimized_depth}")
        print(f"Final physical depth: {physical_depth}")
        print(f"Depth reduction from logical: {depth_reduction:.2f}%")
        print(f"Depth reduction from naive: {depth_reduction_from_naive:.2f}%")
        
        # Save the optimized circuit
        output_dir = os.path.join("example_results", topology)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"optimized_circuit")
        save_circuit_to_qasm(results["final"]["circuit"], output_path)
        print(f"Optimized circuit saved to {output_path}.qasm")
        
        # Save circuit visualizations
        try:
            save_circuit_image(results["logical"]["circuit"], "logical_circuit", output_dir)
            save_circuit_image(results["naive_physical"]["circuit"], "naive_physical_circuit", output_dir)
            save_circuit_image(results["optimized"]["circuit"], "optimized_circuit", output_dir)
            save_circuit_image(results["mapped"]["circuit"], "mapped_circuit", output_dir)
            print(f"Circuit visualizations saved to {output_dir}")
        except Exception as e:
            print(f"Warning: Failed to generate visualizations: {e}")
    
    print("\n=== Example completed successfully ===")

def create_toffoli_network_from_scratch():
    """Create a Toffoli network from a quantum circuit definition"""
    print("\n=== Creating Toffoli Network from Quantum Circuit ===")
    
    # Create a quantum circuit
    qc = QuantumCircuit(4)
    
    # Add some Toffoli gates
    qc.ccx(0, 1, 2)
    qc.ccx(1, 2, 3)
    qc.ccx(0, 3, 1)
    
    # Extract Toffoli gates
    toffoli_gates = []
    for instruction in qc.data:
        if instruction.operation.name == 'ccx':
            control1 = instruction.qubits[0].index
            control2 = instruction.qubits[1].index
            target = instruction.qubits[2].index
            toffoli_gates.append(([control1, control2], target))
    
    print(f"Extracted {len(toffoli_gates)} Toffoli gates from circuit")
    for i, (controls, target) in enumerate(toffoli_gates):
        print(f"  Gate {i+1}: Controls {controls}, Target {target}")
    
    # Define I/O qubits
    num_qubits = qc.num_qubits
    input_qubits = [0, 1]
    output_qubits = [2, 3]
    
    # Now optimize the network
    compiler = ToffoliCompiler()
    optimizer = ToffoliDepthOptimizer(target_fidelity=0.95)
    optimizer.compiler = compiler
    
    # Run optimization
    results = optimizer.optimize_toffoli_network(
        toffoli_gates,
        output_qubits,
        input_qubits,
        num_qubits,
        topology='linear'
    )
    
    # Print results
    logical_depth = results["logical"]["depth"]
    optimized_depth = results["optimized"]["depth"]
    
    print(f"Original circuit depth: {qc.depth()}")
    print(f"Logical circuit depth: {logical_depth}")
    print(f"Optimized circuit depth: {optimized_depth}")
    print(f"Depth reduction: {((logical_depth - optimized_depth) / logical_depth * 100) if logical_depth > 0 else 0:.2f}%")

if __name__ == "__main__":
    # Run the examples
    run_optimization_example()
    create_toffoli_network_from_scratch()
