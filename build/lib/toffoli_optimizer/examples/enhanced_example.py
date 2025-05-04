#!/usr/bin/env python3
"""
Enhanced Toffoli Optimization Example

This example demonstrates how to use the enhanced Toffoli optimization capabilities,
including ZX-calculus and RL-based approaches for more aggressive depth reduction.
"""

import os
import time
import argparse
import sys
from qiskit import QuantumCircuit

# Import from the toffoli_optimizer package
from toffoli_optimizer.core import (
    ToffoliCompiler, 
    ToffoliType,
    EnhancedToffoliDepthOptimizer, 
    OptimizationStrategy,
    ZXOptimizer
)
from toffoli_optimizer.utils import (
    save_circuit_to_qasm,
    save_circuit_image
)

# Try to import RL components
try:
    from toffoli_optimizer.core import RLOptimizationWrapper, RL_OPTIMIZER_AVAILABLE
except ImportError:
    RL_OPTIMIZER_AVAILABLE = False

def run_enhanced_optimization():
    """Run an example using the enhanced optimization capabilities"""
    print("=== Enhanced Toffoli Circuit Optimization Example ===")
    
    # Define a Toffoli network (2-bit adder)
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
    
    # Create a compiler and different optimizer types
    compiler = ToffoliCompiler()
    
    # Create directory for results
    output_dir = "enhanced_example_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run with different optimization strategies
    strategies = [
        OptimizationStrategy.STANDARD,
        OptimizationStrategy.HYBRID,
        OptimizationStrategy.ULTRA_DEPTH_REDUCTION,
        OptimizationStrategy.DEPTH_FIDELITY_BALANCE
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\nRunning optimization with {strategy.name} strategy...")
        
        # Create optimizer with this strategy
        optimizer = EnhancedToffoliDepthOptimizer(
            target_fidelity=0.95,
            min_fidelity=0.85,  # Allow fidelity to drop to this level for more aggressive optimization
            max_passes=2,
            output_dir=output_dir,
            debug_mode=True,
            strategy=strategy,
            use_zx_optimization=True,
            use_rl_optimization=RL_OPTIMIZER_AVAILABLE
        )
        
        start_time = time.time()
        strategy_results = optimizer.optimize_toffoli_network(
            toffoli_gates,
            output_qubits,
            input_qubits,
            num_qubits,
            topology='linear'
        )
        
        # Extract key results
        logical_depth = strategy_results["logical"]["depth"]
        optimized_depth = strategy_results["optimized"]["depth"]
        physical_depth = strategy_results["mapped"]["depth"]
        logical_fidelity = strategy_results["optimized"]["logical_fidelity"]
        physical_fidelity = strategy_results["mapped"]["physical_fidelity"]
        
        # Calculate depth reductions
        depth_reduction = ((logical_depth - optimized_depth) / logical_depth * 100) if logical_depth > 0 else 0
        
        # Store results for comparison
        results[strategy.name] = {
            "logical_depth": logical_depth,
            "optimized_depth": optimized_depth,
            "physical_depth": physical_depth,
            "depth_reduction": depth_reduction,
            "logical_fidelity": logical_fidelity,
            "physical_fidelity": physical_fidelity,
            "runtime": time.time() - start_time
        }
        
        # Save the optimized circuit
        strategy_dir = os.path.join(output_dir, strategy.name.lower())
        os.makedirs(strategy_dir, exist_ok=True)
        
        save_circuit_to_qasm(
            strategy_results["optimized"]["circuit"],
            os.path.join(strategy_dir, "optimized_circuit")
        )
        
        # Save circuit images
        try:
            save_circuit_image(
                strategy_results["optimized"]["circuit"],
                "optimized_circuit",
                output_dir=strategy_dir,
                use_text_mode=True
            )
            save_circuit_image(
                strategy_results["mapped"]["circuit"],
                "mapped_circuit",
                output_dir=strategy_dir,
                use_text_mode=True
            )
        except Exception as e:
            print(f"Warning: Could not save circuit images: {e}")
    
    # Print comparison of all strategies
    print("\n=== Optimization Strategy Comparison ===")
    print(f"{'Strategy':<25}