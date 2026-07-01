#!/usr/bin/env python3
"""
Optimize Module

This module provides the Optimize class which implements the 'optimize' command
for the Toffoli Optimizer project. It inherits from the BaseOptimizer class and
adds functionality specific to optimizing a single Toffoli network.
"""

import os
import time
import gc
import traceback
import sys

from toffoli_optimizer.core.optimizer import ToffoliDepthOptimizer
from toffoli_optimizer.utils.io_utils import define_loaded_toffoli_network, save_circuit_to_qasm
from toffoli_optimizer.utils.circuit_utils import get_default_coupling_map
from toffoli_optimizer.scripts.base_optimizer import BaseOptimizer

class Optimize(BaseOptimizer):
    """
    Optimize a Toffoli network with various optimization strategies.
    
    This class implements the 'optimize' command for the Toffoli Optimizer project.
    It provides functionality for optimizing a single Toffoli network with various
    optimization strategies and parameters.
    """
    
    def __init__(self, args):
        """
        Initialize the Optimize class with command line arguments.
        
        Args:
            args: Command line arguments for the 'optimize' command
        """
        super().__init__()
        self.args = args
        self.debug_mode = args.debug
    
    def run(self):
        """
        Run the optimization process on the specified input file.
        
        Returns:
            int: Exit code (0 for success, 1 for failure)
        """
        print(f"Running Toffoli Depth Optimizer on {self.args.input}")
        self.start_time = time.time()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Get the optimization strategy
        strategy = self.get_optimization_strategy(self.args.strategy)
        
        # Initialize the optimizer with parameters
        optimizer = ToffoliDepthOptimizer(
            target_fidelity=self.args.fidelity,
            min_fidelity=self.args.min_fidelity,
            max_passes=self.args.max_passes,
            strategy=strategy,
            depth_weight=self.args.depth_weight,
            fidelity_weight=self.args.fidelity_weight,
            debug_mode=self.debug_mode
        )
                
        # Load the Toffoli network
        result = define_loaded_toffoli_network(self.args.input)
        if result is None:
            print(f"Error: Failed to load Toffoli network from {self.args.input}")
            return 1
        
        toffoli_gates, output_qubits, input_qubits = result
        
        # Use default num_qubits if not defined in the loaded network
        num_qubits = self.args.num_qubits or max(8,
            max([max(controls + [target]) for controls, target in toffoli_gates]) + 1)
        
        # Override the number of qubits if specified
        if self.args.num_qubits is not None:
            if self.args.num_qubits < num_qubits:
                print(f"Warning: Specified number of qubits ({self.args.num_qubits}) is less than the number of qubits in the input file ({num_qubits})")
                print("Using the larger value to ensure all qubits are included")
                num_qubits = max(self.args.num_qubits, num_qubits)
            else:
                num_qubits = self.args.num_qubits
        
        # Get the coupling map for the specified topology
        coupling_map = get_default_coupling_map(self.args.topology, num_qubits)
        
        # Check if using transpiler optimization strategy
        if self.args.strategy.startswith('TRANSPILER'):
            try:
                # Create the circuit first
                from qiskit import QuantumCircuit
                from toffoli_optimizer.core.compiler import ToffoliCompiler
                
                # Create compiler if not already initialized
                if not hasattr(self, 'compiler') or self.compiler is None:
                    self.compiler = ToffoliCompiler(debug_mode=self.debug_mode)
                
                # Create logical circuit from Toffoli gates
                print("Creating logical circuit for transpiler optimization...")
                logical_circuit, _ = self.compiler.create_toffoli_network(
                    toffoli_gates,
                    num_qubits,
                    use_ancilla=True,
                    target_fidelity=1.0  # Use maximum fidelity for logical circuit
                )
                
                if logical_circuit is None:
                    print("Error: Failed to create logical circuit for transpiler optimization")
                    return 1
                
                # Determine optimization level from strategy
                if self.args.strategy == 'TRANSPILER_L1':
                    opt_level = 1
                elif self.args.strategy == 'TRANSPILER_L2':
                    opt_level = 2
                elif self.args.strategy == 'TRANSPILER_L3':
                    opt_level = 3
                else:  # Default to level 3 for 'TRANSPILER'
                    opt_level = 3
                
                # Use transpiler optimization
                transpiler_result = self.optimize_with_transpiler(
                    logical_circuit,
                    coupling_map,
                    optimization_level=opt_level,
                    target_fidelity=self.args.fidelity
                )
                
                if transpiler_result is None:
                    print("Error: Transpiler optimization failed")
                    return 1
                
                # Create results in the expected format
                self.results = {
                    "logical": {
                        "circuit": logical_circuit,
                        "depth": logical_circuit.depth()
                    },
                    "naive_physical": {
                        "circuit": transpiler_result.get("original_physical_circuit", None),
                        "depth": transpiler_result.get("original_physical_depth", 0)
                    },
                    "optimized": {
                        "circuit": transpiler_result.get("circuit", None),
                        "depth": transpiler_result.get("physical_depth", 0)
                    },
                    "mapped": {
                        "circuit": transpiler_result.get("circuit", None),
                        "depth": transpiler_result.get("physical_depth", 0),
                        "gate_count": transpiler_result.get("gate_count", 0),
                        "cx_count": transpiler_result.get("cx_count", 0),
                        "fidelity": self.args.fidelity,  # Assume target fidelity is met
                        "name": f"Transpiler L{opt_level}"
                    },
                    "optimization_time": transpiler_result.get("runtime", 0),
                    "optimization_strategy": self.args.strategy
                }
                
                # Get the optimized circuit
                optimized_circuit = transpiler_result.get("circuit", None)
                
                # Save the optimized circuit
                if optimized_circuit and save_circuit_to_qasm(optimized_circuit, self.args.output):
                    print(f"Optimized circuit saved to {self.args.output}")
                else:
                    print(f"Error: Failed to save optimized circuit to {self.args.output}")
                    return 1
                
                # Print optimization summary
                self.print_optimization_summary(
                    toffoli_gates=toffoli_gates,
                    num_qubits=num_qubits,
                    topology=self.args.topology,
                    strategy=self.args.strategy,
                    target_fidelity=self.args.fidelity,
                    min_fidelity=self.args.min_fidelity,
                    max_passes=self.args.max_passes
                )
                
                # Generate visualization if requested
                if self.args.visualize:
                    output_base = os.path.splitext(self.args.output)[0]
                    self.generate_visualization(optimized_circuit, output_base)
                
                # Save optimization report
                report_file = os.path.splitext(self.args.output)[0] + "_report.json"
                self.save_optimization_report(report_file)
                
                # Display the summary table
                self.display_summary_table()
                
                return 0
                
            except Exception as e:
                print(f"Error during transpiler optimization: {e}")
                if self.debug_mode:
                    traceback.print_exc()
                return 1
        
        
        
        # Run the regular ToffoliDepthOptimizer if not using transpiler strategy
        # Initialize the optimizer with parameters
        optimizer = ToffoliDepthOptimizer(
            target_fidelity=self.args.fidelity,
            min_fidelity=self.args.min_fidelity,
            max_passes=self.args.max_passes,
            strategy=strategy,
            depth_weight=self.args.depth_weight,
            fidelity_weight=self.args.fidelity_weight,
            debug_mode=self.debug_mode
        )
        
        # Run the optimization with proper error handling
        try:
            # Force garbage collection before starting optimization
            gc.collect()
            
            # Run the optimization
            self.results = optimizer.optimize_toffoli_network(
                toffoli_gates,
                output_qubits,
                input_qubits,
                num_qubits,
                topology=self.args.topology,
                coupling_map=coupling_map
            )
            
            # Force garbage collection after initial optimization
            gc.collect()
            
            # Apply additional optimizations if enabled and available
            if "optimized" in self.results:
                optimized_circuit = self.results["optimized"]["circuit"]
                
                # Apply additional optimizations if requested
                if self.args.use_patterns:
                    optimized_circuit = self.apply_additional_optimizations(
                        optimized_circuit,
                        coupling_map
                    )
                    
                    # Update the optimized results
                    self.results["optimized"]["circuit"] = optimized_circuit
                    self.results["optimized"]["depth"] = optimized_circuit.depth()
                
                # Update final results
                if "final" in self.results:
                    self.results["final"]["circuit"] = self.results["optimized"]["circuit"]
                    self.results["final"]["depth"] = self.results["optimized"]["depth"]
                
            
                # Create physical mappings and choose the best one
                mapped_result = self.create_physical_mappings(optimized_circuit, coupling_map)
                
                if mapped_result:
                    self.results["mapped"] = mapped_result
        
        except Exception as e:
            print(f"Error during optimization: {e}")
            if self.debug_mode:
                traceback.print_exc()
            return 1
        
        # Get the optimized circuit
        if "mapped" in self.results and "circuit" in self.results["mapped"]:
            optimized_circuit = self.results["mapped"]["circuit"]
        else:
            print("Error: No optimized circuit produced")
            return 1
        
        

        
        # Save the optimized circuit
        if save_circuit_to_qasm(optimized_circuit, self.args.output):
            print(f"Optimized circuit saved to {self.args.output}")
        else:
            print(f"Error: Failed to save optimized circuit to {self.args.output}")
            return 1
        
        # Print optimization summary
        self.print_optimization_summary(
            toffoli_gates=toffoli_gates,
            num_qubits=num_qubits,
            topology=self.args.topology,
            strategy=self.args.strategy,
            target_fidelity=self.args.fidelity,
            min_fidelity=self.args.min_fidelity,
            max_passes=self.args.max_passes
        )
        
        # Record execution time if not already set
        if 'optimization_time' not in self.results:
            self.results['optimization_time'] = time.time() - self.start_time
        
        # Generate visualization if requested
        if self.args.visualize:
            output_base = os.path.splitext(self.args.output)[0]
            self.generate_visualization(optimized_circuit, output_base)
        
        # Save optimization report
        report_file = os.path.splitext(self.args.output)[0] + "_report.json"
        self.save_optimization_report(report_file)
        
        # Display the summary table
        self.display_summary_table()
        
        return 0
    
    def display_summary_table(self):
        """
        Display a tabular summary of optimization results.
        
        For the Optimize class, this displays a simple table with depth metrics
        and reduction percentages.
        """
        # Extract metrics
        logical_depth = self.results.get("logical", {}).get("depth", 0)
        naive_depth = self.results.get("naive_physical", {}).get("depth", 0)
        optimized_depth = self.results.get("optimized", {}).get("depth", 0)
        physical_depth = self.results.get("mapped", {}).get("depth", 0)
        
        # Calculate reductions
        logical_to_optimized_reduction = ((logical_depth - optimized_depth) / logical_depth * 100) if logical_depth > 0 else 0
        naive_to_physical_reduction = ((naive_depth - physical_depth) / naive_depth * 100) if naive_depth > 0 else 0
        
        # Format the table
        print("\nOptimization Results Table:")
        print("-" * 60)
        print(f"{'Circuit Type':<20} | {'Depth':<10} | {'Reduction (%)':<15}")
        print("-" * 60)
        print(f"{'Logical':<20} | {logical_depth:<10} | {'N/A':<15}")
        print(f"{'Naive Physical':<20} | {naive_depth:<10} | {'N/A':<15}")
        print(f"{'Optimized Logical':<20} | {optimized_depth:<10} | {logical_to_optimized_reduction:>6.2f}%")
        print(f"{'Final Physical':<20} | {physical_depth:<10} | {naive_to_physical_reduction:>6.2f}%")
        print("-" * 60)
        
        # Add additional metrics
        if "mapped" in self.results:
            gate_count = self.results["mapped"].get("gate_count", 0)
            cx_count = self.results["mapped"].get("cx_count", 0)
            fidelity = self.results["mapped"].get("fidelity", 0)
            
            print(f"Gate count: {gate_count}")
            print(f"CX gate count: {cx_count}")
            print(f"Estimated fidelity: {fidelity:.4f}")
        
        # Show timing information
        optimization_time = self.results.get("optimization_time", 0)
        print(f"Optimization time: {optimization_time:.2f} seconds")
        print("-" * 60)
