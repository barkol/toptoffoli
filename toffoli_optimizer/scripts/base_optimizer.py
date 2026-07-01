#!/usr/bin/env python3
"""
Base Optimizer Module

This module provides the BaseOptimizer class which serves as the foundation for all optimizer classes
in the Toffoli Optimizer project. It contains common methods and attributes used by the Optimize class.
"""

import os
import time
import json
import traceback
import gc
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from toffoli_optimizer.core.compiler import ToffoliCompiler, ToffoliType
from toffoli_optimizer.core.optimizer import ToffoliDepthOptimizer, OptimizationStrategy
from toffoli_optimizer.utils.io_utils import define_loaded_toffoli_network, save_circuit_to_qasm
from toffoli_optimizer.utils.circuit_utils import get_default_coupling_map, estimate_fidelity
from toffoli_optimizer.utils.visualization import save_circuit_image, plot_optimization_results

class BaseOptimizer:
    """
    Base class for all optimizer classes in the Toffoli Optimizer project.
    
    This class provides common methods and attributes for optimizing Toffoli networks,
    which are inherited and possibly overridden by more specific optimizer classes.
    """
    
    def __init__(self):
        """Initialize the base optimizer with default values."""
        self.results = {}
        self.start_time = None
        self.debug_mode = False
    
    def get_optimization_strategy(self, strategy_name):
        """Convert strategy name string to OptimizationStrategy enum value"""
        strategy_map = {
            "STANDARD": OptimizationStrategy.STANDARD,
            "DEPTH_REDUCTION": OptimizationStrategy.DEPTH_REDUCTION,
            "GATE_REDUCTION": OptimizationStrategy.GATE_REDUCTION,
            "FIDELITY": OptimizationStrategy.FIDELITY,
            "HYBRID": OptimizationStrategy.HYBRID,
            "ULTRA_DEPTH_REDUCTION": OptimizationStrategy.ULTRA_DEPTH_REDUCTION,
            "DEPTH_FIDELITY_BALANCE": OptimizationStrategy.DEPTH_FIDELITY_BALANCE,
        }
        return strategy_map.get(strategy_name, OptimizationStrategy.STANDARD)
    
    def apply_additional_optimizations(self, circuit, coupling_map):
        """
        Apply additional optimizations to the given circuit if the corresponding optimizers are available.
        
        Args:
            circuit: The quantum circuit to optimize
            coupling_map: The coupling map to use for optimization
            
        Returns:
            The optimized circuit
        """
        optimized_circuit = circuit.copy()

        from toffoli_optimizer.core.pattern_library_module import ToffoliPatternLibrary

        try:
            print("Applying pattern-based optimization...")
            pattern_lib = ToffoliPatternLibrary()
            pattern_optimized = pattern_lib.optimize_circuit(
                optimized_circuit.copy(), require_verified=True
            )
            if pattern_optimized.depth() < optimized_circuit.depth():
                optimized_circuit = pattern_optimized
                print(f"Pattern optimization improved depth to {pattern_optimized.depth()}")
            else:
                print("Pattern optimization did not improve circuit depth")
            gc.collect()
        except Exception as e:
            print(f"Pattern optimization failed: {e}")

        return optimized_circuit
    
    
    def create_physical_mappings(self, optimized_circuit, coupling_map):
        """
        Create multiple physical mappings with different strategies and choose the best one.
        
        Args:
            optimized_circuit: The optimized logical circuit
            coupling_map: The coupling map to use for mapping
            
        Returns:
            A dictionary with information about the best physical mapping
        """
        best_physical_circuit = None
        best_physical_depth = float('inf')
        
        # 1. Create naive physical mapping
        try:
            from qiskit import transpile
            naive_physical_circuit = transpile(
                optimized_circuit.copy(),
                coupling_map=coupling_map,
                basis_gates=["id", "rz", "sx", "x", "cx"],
                optimization_level=1
            )
            
            naive_physical_depth = naive_physical_circuit.depth()
            print(f"Naive physical mapping depth: {naive_physical_depth}")
            
            # Track all physical mappings including the naive one
            physical_mappings = [
                {"name": "Naive", "circuit": naive_physical_circuit, "depth": naive_physical_depth}
            ]
            
            # 2. Create optimized physical mapping with different layouts and optimization levels
            mapping_methods = [
                {"name": "Standard", "layout": "sabre", "routing": "sabre", "optimization": 1},
                {"name": "Aggressive", "layout": "sabre", "routing": "stochastic", "optimization": 3},
                {"name": "Dense", "layout": "dense", "routing": "sabre", "optimization": 2},
                {"name": "Noise Adaptive", "layout": "noise_adaptive", "routing": "stochastic", "optimization": 2}
            ]
            
            for method in mapping_methods:
                try:
                    print(f"Trying {method['name']} physical mapping...")
                    
                    # Qiskit 2.0 compatibility for transpile
                    # The layout_method and routing_method params have been updated in Qiskit 2.0
                    # Use try/except to handle different versions
                    try:
                        # First try with Qiskit 2.0 style parameters
                        mapped_circuit = transpile(
                            optimized_circuit.copy(),
                            coupling_map=coupling_map,
                            basis_gates=["id", "rz", "sx", "x", "cx"],
                            optimization_level=method["optimization"]
                        )
                    except TypeError:
                        # Fall back to older Qiskit style parameters
                        mapped_circuit = transpile(
                            optimized_circuit.copy(),
                            coupling_map=coupling_map,
                            basis_gates=["id", "rz", "sx", "x", "cx"],
                            layout_method=method["layout"],
                            routing_method=method["routing"],
                            optimization_level=method["optimization"]
                        )
                    
                    # Further optimize the mapped circuit for depth
                    optimized_physical = transpile(
                        mapped_circuit,
                        coupling_map=coupling_map,
                        optimization_level=3
                    )
                    
                    # Get metrics after physical optimization
                    physical_depth = optimized_physical.depth()
                    
                    print(f"  {method['name']} mapping depth: {physical_depth}")
                    
                    # Track this mapping
                    physical_mappings.append({
                        "name": method["name"], 
                        "circuit": optimized_physical, 
                        "depth": physical_depth
                    })
                    
                    # Update best if this is better
                    if physical_depth < best_physical_depth:
                        best_physical_circuit = optimized_physical
                        best_physical_depth = physical_depth
                        
                    # Force garbage collection after each mapping method
                    del mapped_circuit
                    gc.collect()
                        
                except Exception as e:
                    print(f"  Error with {method['name']} mapping: {e}")
                    if self.debug_mode:
                        traceback.print_exc()
        except Exception as e:
            print(f"Error during physical mapping optimization: {e}")
            if self.debug_mode:
                traceback.print_exc()
        
        # Choose the best physical mapping (including naive if it's better)
        if physical_mappings:
            # Sort by depth
            physical_mappings.sort(key=lambda x: x["depth"])
            best_mapping = physical_mappings[0]
            
            print(f"\nComparing physical mapping strategies:")
            for mapping in physical_mappings:
                print(f"  {mapping['name']}: depth {mapping['depth']}")
                
            print(f"\nSelected best physical mapping: {best_mapping['name']} with depth {best_mapping['depth']}")
            
            # Get more detailed metrics for the chosen mapping
            gate_count = len(best_mapping["circuit"].data)
            
            # Count CX gates with proper error handling for Qiskit 2.0
            try:
                cx_count = 0
                for inst in best_mapping["circuit"].data:
                    if hasattr(inst, 'operation') and hasattr(inst.operation, 'name'):
                        if inst.operation.name == 'cx':
                            cx_count += 1
            except Exception as e:
                if self.debug_mode:
                    print(f"Error counting CX gates: {e}")
                cx_count = 0
            
            # Estimate fidelity
            try:
                fidelity = estimate_fidelity(best_mapping["circuit"])
            except Exception as e:
                if self.debug_mode:
                    print(f"Error estimating fidelity: {e}")
                fidelity = 0.9  # Default value
            
            # Return the result with the best physical circuit and additional metrics
            return {
                "circuit": best_mapping["circuit"],
                "depth": best_mapping["depth"],
                "name": best_mapping["name"],
                "gate_count": gate_count,
                "cx_count": cx_count,
                "fidelity": fidelity
            }
        
        # Return None if no physical mappings were created
        return None
    
    def make_serializable(self, obj):
        """
        Make an object JSON serializable by removing non-serializable components.
        
        Args:
            obj: Object to make serializable
            
        Returns:
            object: JSON serializable version of the object
        """
        if isinstance(obj, dict):
            serializable_dict = {}
            for key, value in obj.items():
                # Skip circuit objects and other non-serializable things
                if key.endswith("circuit") or key in ["coupling_map"]:
                    continue
                serializable_dict[key] = self.make_serializable(value)
            return serializable_dict
        elif isinstance(obj, list):
            return [self.make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Convert other objects to string representation
            return str(obj)
    
    def optimize_with_transpiler(self, circuit, coupling_map, optimization_level=3, target_fidelity=0.95):
        """
        Optimize a circuit using Qiskit's transpiler with the specified optimization level.
        
        Args:
            circuit: The circuit to optimize
            coupling_map: The coupling map to use
            optimization_level: The optimization level (1-3)
            target_fidelity: Target fidelity (not directly used by transpiler, but tracked for comparison)
            
        Returns:
            dict: A dictionary with optimization results including the optimized circuit and metrics
        """
        print(f"Running Qiskit transpiler with optimization level {optimization_level}...")
        
        try:
            from qiskit import transpile
            
            start_time = time.time()
            
            # Get original physical depth for accurate comparison
            original_physical_circuit = transpile(
                circuit.copy(),
                coupling_map=coupling_map,
                basis_gates=['id', 'rz', 'sx', 'x', 'cx'],
                optimization_level=1
            )
            original_physical_depth = original_physical_circuit.depth()
            
            # Run the transpiler
            transpiled = transpile(
                circuit.copy(),
                coupling_map=coupling_map,
                basis_gates=['id', 'rz', 'sx', 'x', 'cx'],
                optimization_level=optimization_level
            )
            
            runtime = time.time() - start_time
            
            # Get metrics
            transpiled_depth = transpiled.depth()
            
            # Count gates and CX gates
            gate_count = len(transpiled.data)
            cx_count = sum(1 for g in transpiled.data if hasattr(g, 'operation') and 
                          hasattr(g.operation, 'name') and g.operation.name == 'cx')
            
            # Calculate reductions
            if original_physical_depth > 0:
                naive_to_physical_reduction = ((original_physical_depth - transpiled_depth) / original_physical_depth * 100)
            else:
                naive_to_physical_reduction = 0
            
            print(f"  Qiskit L{optimization_level}: depth={transpiled_depth}, gates={gate_count}, CX={cx_count}")
            print(f"  Depth reduction: {naive_to_physical_reduction:.2f}%")
            print(f"  Runtime: {runtime:.2f} seconds")
            
            # Store results
            result = {
                "method": f"Qiskit_L{optimization_level}",
                "description": f"Qiskit Transpiler (Level {optimization_level})",
                "target_fidelity": target_fidelity,
                "original_physical_depth": original_physical_depth,
                "physical_depth": transpiled_depth,
                "gate_count": gate_count,
                "cx_count": cx_count,
                "naive_to_physical_reduction": naive_to_physical_reduction,
                "runtime": runtime,
                "circuit": transpiled,
                "optimization_level": optimization_level
            }
            
            return result
        
        except Exception as e:
            print(f"Error running transpiler optimization: {e}")
            if self.debug_mode:
                traceback.print_exc()
            return None
    
    def print_optimization_summary(self, toffoli_gates=None, num_qubits=None, topology=None, strategy=None,
                                  target_fidelity=None, min_fidelity=None, max_passes=None):
        """
        Print a summary of the optimization results.
        
        Args:
            toffoli_gates: List of Toffoli gates
            num_qubits: Number of qubits in the circuit
            topology: Target topology (e.g., 'linear', 'grid', 'falcon')
            strategy: Optimization strategy used
            target_fidelity: Target fidelity for optimization
            min_fidelity: Minimum acceptable fidelity
            max_passes: Maximum number of optimization passes
        """
        logical_depth = self.results.get("logical", {}).get("depth", 0)
        naive_depth = self.results.get("naive_physical", {}).get("depth", 0)
        optimized_depth = self.results.get("optimized", {}).get("depth", 0)
        physical_depth = self.results.get("mapped", {}).get("depth", 0)
        
        print("\nOptimization Summary:")
        print("-" * 80)
        
        if toffoli_gates is not None:
            print(f"Number of Toffoli gates: {len(toffoli_gates)}")
        
        if num_qubits is not None:
            print(f"Number of qubits: {num_qubits}")
        
        if topology is not None:
            print(f"Topology: {topology}")
        
        if strategy is not None:
            print(f"Optimization strategy: {strategy}")
        
        if target_fidelity is not None:
            print(f"Target fidelity: {target_fidelity}")
        
        if min_fidelity is not None:
            print(f"Minimum fidelity: {min_fidelity}")
        
        if max_passes is not None:
            print(f"Maximum optimization passes: {max_passes}")
        
        print("\nDepth Metrics:")
        print(f"Logical circuit depth: {logical_depth}")
        print(f"Naive physical mapping depth: {naive_depth}")
        print(f"Optimized logical depth: {optimized_depth}")
        print(f"Final physical depth: {physical_depth}")
        
        # Calculate reductions
        logical_to_optimized_reduction = ((logical_depth - optimized_depth) / logical_depth * 100) if logical_depth > 0 else 0
        naive_to_physical_reduction = ((naive_depth - physical_depth) / naive_depth * 100) if naive_depth > 0 else 0
        
        print("\nOptimization Results:")
        print(f"Logical-to-optimized depth reduction: {logical_to_optimized_reduction:.2f}%")
        print(f"Naive-physical-to-final-physical depth reduction: {naive_to_physical_reduction:.2f}%")
        
        if "mapped" in self.results and "fidelity" in self.results["mapped"]:
            print(f"Final physical fidelity: {self.results['mapped']['fidelity']:.4f}")
        
        optimization_time = self.results.get("optimization_time", time.time() - self.start_time if self.start_time else 0)
        print(f"Optimization time: {optimization_time:.4f} seconds")
        print("-" * 80)
    
    def save_optimization_report(self, output_file):
        """
        Save the optimization results to a JSON file.
        
        Args:
            output_file: Path to save the report to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a serializable report
            serializable_report = self.make_serializable(self.results)
            
            # Add key metrics
            logical_depth = self.results.get("logical", {}).get("depth", 0)
            naive_depth = self.results.get("naive_physical", {}).get("depth", 0)
            optimized_depth = self.results.get("optimized", {}).get("depth", 0)
            physical_depth = self.results.get("mapped", {}).get("depth", 0)
            
            # Calculate reductions
            logical_to_optimized_reduction = ((logical_depth - optimized_depth) / logical_depth * 100) if logical_depth > 0 else 0
            naive_to_physical_reduction = ((naive_depth - physical_depth) / naive_depth * 100) if naive_depth > 0 else 0
            
            serializable_report["logical_depth"] = logical_depth
            serializable_report["naive_depth"] = naive_depth
            serializable_report["optimized_depth"] = optimized_depth
            serializable_report["physical_depth"] = physical_depth
            serializable_report["logical_to_optimized_reduction"] = logical_to_optimized_reduction
            serializable_report["naive_to_physical_reduction"] = naive_to_physical_reduction
            
            if "mapped" in self.results and "name" in self.results["mapped"]:
                serializable_report["physical_mapping_method"] = self.results["mapped"]["name"]
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(serializable_report, f, indent=2)
            
            print(f"Optimization report saved to {output_file}")
            return True
        
        except Exception as e:
            print(f"Error saving optimization report: {e}")
            if self.debug_mode:
                traceback.print_exc()
            return False
    
    def generate_visualization(self, optimized_circuit, output_base):
        """
        Generate visualization of the optimization results.
        
        Args:
            optimized_circuit: The optimized circuit to visualize
            output_base: Base path for saving visualization files
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from toffoli_optimizer.utils.visualization import visualize_optimization
            visualize_optimization(optimized_circuit, self.results, filename=output_base)
            print(f"Visualization saved to {output_base}_*.png")
            return True
        except Exception as e:
            print(f"Error generating visualization: {e}")
            if self.debug_mode:
                traceback.print_exc()
            return False
    
    def display_summary_table(self):
        """
        Display a tabular summary of optimization results.
        This method should be implemented by derived classes.
        """
        raise NotImplementedError("This method should be implemented by derived classes")
