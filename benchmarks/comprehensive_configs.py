#!/usr/bin/env python3
"""
Toffoli Optimizer Configuration Test Script

This script tests various configurations of the toffoli-optimizer command on circuit.json,
capturing performance metrics and generating comparative visualizations.

It runs combinations of different parameter configurations and averages results over multiple runs.
"""

import subprocess
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
import time
import traceback
from pathlib import Path
import itertools
from collections import defaultdict
import re
import hashlib

# Define configuration parameters to test
TOPOLOGIES = ["linear", "grid", "falcon"]
STRATEGIES = ["STANDARD", "DEPTH_REDUCTION", "GATE_REDUCTION", "HYBRID", "ULTRA_DEPTH_REDUCTION",
              "DEPTH_FIDELITY_BALANCE", "TRANSPILER_L1", "TRANSPILER_L2", "TRANSPILER_L3"]
FIDELITIES = [0.85, 0.90, 0.95]
MIN_FIDELITIES = [0.70, 0.80, 0.90]
MAX_PASSES = [1, 2, 3]
DEPTH_FIDELITY_WEIGHTS = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]

def generate_test_configurations():
    """Generate more comprehensive test configurations with parameter combinations"""
    test_configs = []
    
    # 1. Core combinations: Test all combinations of topology and strategy
    for topology in TOPOLOGIES:
        for strategy in STRATEGIES:
            # Handle DEPTH_FIDELITY_BALANCE specially
            if strategy == "DEPTH_FIDELITY_BALANCE":
                # For this strategy, test all weight combinations
                for depth_weight, fidelity_weight in DEPTH_FIDELITY_WEIGHTS:
                    config = {
                        "topology": topology,
                        "strategy": strategy,
                        "fidelity": 0.95,  # Default
                        "min_fidelity": 0.80,  # Default
                        "max_passes": 2,  # Default
                        "depth_weight": depth_weight,
                        "fidelity_weight": fidelity_weight
                    }
                    test_configs.append(config)
            else:
                # Standard parameters for all other strategies
                config = {
                    "topology": topology,
                    "strategy": strategy,
                    "fidelity": 0.95,  # Default
                    "min_fidelity": 0.80,  # Default
                    "max_passes": 2  # Default
                }
                test_configs.append(config)
    
    # 2. Parameter variations for promising strategies
    promising_strategies = ["STANDARD", "ULTRA_DEPTH_REDUCTION", "DEPTH_REDUCTION", "HYBRID"]
    
    for topology in TOPOLOGIES:
        for strategy in promising_strategies:
            # Vary fidelity
            for fidelity in [0.85, 0.90]:  # 0.95 already covered above
                config = {
                    "topology": topology,
                    "strategy": strategy,
                    "fidelity": fidelity,
                    "min_fidelity": 0.80,  # Default
                    "max_passes": 2  # Default
                }
                test_configs.append(config)
            
            # Vary min_fidelity
            for min_fidelity in [0.70, 0.90]:  # 0.80 already covered above
                config = {
                    "topology": topology,
                    "strategy": strategy,
                    "fidelity": 0.95,  # Default
                    "min_fidelity": min_fidelity,
                    "max_passes": 2  # Default
                }
                test_configs.append(config)
            
            # Vary max_passes
            for max_passes in [1, 3]:  # 2 already covered above
                config = {
                    "topology": topology,
                    "strategy": strategy,
                    "fidelity": 0.95,  # Default
                    "min_fidelity": 0.80,  # Default
                    "max_passes": max_passes
                }
                test_configs.append(config)
    
    # 3. Add some specific, interesting combinations for all topologies
    for topology in TOPOLOGIES:
        # Add combinations known to be effective for depth reduction
        test_configs.append({
            "topology": topology,
            "strategy": "ULTRA_DEPTH_REDUCTION",
            "fidelity": 0.90,
            "min_fidelity": 0.70,
            "max_passes": 3
        })
        
        # Add combinations for balancing fidelity and depth
        test_configs.append({
            "topology": topology,
            "strategy": "DEPTH_FIDELITY_BALANCE",
            "fidelity": 0.95,
            "min_fidelity": 0.85,
            "max_passes": 3,
            "depth_weight": 0.6,
            "fidelity_weight": 0.4
        })
        
        # Add promising hybrid configurations
        test_configs.append({
            "topology": topology,
            "strategy": "HYBRID",
            "fidelity": 0.92,
            "min_fidelity": 0.75,
            "max_passes": 3
        })
    
    # Remove any duplicates (based on all parameter values)
    unique_configs = []
    seen_configs = set()
    
    for config in test_configs:
        # Convert config to a hashable representation (JSON string)
        config_key = json.dumps(config, sort_keys=True)
        if config_key not in seen_configs:
            seen_configs.add(config_key)
            unique_configs.append(config)
    
    return unique_configs

def compute_config_hash(config):
    """Compute a short hash for a configuration to use in filenames"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def config_to_string(config):
    """Convert a configuration dictionary to a readable name string"""
    name_parts = []
    name_parts.append(f"{config['topology'].capitalize()}")
    name_parts.append(f"{config['strategy']}")
    
    # Add fidelity if not default
    if config.get('fidelity', 0.95) != 0.95:
        name_parts.append(f"Fid{config['fidelity']}")
    
    # Add min_fidelity if not default
    if config.get('min_fidelity', 0.80) != 0.80:
        name_parts.append(f"MinFid{config['min_fidelity']}")
    
    # Add max_passes if not default
    if config.get('max_passes', 2) != 2:
        name_parts.append(f"P{config['max_passes']}")
    
    # Add depth/fidelity weights if present
    if 'depth_weight' in config and 'fidelity_weight' in config:
        name_parts.append(f"D{config['depth_weight']}-F{config['fidelity_weight']}")
    
    return " ".join(name_parts)

def config_to_params(config):
    """Convert a configuration dictionary to command line parameters"""
    params = []
    params.append(f"--topology {config['topology']}")
    params.append(f"--strategy {config['strategy']}")
    params.append(f"--fidelity {config['fidelity']}")
    params.append(f"--min_fidelity {config['min_fidelity']}")
    params.append(f"--max_passes {config['max_passes']}")
    
    # Add depth_weight and fidelity_weight if present (for DEPTH_FIDELITY_BALANCE)
    if 'depth_weight' in config and 'fidelity_weight' in config:
        params.append(f"--depth_weight {config['depth_weight']}")
        params.append(f"--fidelity_weight {config['fidelity_weight']}")
    
    return " ".join(params)

def run_baseline(topology, input_file="circuit", timeout=300):
    """Run just the initial mapping to get the naive physical depth for a topology"""
    # Create a temporary directory for this run
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "output.qasm")
    
    try:
        # Build a command that will just do the naive mapping without optimization
        # Use TRANSPILER_L1 with optimization_level 1 to get the "naive" mapping
        cmd = f"python -m toffoli_optimizer.scripts.main optimize --input {input_file} --output {output_path} --topology {topology} --strategy TRANSPILER_L1 --max_passes 1"
        
        print(f"Running baseline command for {topology} topology: {cmd}")
        start_time = time.time()
        
        # Run the command with timeout
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Baseline command completed in {execution_time:.2f} seconds with return code {result.returncode}")
        
        # Try to extract the naive physical depth from the output
        naive_depth = None
        
        # Look for "Naive physical mapping depth: X" in the output
        for line in result.stdout.split('\n'):
            match = re.search(r"Naive physical (?:mapping )?depth: (\d+)", line)
            if match:
                naive_depth = int(match.group(1))
                print(f"Found naive physical depth for {topology}: {naive_depth}")
                break
        
        if naive_depth is None:
            print(f"Could not find naive physical depth in output for {topology}")
        
        return {
            "topology": topology,
            "naive_depth": naive_depth,
            "execution_time": execution_time,
            "return_code": result.returncode,
            "success": naive_depth is not None
        }
    
    except subprocess.TimeoutExpired:
        print(f"Baseline command for {topology} timed out after {timeout} seconds")
        return {
            "topology": topology,
            "naive_depth": None,
            "execution_time": timeout,
            "return_code": None,
            "success": False
        }
    
    except Exception as e:
        print(f"Error running baseline command for {topology}: {e}")
        traceback.print_exc()
        return {
            "topology": topology,
            "naive_depth": None,
            "execution_time": None,
            "return_code": None,
            "success": False
        }
    
    finally:
        # Clean up the temporary directory
        shutil.rmtree(tmp_dir)

def run_optimizer(config, input_file="circuit", timeout=300):
    """Run the optimizer with the given configuration and extract results"""
    # Create a temporary directory for this run
    tmp_dir = tempfile.mkdtemp()
    output_path = os.path.join(tmp_dir, "output.qasm")
    report_path = os.path.splitext(output_path)[0] + "_report.json"
    
    try:
        # Build the command using config_to_params
        params = config_to_params(config)
        cmd = f"python -m toffoli_optimizer.scripts.main optimize --input {input_file} --output {output_path} {params}"
        
        print(f"Running command: {cmd}")
        start_time = time.time()
        
        # Run the command with timeout
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Command completed in {execution_time:.2f} seconds with return code {result.returncode}")
        
        # Check for errors
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            physical_depth = None
            success = False
        else:
            # Try to extract physical depth from various sources
            physical_depth = None
            
            # First check the report file
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    try:
                        report = json.load(f)
                        
                        # Try different possible locations of the depth in the report
                        if "physical_depth" in report:
                            physical_depth = report["physical_depth"]
                        elif "mapped" in report and "depth" in report["mapped"]:
                            physical_depth = report["mapped"]["depth"]
                        elif "mapped" in report and "physical_depth" in report["mapped"]:
                            physical_depth = report["mapped"]["physical_depth"]
                        
                        if physical_depth is not None:
                            print(f"Found physical depth in report: {physical_depth}")
                            success = True
                    except json.JSONDecodeError:
                        print(f"Failed to parse report file: {report_path}")
            
            # If depth not found in report, try to extract from stdout
            if physical_depth is None:
                for line in result.stdout.split('\n'):
                    match = re.search(r"Final physical depth: (\d+)", line)
                    if match:
                        physical_depth = int(match.group(1))
                        print(f"Found physical depth in output: {physical_depth}")
                        success = True
                        break
                        
                # If still not found, see if we can find "physical_depth" anywhere in the output
                if physical_depth is None:
                    for line in result.stdout.split('\n'):
                        match = re.search(r"physical[_ ]depth[: =]+(\d+)", line, re.IGNORECASE)
                        if match:
                            physical_depth = int(match.group(1))
                            print(f"Found physical depth with regex: {physical_depth}")
                            success = True
                            break
            
            if physical_depth is None:
                print("Could not find physical depth in output or report")
                success = False
                
        return {
            "config": config,
            "config_name": config_to_string(config),
            "physical_depth": physical_depth,
            "success": success and physical_depth is not None,
            "execution_time": execution_time,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds")
        return {
            "config": config,
            "config_name": config_to_string(config),
            "physical_depth": None,
            "success": False,
            "execution_time": timeout,
            "return_code": None,
            "stdout": "Timeout",
            "stderr": f"Command timed out after {timeout} seconds"
        }
    
    except Exception as e:
        print(f"Error running command: {e}")
        traceback.print_exc()
        return {
            "config": config,
            "config_name": config_to_string(config),
            "physical_depth": None,
            "success": False,
            "execution_time": None,
            "return_code": None,
            "stdout": "",
            "stderr": str(e)
        }
    
    finally:
        # Clean up the temporary directory
        shutil.rmtree(tmp_dir)

def run_multiple_times(config, input_file="circuit", num_runs=5, timeout=300,
                       results_dir=None, config_hash=None):
    """Run the optimizer multiple times with the same configuration and average the results"""
    results = []
    successful_depths = []
    
    # Generate a unique identifier for this config if not provided
    if config_hash is None:
        config_hash = compute_config_hash(config)
    
    # Check for existing results
    if results_dir is not None:
        result_path = os.path.join(results_dir, f"multi_run_{config_hash}.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r') as f:
                    saved_result = json.load(f)
                    # Verify this is the same config
                    if saved_result.get("config_name") == config_to_string(config):
                        print(f"Loading existing results for {config_to_string(config)}")
                        return saved_result
            except:
                print(f"Failed to load existing results, running again")
    
    # Run the optimizer multiple times
    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs} for configuration: {config_to_string(config)}")
        result = run_optimizer(config, input_file=input_file, timeout=timeout)
        results.append(result)
        
        if result["success"] and result["physical_depth"] is not None:
            successful_depths.append(result["physical_depth"])
        
        # Save intermediate results if we have a results_dir
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)
            intermediate_path = os.path.join(results_dir, f"run_{config_hash}_{i+1}.json")
            with open(intermediate_path, 'w') as f:
                json.dump(result, f, indent=2)
    
    # Calculate the average depth if we have successful runs
    avg_depth = None
    std_dev = None
    if successful_depths:
        avg_depth = sum(successful_depths) / len(successful_depths)
        if len(successful_depths) > 1:
            std_dev = np.std(successful_depths)
        else:
            std_dev = 0.0
        print(f"Average physical depth over {len(successful_depths)} successful runs: {avg_depth:.2f}")
    else:
        print(f"No successful runs for this configuration")
    
    # Create a summary result
    summary_result = {
        "config": config,
        "config_name": config_to_string(config),
        "config_hash": config_hash,
        "individual_results": results,
        "num_runs": num_runs,
        "successful_runs": len(successful_depths),
        "average_depth": avg_depth,
        "min_depth": min(successful_depths) if successful_depths else None,
        "max_depth": max(successful_depths) if successful_depths else None,
        "std_dev": std_dev
    }
    
    # Save the full result if we have a results_dir
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        result_path = os.path.join(results_dir, f"multi_run_{config_hash}.json")
        with open(result_path, 'w') as f:
            json.dump(summary_result, f, indent=2)
    
    return summary_result

def plot_results_by_topology(results, baselines, output_dir="results"):
    """Create comparison plots for each topology"""
    # Group results by topology
    results_by_topology = defaultdict(list)
    for result in results:
        if result["average_depth"] is not None:
            topology = result["config"]["topology"]
            results_by_topology[topology].append(result)
    
    # Create a plot for each topology
    for topology, topology_results in results_by_topology.items():
        if not topology_results:
            continue
            
        # Sort by average depth
        sorted_results = sorted(topology_results, key=lambda x: x["average_depth"])
        
        # Get the baseline depth for this topology
        baseline_depth = None
        for baseline in baselines:
            if baseline["topology"] == topology and baseline["success"]:
                baseline_depth = baseline["naive_depth"]
                break
        
        # For large number of configurations, create multiple plots
        max_bars_per_plot = 20
        num_plots = (len(sorted_results) + max_bars_per_plot - 1) // max_bars_per_plot
        
        for plot_index in range(num_plots):
            start_idx = plot_index * max_bars_per_plot
            end_idx = min(start_idx + max_bars_per_plot, len(sorted_results))
            plot_results = sorted_results[start_idx:end_idx]
            
            # Create the figure and plot
            plt.figure(figsize=(14, 8))
            
            # Plot the data
            names = [r["config_name"] for r in plot_results]
            avg_depths = [r["average_depth"] for r in plot_results]
            
            # Create the bar plot
            bars = plt.bar(names, avg_depths)
            
            # Add error bars if we have std_dev
            if any(r["std_dev"] > 0 for r in plot_results):
                errors = [r["std_dev"] for r in plot_results]
                plt.errorbar(names, avg_depths, yerr=errors, fmt='none', ecolor='black', capsize=5)
            
            # Add a horizontal line for the baseline
            if baseline_depth is not None:
                plt.axhline(y=baseline_depth, color='r', linestyle='--', label=f'Baseline ({baseline_depth})')
                plt.legend()
            
            # Add labels and title
            plt.xlabel('Configuration')
            plt.ylabel('Average Final Physical Depth')
            title = f'Toffoli Optimizer Configurations - {topology.capitalize()} Topology'
            if num_plots > 1:
                title += f' (Group {plot_index+1}/{num_plots})'
            plt.title(title)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save the figure
            os.makedirs(output_dir, exist_ok=True)
            suffix = f"_group{plot_index+1}" if num_plots > 1 else ""
            output_path = os.path.join(output_dir, f"{topology}_comparison{suffix}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Plot for {topology} topology {suffix} saved to {output_path}")

def plot_overall_comparison(results, baselines, output_dir="results"):
    """Create an overall comparison plot with all topologies"""
    # Filter successful results
    successful_results = [r for r in results if r["average_depth"] is not None]
    
    if not successful_results:
        print("No successful results to plot")
        return
    
    # Sort by topology and then by average depth
    sorted_results = sorted(successful_results, key=lambda x: (x["config"]["topology"], x["average_depth"]))
    
    # For large number of configurations, create multiple plots
    max_bars_per_plot = 30
    num_plots = (len(sorted_results) + max_bars_per_plot - 1) // max_bars_per_plot
    
    for plot_index in range(num_plots):
        start_idx = plot_index * max_bars_per_plot
        end_idx = min(start_idx + max_bars_per_plot, len(sorted_results))
        plot_results = sorted_results[start_idx:end_idx]
        
        # Prepare the data for plotting
        names = [r["config_name"] for r in plot_results]
        avg_depths = [r["average_depth"] for r in plot_results]
        topologies = [r["config"]["topology"] for r in plot_results]
        
        # Create a mapping of topology to color
        topology_colors = {
            "linear": "blue",
            "grid": "green",
            "falcon": "purple"
        }
        
        # Create the figure
        plt.figure(figsize=(16, 10))
        
        # Create a list of colors based on topology
        colors = [topology_colors.get(t, "gray") for t in topologies]
        
        # Create the bar plot
        bars = plt.bar(names, avg_depths, color=colors)
        
        # Add error bars
        errors = [r["std_dev"] for r in plot_results]
        plt.errorbar(names, avg_depths, yerr=errors, fmt='none', ecolor='black', capsize=5)
        
        # Add baseline lines for each topology
        for baseline in baselines:
            if baseline["success"] and baseline["naive_depth"] is not None:
                topology = baseline["topology"]
                depth = baseline["naive_depth"]
                plt.axhline(y=depth, color=topology_colors.get(topology, "red"),
                           linestyle='--', label=f'{topology.capitalize()} Baseline ({depth})')
        
        # Add a legend for topologies
        plt.legend()
        
        # Add labels and title
        plt.xlabel('Configuration')
        plt.ylabel('Average Final Physical Depth')
        title = 'Comparison of Toffoli Optimizer Configurations Across Topologies'
        if num_plots > 1:
            title += f' (Group {plot_index+1}/{num_plots})'
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"_group{plot_index+1}" if num_plots > 1 else ""
        output_path = os.path.join(output_dir, f"overall_comparison{suffix}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Overall comparison plot {suffix} saved to {output_path}")

def plot_strategy_comparison(results, baselines, output_dir="results"):
    """Create a comparison plot for different optimization strategies"""
    # Group results by strategy
    results_by_strategy = defaultdict(list)
    for result in results:
        if result["average_depth"] is not None:
            strategy = result["config"]["strategy"]
            # Only include results with default values for other parameters
            if (result["config"].get("fidelity") == 0.95 and
                result["config"].get("min_fidelity") == 0.80 and
                result["config"].get("max_passes") == 2):
                # Handle DEPTH_FIDELITY_BALANCE separately
                if strategy == "DEPTH_FIDELITY_BALANCE":
                    if (result["config"].get("depth_weight") == 0.5 and
                        result["config"].get("fidelity_weight") == 0.5):
                        results_by_strategy[strategy].append(result)
                else:
                    results_by_strategy[strategy].append(result)
    
    # Create plots by topology
    for topology in TOPOLOGIES:
        # Filter results for this topology
        topology_results = []
        for strategy, strategy_results in results_by_strategy.items():
            for result in strategy_results:
                if result["config"]["topology"] == topology:
                    topology_results.append(result)
        
        if not topology_results:
            continue
            
        # Get all strategies that have at least one result
        strategies = sorted(set(r["config"]["strategy"] for r in topology_results))
        
        # Get the baseline depth for this topology
        baseline_depth = None
        for baseline in baselines:
            if baseline["topology"] == topology and baseline["success"]:
                baseline_depth = baseline["naive_depth"]
                break
        
        # Create the figure
        plt.figure(figsize=(14, 8))
        
        # Get data for plotting
        strategy_names = []
        avg_depths = []
        std_devs = []
        bar_colors = []
        
        for strategy in strategies:
            # Find the result for this strategy and topology
            strategy_result = None
            for result in topology_results:
                if result["config"]["strategy"] == strategy:
                    strategy_result = result
                    break
                    
            if strategy_result:
                strategy_names.append(strategy)
                avg_depths.append(strategy_result["average_depth"])
                std_devs.append(strategy_result["std_dev"])
                
                # Use special colors for transpiler strategies
                if "TRANSPILER" in strategy:
                    bar_colors.append('orange')
                else:
                    bar_colors.append('blue')
        
        # Create the bar plot
        bars = plt.bar(strategy_names, avg_depths, color=bar_colors)
        
        # Add error bars
        plt.errorbar(strategy_names, avg_depths, yerr=std_devs, fmt='none', ecolor='black', capsize=5)
        
        # Add baseline line for this topology
        if baseline_depth is not None:
            plt.axhline(y=baseline_depth, color='r', linestyle='--',
                      label=f'{topology.capitalize()} Baseline ({baseline_depth})')
            plt.legend()
        
        # Add labels and title
        plt.xlabel('Optimization Strategy')
        plt.ylabel('Average Final Physical Depth')
        plt.title(f'Comparison of Optimization Strategies ({topology.capitalize()} Topology, Default Parameters)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"strategy_comparison_{topology}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Strategy comparison plot for {topology} saved to {output_path}")

def plot_transpiler_comparison(results, baselines, output_dir="results"):
    """Create a plot comparing different transpiler optimization levels for each topology"""
    # Filter for just the transpiler strategies
    transpiler_strategies = ["TRANSPILER_L1", "TRANSPILER_L2", "TRANSPILER_L3"]
    
    # Create plots by topology
    for topology in TOPOLOGIES:
        # Find results with this topology and transpiler strategies
        topology_results = []
        
        for result in results:
            if (result["average_depth"] is not None and
                result["config"]["topology"] == topology and
                result["config"]["strategy"] in transpiler_strategies):
                topology_results.append(result)
        
        if not topology_results:
            print(f"No transpiler results found for {topology} topology")
            continue
            
        # Get the baseline depth for this topology
        baseline_depth = None
        for baseline in baselines:
            if baseline["topology"] == topology and baseline["success"]:
                baseline_depth = baseline["naive_depth"]
                break
        
        # Create the figure
        plt.figure(figsize=(10, 6))
        
        # Sort by optimization level
        sorted_results = sorted(topology_results,
                               key=lambda x: int(x["config"]["strategy"].split("_")[1][1:]))
        
        # Get data for plotting
        levels = [f"Level {r['config']['strategy'].split('_')[1][1:]}" for r in sorted_results]
        avg_depths = [r["average_depth"] for r in sorted_results]
        std_devs = [r["std_dev"] for r in sorted_results]
        
        # Create the bar plot
        bars = plt.bar(levels, avg_depths, color='orange')
        
        # Add error bars
        plt.errorbar(levels, avg_depths, yerr=std_devs, fmt='none', ecolor='black', capsize=5)
        
        # Add baseline line
        if baseline_depth is not None:
            plt.axhline(y=baseline_depth, color='r', linestyle='--',
                      label=f'Baseline Naive Depth ({baseline_depth})')
            plt.legend()
        
        # Add labels and title
        plt.xlabel('Transpiler Optimization Level')
        plt.ylabel('Average Final Physical Depth')
        plt.title(f'Qiskit Transpiler Performance ({topology.capitalize()} Topology)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Calculate percentage improvements
        if baseline_depth is not None:
            for i, depth in enumerate(avg_depths):
                improvement = (baseline_depth - depth) / baseline_depth * 100
                plt.text(i, depth - 2, f"{improvement:.1f}%", ha='center', va='top',
                       fontweight='bold')
        
        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"transpiler_comparison_{topology}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Transpiler comparison plot for {topology} saved to {output_path}")

def plot_parameter_impact(results, parameter, topologies, strategies, output_dir="results"):
    """Create a plot showing the impact of varying a specific parameter"""
    # Get the possible values for this parameter
    if parameter == "fidelity":
        param_values = FIDELITIES
        default_value = 0.95
    elif parameter == "min_fidelity":
        param_values = MIN_FIDELITIES
        default_value = 0.80
    elif parameter == "max_passes":
        param_values = MAX_PASSES
        default_value = 2
    else:
        print(f"Unknown parameter: {parameter}")
        return
    
    # Create a plot for each topology
    for topology in topologies:
        # For each strategy
        for strategy in strategies:
            # Find results with this topology and strategy, varying only the specified parameter
            param_results = []
            
            for param_value in param_values:
                # Find matching results
                matching_results = []
                for result in results:
                    if (result["average_depth"] is None or
                        result["config"]["topology"] != topology or
                        result["config"]["strategy"] != strategy):
                        continue
                        
                    # Check if only the specified parameter varies from defaults
                    if parameter == "fidelity":
                        if (result["config"]["fidelity"] == param_value and
                            result["config"].get("min_fidelity", 0.80) == 0.80 and
                            result["config"].get("max_passes", 2) == 2):
                            matching_results.append(result)
                    elif parameter == "min_fidelity":
                        if (result["config"]["min_fidelity"] == param_value and
                            result["config"].get("fidelity", 0.95) == 0.95 and
                            result["config"].get("max_passes", 2) == 2):
                            matching_results.append(result)
                    elif parameter == "max_passes":
                        if (result["config"]["max_passes"] == param_value and
                            result["config"].get("fidelity", 0.95) == 0.95 and
                            result["config"].get("min_fidelity", 0.80) == 0.80):
                            matching_results.append(result)
                
                # Use the first matching result if any
                if matching_results:
                    param_results.append((param_value, matching_results[0]))
            
            # If we have at least 2 data points, create a plot
            if len(param_results) >= 2:
                # Create the figure
                plt.figure(figsize=(10, 6))
                
                # Extract data for plotting
                x_values = [p[0] for p in param_results]
                y_values = [p[1]["average_depth"] for p in param_results]
                errors = [p[1]["std_dev"] for p in param_results]
                
                # Create the plot
                plt.errorbar(x_values, y_values, yerr=errors, marker='o', linestyle='-',
                           linewidth=2, markersize=8)
                
                # Add labels and title
                plt.xlabel(f'{parameter.capitalize()} Value')
                plt.ylabel('Average Final Physical Depth')
                plt.title(f'Impact of {parameter.capitalize()} on Depth ({topology.capitalize()} Topology, {strategy} Strategy)')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                # Save the figure
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{parameter}_impact_{topology}_{strategy}.png")
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                plt.close()
                
                print(f"Parameter impact plot for {parameter} on {topology}/{strategy} saved to {output_path}")

def generate_report(results, baselines, output_dir="results"):
    """Generate a report summarizing the test results"""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the successful results
    successful_results = [r for r in results if r["average_depth"] is not None]
    
    # Prepare the report
    report_path = os.path.join(output_dir, "report.txt")
    
    with open(report_path, 'w') as f:
        f.write("Toffoli Optimizer Configuration Test Report\n")
        f.write("=======================================\n\n")
        
        # Write summary statistics
        f.write(f"Total configurations tested: {len(results)}\n")
        f.write(f"Successful configurations: {len(successful_results)}\n")
        f.write(f"Failed configurations: {len(results) - len(successful_results)}\n\n")
        
        # Write baseline information
        f.write("Baseline Naive Physical Depths:\n")
        for baseline in baselines:
            if baseline["success"]:
                f.write(f"  {baseline['topology'].capitalize()}: {baseline['naive_depth']}\n")
            else:
                f.write(f"  {baseline['topology'].capitalize()}: Failed to determine\n")
        f.write("\n")
        
        # Write top results overall
        f.write("Top Configurations by Average Depth (All Topologies):\n")
        sorted_results = sorted(successful_results, key=lambda x: x["average_depth"])
        for i, result in enumerate(sorted_results[:15]):  # Show top 15
            f.write(f"{i+1}. {result['config_name']}: {result['average_depth']:.2f} ± {result['std_dev']:.2f}\n")
        f.write("\n")
        
        # Write top results for each topology
        for topology in TOPOLOGIES:
            f.write(f"Top Configurations for {topology.capitalize()} Topology:\n")
            topology_results = [r for r in successful_results if r["config"]["topology"] == topology]
            sorted_topology_results = sorted(topology_results, key=lambda x: x["average_depth"])
            
            # Get baseline for this topology
            baseline_depth = None
            for baseline in baselines:
                if baseline["topology"] == topology and baseline["success"]:
                    baseline_depth = baseline["naive_depth"]
                    break
            
            # Show top 10 for each topology
            for i, result in enumerate(sorted_topology_results[:10]):
                if baseline_depth is not None:
                    improvement = (baseline_depth - result["average_depth"]) / baseline_depth * 100
                    f.write(f"{i+1}. {result['config_name']}: {result['average_depth']:.2f} ± {result['std_dev']:.2f} ")
                    f.write(f"({improvement:.2f}% improvement over baseline)\n")
                else:
                    f.write(f"{i+1}. {result['config_name']}: {result['average_depth']:.2f} ± {result['std_dev']:.2f}\n")
            f.write("\n")
        
        # Add a dedicated section for transpiler performance
        f.write("Qiskit Transpiler Performance Comparison:\n")
        f.write("--------------------------------------\n")
        
        for topology in TOPOLOGIES:
            f.write(f"\n{topology.capitalize()} Topology Transpiler Results:\n")
            
            # Get transpiler results for this topology
            transpiler_results = [r for r in successful_results if
                                r["config"]["topology"] == topology and
                                "TRANSPILER_L" in r["config"]["strategy"]]
            
            # Sort by optimization level
            sorted_transpiler = sorted(transpiler_results,
                                     key=lambda x: int(x["config"]["strategy"].split("_")[1][1:]))
            
            # Get baseline for this topology
            baseline_depth = None
            for baseline in baselines:
                if baseline["topology"] == topology and baseline["success"]:
                    baseline_depth = baseline["naive_depth"]
                    break
            
            if sorted_transpiler:
                for result in sorted_transpiler:
                    level = result["config"]["strategy"].split("_")[1]
                    depth = result["average_depth"]
                    std_dev = result["std_dev"]
                    
                    if baseline_depth is not None:
                        improvement = (baseline_depth - depth) / baseline_depth * 100
                        f.write(f"  Transpiler {level}: {depth:.2f} ± {std_dev:.2f}")
                        f.write(f" ({improvement:.2f}% improvement over baseline)\n")
                    else:
                        f.write(f"  Transpiler {level}: {depth:.2f} ± {std_dev:.2f}\n")
            else:
                f.write("  No transpiler results available for this topology\n")
        
        f.write("\n")
        
        # Write results grouped by strategy for each topology
        for topology in TOPOLOGIES:
            f.write(f"Comparison of Strategies ({topology.capitalize()} Topology, Default Parameters):\n")
            linear_default_results = [r for r in successful_results
                                  if r["config"]["topology"] == topology and
                                   r["config"].get("fidelity", 0.95) == 0.95 and
                                   r["config"].get("min_fidelity", 0.80) == 0.80 and
                                   r["config"].get("max_passes", 2) == 2]
            
            # Group by strategy
            by_strategy = {}
            for result in linear_default_results:
                strategy = result["config"]["strategy"]
                # Handle DEPTH_FIDELITY_BALANCE specially
                if strategy == "DEPTH_FIDELITY_BALANCE":
                    depth_weight = result["config"].get("depth_weight", "unknown")
                    fidelity_weight = result["config"].get("fidelity_weight", "unknown")
                    strategy = f"{strategy} (D:{depth_weight}/F:{fidelity_weight})"
                by_strategy[strategy] = result
            
            # Sort by strategy name
            for strategy in sorted(by_strategy.keys()):
                result = by_strategy[strategy]
                f.write(f"  {strategy}: {result['average_depth']:.2f} ± {result['std_dev']:.2f}\n")
            f.write("\n")
        
        # Write conclusion - identify best configurations
        f.write("Summary of Best Configurations:\n")
        for topology in TOPOLOGIES:
            topology_results = [r for r in successful_results if r["config"]["topology"] == topology]
            if topology_results:
                best_result = min(topology_results, key=lambda x: x["average_depth"])
                baseline_depth = None
                for baseline in baselines:
                    if baseline["topology"] == topology and baseline["success"]:
                        baseline_depth = baseline["naive_depth"]
                        break
                
                f.write(f"  Best for {topology.capitalize()}: {best_result['config_name']}\n")
                f.write(f"    Average depth: {best_result['average_depth']:.2f} ± {best_result['std_dev']:.2f}\n")
                
                if baseline_depth is not None:
                    improvement = (baseline_depth - best_result["average_depth"]) / baseline_depth * 100
                    f.write(f"    Improvement over baseline: {improvement:.2f}%\n")
                f.write("\n")
                
                # Also show best non-transpiler configuration
                non_transpiler_results = [r for r in topology_results
                                        if "TRANSPILER" not in r["config"]["strategy"]]
                if non_transpiler_results:
                    best_non_transpiler = min(non_transpiler_results, key=lambda x: x["average_depth"])
                    f.write(f"  Best Non-Transpiler for {topology.capitalize()}: {best_non_transpiler['config_name']}\n")
                    f.write(f"    Average depth: {best_non_transpiler['average_depth']:.2f} ± {best_non_transpiler['std_dev']:.2f}\n")
                    
                    if baseline_depth is not None:
                        improvement = (baseline_depth - best_non_transpiler["average_depth"]) / baseline_depth * 100
                        f.write(f"    Improvement over baseline: {improvement:.2f}%\n")
                    f.write("\n")
        
        # Write list of failed configurations
        failed_results = [r for r in results if r["average_depth"] is None]
        if failed_results:
            f.write(f"Failed Configurations ({len(failed_results)}):\n")
            for i, result in enumerate(failed_results):
                f.write(f"{i+1}. {result['config_name']}\n")
    
    print(f"Report saved to {report_path}")
    return report_path

def run_all_tests(input_file="circuit", output_dir="toffoli_optimizer_results", num_runs=5, timeout=300):
    """Run all tests and generate reports and visualizations"""
    print("Toffoli Optimizer Configuration Test")
    print("===================================")
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run baselines for each topology
    print("\nRunning baseline measurements for each topology...")
    baselines = []
    
    # Check for existing baseline results
    baseline_path = os.path.join(output_dir, "baselines.json")
    if os.path.exists(baseline_path):
        try:
            with open(baseline_path, 'r') as f:
                baselines = json.load(f)
            print(f"Loaded {len(baselines)} existing baseline results")
        except:
            print("Failed to load existing baselines, measuring again")
            baselines = []
    
    # Run baselines if needed
    if not baselines:
        for topology in TOPOLOGIES:
            baseline = run_baseline(topology, input_file=input_file, timeout=timeout)
            baselines.append(baseline)
        
        # Save the baseline results
        with open(baseline_path, 'w') as f:
            json.dump(baselines, f, indent=2)
    
    # Check if any baselines failed
    failed_baselines = [b for b in baselines if not b["success"]]
    if failed_baselines:
        print(f"Warning: {len(failed_baselines)} baselines failed to determine naive physical depth")
        for b in failed_baselines:
            print(f"  {b['topology']}")
    
    # Generate configurations to test
    test_configs = generate_test_configurations()
    print(f"\nGenerated {len(test_configs)} test configurations")
    
    # Explicitly add transpiler configs for each topology
    for topology in TOPOLOGIES:
        for level in [1, 2, 3]:
            transpiler_config = {
                "topology": topology,
                "strategy": f"TRANSPILER_L{level}",
                "fidelity": 0.95,  # Default
                "min_fidelity": 0.80,  # Default
                "max_passes": 2  # Default
            }
            
            # Check if it's already in the list
            config_key = json.dumps(transpiler_config, sort_keys=True)
            if config_key not in [json.dumps(c, sort_keys=True) for c in test_configs]:
                test_configs.append(transpiler_config)
                print(f"Added explicit transpiler config: {topology.capitalize()} TRANSPILER_L{level}")
    
    # Save the test configurations
    configs_path = os.path.join(output_dir, "test_configurations.json")
    with open(configs_path, 'w') as f:
        json.dump(test_configs, f, indent=2)
    
    # Run tests for each configuration
    all_results = []
    
    # Check for existing results file
    all_results_path = os.path.join(output_dir, "all_results.json")
    existing_results = {}
    if os.path.exists(all_results_path):
        try:
            with open(all_results_path, 'r') as f:
                existing_data = json.load(f)
                for result in existing_data:
                    config_hash = result.get("config_hash")
                    if config_hash:
                        existing_results[config_hash] = result
            print(f"Loaded {len(existing_results)} existing results")
        except:
            print("Failed to load existing results, starting fresh")
    
    for i, config in enumerate(test_configs):
        config_hash = compute_config_hash(config)
        print(f"\nTesting configuration {i+1}/{len(test_configs)}: {config_to_string(config)} (hash: {config_hash})")
        
        # Check if we already have results for this configuration
        if config_hash in existing_results:
            print(f"Using existing results for this configuration")
            result = existing_results[config_hash]
        else:
            # Run the configuration multiple times and average the results
            result = run_multiple_times(
                config,
                input_file=input_file,
                num_runs=num_runs,
                timeout=timeout,
                results_dir=output_dir,
                config_hash=config_hash
            )
        
        all_results.append(result)
        
        # Save all results to a single file periodically
        if (i + 1) % 5 == 0 or i + 1 == len(test_configs):
            with open(all_results_path, 'w') as f:
                # Strip out the detailed individual_results to keep file size manageable
                simplified_results = []
                for res in all_results:
                    simplified_result = {
                        "config": res["config"],
                        "config_name": res["config_name"],
                        "config_hash": res.get("config_hash", compute_config_hash(res["config"])),
                        "num_runs": res["num_runs"],
                        "successful_runs": res["successful_runs"],
                        "average_depth": float(res["average_depth"]) if res["average_depth"] is not None else None,
                        "min_depth": float(res["min_depth"]) if res["min_depth"] is not None else None,
                        "max_depth": float(res["max_depth"]) if res["max_depth"] is not None else None,
                        "std_dev": float(res["std_dev"]) if res["std_dev"] is not None else None
                    }
                    simplified_results.append(simplified_result)
                json.dump(simplified_results, f, indent=2)
            print(f"Saved intermediate results ({i+1}/{len(test_configs)} configurations completed)")
    
    # Generate visualizations
    print("\nGenerating visualization plots...")
    plot_results_by_topology(all_results, baselines, output_dir=output_dir)
    plot_overall_comparison(all_results, baselines, output_dir=output_dir)
    plot_strategy_comparison(all_results, baselines, output_dir=output_dir)
    
    # Generate specialized transpiler comparison
    print("\nGenerating transpiler comparison plots...")
    plot_transpiler_comparison(all_results, baselines, output_dir=output_dir)
    
    # Generate parameter impact plots for key parameters
    print("\nGenerating parameter impact plots...")
    promising_strategies = ["STANDARD", "ULTRA_DEPTH_REDUCTION", "DEPTH_REDUCTION", "HYBRID"]
    for param in ["fidelity", "min_fidelity", "max_passes"]:
        plot_parameter_impact(all_results, param, TOPOLOGIES, promising_strategies, output_dir=output_dir)
    
    # Generate report
    print("\nGenerating summary report...")
    report_path = generate_report(all_results, baselines, output_dir=output_dir)
    
    print("\nAll tests completed!")
    print(f"Results saved to {output_dir}")
    print(f"Summary report: {report_path}")

def main():
    """Main function to run the script with command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Test Toffoli Optimizer configurations')
    parser.add_argument('--input', type=str, default='circuit',
                        help='Input file name without extension (default: circuit)')
    parser.add_argument('--output-dir', type=str, default='toffoli_optimizer_results',
                        help='Output directory for results (default: toffoli_optimizer_results)')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of runs per configuration for averaging (default: 5)')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout in seconds for each optimizer run (default: 300)')
    
    args = parser.parse_args()
    
    run_all_tests(
        input_file=args.input,
        output_dir=args.output_dir,
        num_runs=args.runs,
        timeout=args.timeout
    )

if __name__ == "__main__":
    main()
