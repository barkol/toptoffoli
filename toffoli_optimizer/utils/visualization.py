"""
Visualization Utilities for Toffoli Quantum Circuits

This module provides utilities for visualizing quantum circuits, 
including circuit diagrams, performance metrics, and benchmark results.
"""

import os
import time
import gc  # For garbage collection
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import traceback

# Try to import Qiskit
try:
    from qiskit import QuantumCircuit
    from qiskit.visualization import circuit_drawer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Some visualization features will be disabled.")

def save_circuit_image(circuit, filename, output_dir="circuit_images", figsize=(12, 8),
                      use_text_mode=True, dpi=72, format='png', garbage_collect=True):
    """
    Save a circuit visualization as an image or text file.
    
    Args:
        circuit: Quantum circuit to visualize
        filename: Base filename (without extension)
        output_dir: Directory to save the image to
        figsize: Figure size (width, height) in inches
        use_text_mode: Save as text instead of image (more memory efficient)
        dpi: DPI for image (ignored in text mode)
        format: Image format (png, pdf, svg, etc.) (ignored in text mode)
        garbage_collect: Run garbage collection after saving
        
    Returns:
        str: Path to the saved file
    """
    if not QISKIT_AVAILABLE:
        print("Warning: Qiskit not available. Cannot save circuit image.")
        return None
        
    if circuit is None:
        print(f"Warning: Cannot save circuit image for None circuit")
        return None
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique timestamp for the filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    if use_text_mode:
        # Save in text format (ASCII art)
        output_path = os.path.join(output_dir, f"{filename}_{timestamp}.txt")
        
        try:
            # Use text circuit_drawer for memory efficiency
            text_circuit = circuit_drawer(circuit, output='text', filename=None)
            
            # Convert TextDrawing object to string if needed
            if hasattr(text_circuit, '__str__'):
                text_content = str(text_circuit)
            else:
                text_content = text_circuit
                
            with open(output_path, 'w') as f:
                f.write(text_content)
                
            print(f"Circuit text representation saved to {output_path}")
            
            if garbage_collect:
                # Force garbage collection to free memory
                gc.collect()
                
            return output_path
            
        except Exception as e:
            print(f"Error saving circuit text: {e}")
            # Try alternative approach
            try:
                # Use filename parameter directly
                alt_output_path = os.path.join(output_dir, f"{filename}_{timestamp}_alt.txt")
                circuit_drawer(circuit, output='text', filename=alt_output_path)
                print(f"Circuit text representation saved to {alt_output_path} (alternative method)")
                return alt_output_path
            except Exception as alt_e:
                print(f"Alternative text saving also failed: {alt_e}")
                return None
    else:
        # Create the output path
        output_path = os.path.join(output_dir, f"{filename}_{timestamp}.{format}")
        
        try:
            # Create a new figure to avoid memory issues with existing figures
            plt.figure(figsize=figsize)
            
            # Draw the circuit
            circuit_drawer(circuit, output='mpl', style={'name': 'iqx'})
            
            # Save the figure
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format=format)
            
            # Close the figure to free memory
            plt.close()
            
            print(f"Circuit image saved to {output_path}")
            
            if garbage_collect:
                # Force garbage collection to free memory
                gc.collect()
                
            return output_path
            
        except Exception as e:
            # Ensure figure is closed even on error
            plt.close()
            print(f"Error saving circuit image: {e}")
            return None

def save_circuit_text(circuit, filename, output_dir="circuit_text"):
    """
    Save a circuit as a text-only representation (most memory efficient).
    
    Args:
        circuit: Quantum circuit to visualize
        filename: Base filename (without extension)
        output_dir: Directory to save the text file to
        
    Returns:
        str: Path to the saved file
    """
    if not QISKIT_AVAILABLE:
        print("Warning: Qiskit not available. Cannot save circuit text.")
        return None
        
    if circuit is None:
        print(f"Warning: Cannot save circuit text for None circuit")
        return None
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique timestamp for the filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save in text format (ASCII art)
    output_path = os.path.join(output_dir, f"{filename}_{timestamp}.txt")
    
    try:
        # Method 1: Use direct filename output
        circuit_drawer(circuit, output='text', filename=output_path)
        print(f"Circuit text representation saved to {output_path}")
        
        # Force garbage collection to free memory
        gc.collect()
            
        return output_path
        
    except Exception as e:
        print(f"Error with direct text saving: {e}")
        # Try alternative method
        try:
            # Method 2: Get the text and write it manually
            text_drawing = circuit_drawer(circuit, output='text', filename=None)
            text_str = str(text_drawing)
            
            with open(output_path, 'w') as f:
                f.write(text_str)
                
            print(f"Circuit text representation saved to {output_path} (alternative method)")
            gc.collect()
            return output_path
            
        except Exception as alt_e:
            print(f"Alternative text saving also failed: {alt_e}")
            return None

def save_circuit_stats(circuit, filename, output_dir="circuit_stats"):
    """
    Save just the circuit statistics without visualization (minimal memory usage).
    
    Args:
        circuit: Quantum circuit to analyze
        filename: Base filename (without extension)
        output_dir: Directory to save the stats to
        
    Returns:
        str: Path to the saved file
    """
    if not QISKIT_AVAILABLE:
        print("Warning: Qiskit not available. Cannot save circuit stats.")
        return None
        
    if circuit is None:
        print(f"Warning: Cannot save circuit stats for None circuit")
        return None
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique timestamp for the filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create the output path
    output_path = os.path.join(output_dir, f"{filename}_{timestamp}.txt")
    
    try:
        # Count operations
        gate_counts = circuit.count_ops()
        
        # Write the statistics to a file
        with open(output_path, 'w') as f:
            f.write(f"Circuit Statistics for {filename}\n")
            f.write(f"{'='*40}\n")
            f.write(f"Number of qubits: {circuit.num_qubits}\n")
            f.write(f"Depth: {circuit.depth()}\n")
            f.write(f"Total gates: {sum(gate_counts.values())}\n")
            f.write(f"\nGate counts:\n")
            for gate, count in sorted(gate_counts.items()):
                f.write(f"  {gate}: {count}\n")
            
            # Include instructions list for debugging
            f.write(f"\nInstructions:\n")
            for i, inst in enumerate(circuit.data):
                try:
                    gate_name = inst.operation.name
                    qubits = [q.index for q in inst.qubits]
                    f.write(f"  {i}: {gate_name} on qubits {qubits}\n")
                except:
                    f.write(f"  {i}: {inst}\n")
        
        print(f"Circuit statistics saved to {output_path}")
        
        # Force garbage collection to free memory
        gc.collect()
            
        return output_path
        
    except Exception as e:
        print(f"Error saving circuit stats: {e}")
        return None

def save_circuit_safely(circuit, name, output_dir):
    """
    Save a quantum circuit to a file for later analysis, handling errors gracefully.
    
    Args:
        circuit: Quantum circuit to save
        name: Base name for the file
        output_dir: Directory to save to
        
    Returns:
        str: Path to the saved file, or None if saving failed
    """
    if not QISKIT_AVAILABLE:
        print("Warning: Qiskit not available. Cannot save circuit.")
        return None
        
    if circuit is None:
        print(f"Warning: Cannot save None circuit")
        return None
    
    # Create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {output_dir}: {e}")
        return None
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # First try to save as QASM
    try:
        # Try to import QASM dumping functionality
        try:
            from qiskit.qasm2 import dumps
            qasm_available = True
        except ImportError:
            # Check if the old method is available
            qasm_available = hasattr(circuit, 'qasm')
            
        if qasm_available:
            qasm_filename = f"{name}_{timestamp}.qasm"
            qasm_filepath = os.path.join(output_dir, qasm_filename)
            
            try:
                # First try new method
                from qiskit.qasm2 import dumps
                qasm_str = dumps(circuit)
                with open(qasm_filepath, 'w') as f:
                    f.write(qasm_str)
            except (ImportError, AttributeError):
                # Fallback to old method
                if hasattr(circuit, 'qasm'):
                    circuit.qasm(filename=qasm_filepath)
                else:
                    raise ImportError("No QASM export method available")
            
            print(f"Circuit saved to QASM file: {qasm_filepath}")
            return qasm_filepath
            
    except Exception as e:
        print(f"Warning: Failed to save as QASM: {e}")
        # Continue to backup method
    
    # Backup method: Save circuit statistics to a readable text file
    try:
        return save_circuit_stats(circuit, name, output_dir)
    except Exception as e:
        print(f"Error saving circuit details: {e}")
        traceback.print_exc()
        return None

def save_benchmark_circuits(benchmark_result, output_dir="benchmark_circuits", save_as_text=True):
    """
    Save all circuits from a benchmark result.
    
    Args:
        benchmark_result: Dictionary with benchmark results
        output_dir: Directory to save the circuits to
        save_as_text: Whether to save as text instead of images
        
    Returns:
        dict: Dictionary with paths to saved circuit files
    """
    if not QISKIT_AVAILABLE:
        print("Warning: Qiskit not available. Cannot save benchmark circuits.")
        return {}
        
    # Create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {output_dir}: {e}")
        return {}
    
    saved_paths = {}
    
    # Try to save each circuit in the benchmark result
    try:
        # Save logical circuit
        if "logical" in benchmark_result and "circuit" in benchmark_result["logical"]:
            circuit = benchmark_result["logical"]["circuit"]
            path = save_circuit_image(circuit, "logical_circuit", output_dir, use_text_mode=save_as_text)
            if path:
                saved_paths["logical"] = path
        
        # Save original circuit
        if "original" in benchmark_result and "circuit" in benchmark_result["original"]:
            circuit = benchmark_result["original"]["circuit"]
            path = save_circuit_image(circuit, "original_circuit", output_dir, use_text_mode=save_as_text)
            if path:
                saved_paths["original"] = path
        
        # Save Toffoli optimizer result
        if "toffoli_optimizer" in benchmark_result:
            toffoli_result = benchmark_result["toffoli_optimizer"]
            
            # Handle different result structures
            if "mapped" in toffoli_result and "circuit" in toffoli_result["mapped"]:
                circuit = toffoli_result["mapped"]["circuit"]
                path = save_circuit_image(circuit, "toffoli_mapped_circuit", output_dir, use_text_mode=save_as_text)
                if path:
                    saved_paths["toffoli_mapped"] = path
            
            if "optimized" in toffoli_result and "circuit" in toffoli_result["optimized"]:
                circuit = toffoli_result["optimized"]["circuit"]
                path = save_circuit_image(circuit, "toffoli_optimized_circuit", output_dir, use_text_mode=save_as_text)
                if path:
                    saved_paths["toffoli_optimized"] = path
            
            # Handle direct circuit field if present
            if "circuit" in toffoli_result:
                circuit = toffoli_result["circuit"]
                path = save_circuit_image(circuit, "toffoli_circuit", output_dir, use_text_mode=save_as_text)
                if path:
                    saved_paths["toffoli"] = path
        
        # Save Qiskit transpiler result
        if "qiskit_transpiler" in benchmark_result and "circuit" in benchmark_result["qiskit_transpiler"]:
            circuit = benchmark_result["qiskit_transpiler"]["circuit"]
            path = save_circuit_image(circuit, "qiskit_circuit", output_dir, use_text_mode=save_as_text)
            if path:
                saved_paths["qiskit"] = path
        
        print(f"Saved {len(saved_paths)} benchmark circuits to {output_dir}")
        return saved_paths
        
    except Exception as e:
        print(f"Error saving benchmark circuits: {e}")
        traceback.print_exc()
        return saved_paths

def plot_optimization_results(results, output_dir="optimization_plots"):
    """
    Create visualizations of optimization results.
    
    Args:
        results: Dictionary with optimization results
        output_dir: Directory to save the plots to
        
    Returns:
        list: List of paths to saved plot files
    """
    # Create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {output_dir}: {e}")
        return []
    
    saved_plots = []
    
    # Extract relevant metrics
    try:
        # Ensure we have the necessary data
        if not results or not isinstance(results, dict):
            print("Warning: Invalid results data for plotting")
            return []
        
        logical_depth = results.get("logical", {}).get("depth", 0)
        naive_depth = results.get("naive_physical", {}).get("depth", 0) or results.get("original", {}).get("depth", 0)
        
        # The optimized depth might be in different places depending on the optimizer used
        optimized_depth = 0
        if "optimized" in results:
            optimized_depth = results["optimized"].get("depth", 0)
        elif "final" in results:
            optimized_depth = results["final"].get("depth", 0)
        
        # The physical depth might be under 'mapped' or directly
        physical_depth = 0
        if "mapped" in results:
            physical_depth = results["mapped"].get("depth", 0)
        
        # Calculate depth reductions
        if logical_depth > 0:
            logical_reduction = (logical_depth - optimized_depth) / logical_depth * 100
        else:
            logical_reduction = 0
            
        if naive_depth > 0:
            physical_reduction = (naive_depth - physical_depth) / naive_depth * 100
        else:
            physical_reduction = 0
        
        # Create depth comparison bar chart
        plt.figure(figsize=(10, 6))
        
        depths = [logical_depth, naive_depth, optimized_depth, physical_depth]
        labels = ['Logical Circuit', 'Naive Physical', 'Optimized Logical', 'Final Physical']
        
        plt.bar(labels, depths)
        plt.ylabel('Circuit Depth')
        plt.title('Circuit Depth Comparison')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Add depth values on top of bars
        for i, depth in enumerate(depths):
            plt.text(i, depth + 1, f'{depth}', ha='center')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        depth_plot_path = os.path.join(output_dir, f"depth_comparison_{timestamp}.png")
        plt.savefig(depth_plot_path)
        plt.close()
        saved_plots.append(depth_plot_path)
        
        # Create reduction percentage bar chart
        plt.figure(figsize=(10, 6))
        
        reductions = [logical_reduction, physical_reduction]
        labels = ['From Logical Circuit', 'From Naive Physical']
        
        plt.bar(labels, reductions, color='green')
        plt.ylabel('Depth Reduction (%)')
        plt.title('Optimization Depth Reduction')
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage values on top of bars
        for i, reduction in enumerate(reductions):
            plt.text(i, reduction + 1, f'{reduction:.1f}%', ha='center')
        
        plt.tight_layout()
        
        # Save the plot
        reduction_plot_path = os.path.join(output_dir, f"depth_reduction_{timestamp}.png")
        plt.savefig(reduction_plot_path)
        plt.close()
        saved_plots.append(reduction_plot_path)
        
        print(f"Saved optimization result plots to {output_dir}")
        return saved_plots
        
    except Exception as e:
        print(f"Error plotting optimization results: {e}")
        traceback.print_exc()
        return saved_plots

def visualize_optimization(circuit, report, filename='optimization_report'):
    """

    Create a comprehensive visualization of optimization results with performance metrics.
    
    This function generates multiple visualization files to help analyze the optimization
    results, including circuit diagrams, performance metrics charts, and depth reduction
    visualizations. It provides a holistic view of how the optimization has affected 
    the circuit structure and performance.
    
    Args:
        circuit (QuantumCircuit): The optimized quantum circuit to visualize.
            This should be the final output circuit from the optimization process.
        
        report (dict): Optimization report containing metrics and results.
            Should include keys such as:
            - 'original_depth': Depth of the original circuit
            - 'optimized_depth': Depth of the optimized circuit
            - 'depth_reduction': Percentage reduction in depth (0.0-1.0)
            - Additional metrics may include gate counts, fidelity, etc.
        
        filename (str): Base filename for the visualization outputs.
            This name will be used as a prefix for all generated files.
            If it includes a directory path, that directory will be created if needed.
    
    Returns:
        None: The function generates files on disk rather than returning values.
    
    Generated Files:
        - {filename}_circuit.png: Visualization of the optimized circuit
        - {filename}_depth_reduction.png: Bar chart showing depth reduction
        - {filename}_report.json: JSON file with all serializable optimization metrics
    
    Notes:
        - For large circuits, the circuit visualization may be simplified
        - The function requires matplotlib for plotting
        - All files are saved with the same base filename but different extensions
    """
    try:
        from qiskit.visualization import circuit_drawer
        import matplotlib.pyplot as plt
        
        # Create output directory
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Draw the circuit
        fig = plt.figure(figsize=(12, 8))
        circuit_drawer(circuit, output='mpl', style={'name': 'iqx'})
        plt.title(f"Optimized Circuit - Depth: {report['optimized_depth']}")
        plt.tight_layout()
        plt.savefig(f"{filename}_circuit.png")
        plt.close()
        
        # Create a bar chart showing the optimization
        fig, ax = plt.subplots(figsize=(10, 6))
        original_depth = report['original_depth']
        optimized_depth = report['optimized_depth']
        
        ax.bar(['Original', 'Optimized'], [original_depth, optimized_depth])
        ax.set_ylabel('Circuit Depth')
        ax.set_title('Optimization Results')
        
        # Add percentage reduction
        reduction = report['depth_reduction'] * 100
        plt.text(1, optimized_depth / 2, f"{reduction:.1f}% reduction", 
                 ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{filename}_depth_reduction.png")
        plt.close()
        
        # Save the report as JSON
        import json
        serializable_report = {}
        for key, value in report.items():
            # Skip non-serializable items
            if key not in ['replacements']:
                serializable_report[key] = value
        
        with open(f"{filename}_report.json", 'w') as f:
            json.dump(serializable_report, f, indent=4)
        
        print(f"Visualization saved to {filename}_*.png")
        
    except Exception as e:
        print(f"Error creating visualization: {e}")



"""
Memory-Optimized Visualization Utilities

This module provides optimized visualization functions for quantum circuits
with improved memory management for large circuits.
"""

import os
import time
import gc  # For garbage collection
import traceback
from datetime import datetime
import numpy as np

# Try to import Matplotlib with memory optimizations
try:
    import matplotlib
    # Use the Agg backend for better memory management
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available. Visualization will be limited to text output.")

# Try to import Qiskit
try:
    from qiskit import QuantumCircuit
    from qiskit.visualization import circuit_drawer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Visualization will be limited.")

def memory_optimized_save_circuit_image(circuit, filename, output_dir="circuit_images",
                                      max_qubits_for_image=20, dpi=72, format='png'):
    """
    Save a circuit visualization with optimized memory usage.
    
    For large circuits, this will automatically fall back to text representation
    to avoid memory issues.
    
    Args:
        circuit: Quantum circuit to visualize
        filename: Base filename (without extension)
        output_dir: Directory to save the image to
        max_qubits_for_image: Maximum number of qubits for image rendering
        dpi: DPI for image
        format: Image format (png, pdf, svg, etc.)
        
    Returns:
        str: Path to the saved file
    """
    if not QISKIT_AVAILABLE:
        print("Warning: Qiskit not available. Cannot save circuit image.")
        return None
        
    if circuit is None:
        print(f"Warning: Cannot save circuit image for None circuit")
        return None
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique timestamp for the filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Check if circuit is too large for image rendering
    if circuit.num_qubits > max_qubits_for_image or len(circuit.data) > max_qubits_for_image * 10:
        print(f"Circuit too large for image rendering (has {circuit.num_qubits} qubits, "
              f"{len(circuit.data)} gates). Using text format.")
        return save_circuit_text(circuit, filename, output_dir)
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Using text format.")
        return save_circuit_text(circuit, filename, output_dir)
    
    # Create the output path
    output_path = os.path.join(output_dir, f"{filename}_{timestamp}.{format}")
    
    try:
        # Create a new figure with a fresh memory space
        plt.figure(figsize=(12, min(8, 0.5 * circuit.num_qubits)))
        
        # Draw the circuit with memory optimizations
        try:
            # For smaller circuits, use mpl style
            if circuit.num_qubits <= 10 and len(circuit.data) <= 50:
                circuit_drawer(circuit, output='mpl', style={'name': 'iqx'})
            else:
                # For medium circuits, use a simpler style
                circuit_drawer(circuit, output='mpl',
                              style={'name': 'iqx', 'subfontsize': 8, 'compress': True})
        except Exception as drawer_error:
            print(f"Error with circuit_drawer: {drawer_error}")
            # Fallback to very simple drawing
            plt.text(0.5, 0.5, f"Circuit: {circuit.num_qubits} qubits, "
                     f"{len(circuit.data)} gates, depth {circuit.depth()}",
                     ha='center', va='center')
        
        # Save the figure with tight layout to minimize whitespace
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format=format)
        
        # Close the figure immediately to free memory
        plt.close('all')
        
        # Force garbage collection
        gc.collect()
        
        print(f"Circuit image saved to {output_path}")
        return output_path
        
    except Exception as e:
        # Ensure all figures are closed even on error
        plt.close('all')
        print(f"Error saving circuit image: {e}")
        traceback.print_exc()
        
        # Try text format as fallback
        print("Falling back to text format")
        return save_circuit_text(circuit, filename, output_dir)

def save_circuit_text(circuit, filename, output_dir="circuit_text"):
    """
    Save a circuit as a text-only representation (most memory efficient).
    
    Args:
        circuit: Quantum circuit to visualize
        filename: Base filename (without extension)
        output_dir: Directory to save the text file to
        
    Returns:
        str: Path to the saved file
    """
    if not QISKIT_AVAILABLE:
        print("Warning: Qiskit not available. Cannot save circuit text.")
        return None
        
    if circuit is None:
        print(f"Warning: Cannot save circuit text for None circuit")
        return None
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique timestamp for the filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save in text format (ASCII art)
    output_path = os.path.join(output_dir, f"{filename}_{timestamp}.txt")
    
    # Check if circuit is very large
    if circuit.num_qubits > 50 or len(circuit.data) > 500:
        # For very large circuits, just save statistics instead of full text representation
        try:
            with open(output_path, 'w') as f:
                f.write(f"Circuit Statistics for {filename}\n")
                f.write(f"{'='*40}\n")
                f.write(f"Number of qubits: {circuit.num_qubits}\n")
                f.write(f"Circuit depth: {circuit.depth()}\n")
                f.write(f"Total gates: {len(circuit.data)}\n")
                
                # Count gate types
                gate_counts = {}
                for inst in circuit.data:
                    gate_name = inst.operation.name
                    gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
                
                f.write("\nGate counts:\n")
                for gate, count in sorted(gate_counts.items()):
                    f.write(f"  {gate}: {count}\n")
                
                # Show sample of gates (first 20)
                f.write("\nSample of gates (first 20):\n")
                for i, inst in enumerate(circuit.data[:20]):
                    if i >= 20:
                        break
                    gate_name = inst.operation.name
                    qubits = [q.index if hasattr(q, 'index') else q._index for q in inst.qubits]
                    f.write(f"  {i}: {gate_name} on qubits {qubits}\n")
                
                if len(circuit.data) > 20:
                    f.write(f"  ... and {len(circuit.data) - 20} more gates\n")
            
            print(f"Circuit statistics saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving circuit statistics: {e}")
            return None
    
    try:
        # Method 1: Use direct filename output (most memory efficient)
        circuit_drawer(circuit, output='text', filename=output_path)
        print(f"Circuit text representation saved to {output_path}")
        
        # Force garbage collection to free memory
        gc.collect()
            
        return output_path
        
    except Exception as e:
        print(f"Error with direct text saving: {e}")
        # Try alternative method
        try:
            # Method 2: Get the text and write it manually, in chunks to manage memory
            text_drawing = circuit_drawer(circuit, output='text', filename=None)
            text_str = str(text_drawing)
            
            with open(output_path, 'w') as f:
                # Write in small chunks to prevent memory issues with very large circuits
                chunk_size = 10000  # Characters per chunk
                for i in range(0, len(text_str), chunk_size):
                    f.write(text_str[i:i+chunk_size])
                    # Free memory after writing each chunk
                    if i % (chunk_size * 10) == 0:
                        gc.collect()
                
            print(f"Circuit text representation saved to {output_path} (alternative method)")
            gc.collect()
            return output_path
            
        except Exception as alt_e:
            print(f"Alternative text saving also failed: {alt_e}")
            return None

def save_circuit_stats_no_matplotlib(circuit, filename, output_dir="circuit_stats"):
    """
    Save circuit statistics without any dependency on Matplotlib.
    
    This function is guaranteed to work even for very large circuits
    with minimal memory usage.
    
    Args:
        circuit: Quantum circuit to analyze
        filename: Base filename (without extension)
        output_dir: Directory to save the stats to
        
    Returns:
        str: Path to the saved file
    """
    if not QISKIT_AVAILABLE:
        print("Warning: Qiskit not available. Cannot save circuit stats.")
        return None
        
    if circuit is None:
        print(f"Warning: Cannot save circuit stats for None circuit")
        return None
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique timestamp for the filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create the output path
    output_path = os.path.join(output_dir, f"{filename}_{timestamp}.txt")
    
    try:
        # Count operations in a memory-efficient way
        gate_counts = {}
        for inst in circuit.data:
            try:
                gate_name = inst.operation.name
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            except:
                # Skip if name attribute is not available
                pass
        
        # Calculate circuit depth
        try:
            depth = circuit.depth()
        except:
            depth = "Unknown (calculation failed)"
        
        # Write the statistics to a file
        with open(output_path, 'w') as f:
            f.write(f"Circuit Statistics for {filename}\n")
            f.write(f"{'='*40}\n")
            f.write(f"Number of qubits: {circuit.num_qubits}\n")
            f.write(f"Depth: {depth}\n")
            f.write(f"Total gates: {len(circuit.data)}\n")
            f.write(f"\nGate counts:\n")
            for gate, count in sorted(gate_counts.items()):
                f.write(f"  {gate}: {count}\n")
            
            # Include a sample of instructions for debugging
            f.write(f"\nSample of instructions (first 20):\n")
            for i, inst in enumerate(circuit.data[:20]):
                try:
                    gate_name = inst.operation.name
                    qubits = [q.index if hasattr(q, 'index') else q._index for q in inst.qubits]
                    f.write(f"  {i}: {gate_name} on qubits {qubits}\n")
                except:
                    f.write(f"  {i}: {inst} (error parsing details)\n")
            
            if len(circuit.data) > 20:
                f.write(f"  ... and {len(circuit.data) - 20} more instructions\n")
        
        print(f"Circuit statistics saved to {output_path}")
        
        # Force garbage collection to free memory
        gc.collect()
            
        return output_path
        
    except Exception as e:
        print(f"Error saving circuit stats: {e}")
        traceback.print_exc()
        return None

def memory_efficient_plot_comparison(results, output_file, max_memory_mb=500):
    """
    Create a comparison plot with explicit memory management.
    
    Args:
        results: Dictionary with optimization results
        output_file: Output file path for the plot
        max_memory_mb: Maximum memory to use in MB
        
    Returns:
        bool: Whether the plot was successfully created
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: Matplotlib not available. Cannot create comparison plot.")
        return False
    
    try:
        # Ensure all previous figures are closed
        plt.close('all')
        
        # Set a memory limit for the figure
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if max_memory_mb > 0:
            # Convert MB to bytes
            max_memory = max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (max_memory, hard))
        
        # Extract data for plotting
        methods = []
        original_depths = []
        optimized_depths = []
        depth_reductions = []
        
        # Prepare data in chunks to limit memory usage
        for method, method_results in results.items():
            # Extract results
            method_name = str(method)
            
            # Get metrics safely, with defaults if keys are missing
            original_depth = method_results.get("original_depth", 0)
            optimized_depth = method_results.get("optimized_depth", 0)
            depth_reduction = method_results.get("depth_reduction", 0)
            
            # Add to lists
            methods.append(method_name)
            original_depths.append(original_depth)
            optimized_depths.append(optimized_depth)
            depth_reductions.append(depth_reduction)
            
            # Free memory after processing each method
            if len(methods) % 5 == 0:
                gc.collect()
        
        # Create the figure
        plt.figure(figsize=(10, 6))
        
        # Plot the depths
        x = np.arange(len(methods))
        width = 0.35
        
        ax = plt.subplot(1, 1, 1)
        ax.bar(x - width/2, original_depths, width, label='Original Depth')
        ax.bar(x + width/2, optimized_depths, width, label='Optimized Depth')
        
        ax.set_ylabel('Circuit Depth')
        ax.set_title('Optimization Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_file)
        
        # Close the figure and free memory
        plt.close('all')
        gc.collect()
        
        print(f"Comparison plot saved to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error creating comparison plot: {e}")
        traceback.print_exc()
        
        # Force close all figures on error
        plt.close('all')
        gc.collect()
        
        return False

def save_benchmark_circuits_with_memory_management(benchmark_result, output_dir="benchmark_circuits",
                                                 max_qubits_for_image=15):
    """
    Save circuits from benchmark results with careful memory management.
    
    Args:
        benchmark_result: Dictionary with benchmark results containing circuits
        output_dir: Directory to save the circuits to
        max_qubits_for_image: Maximum number of qubits for image rendering
        
    Returns:
        dict: Dictionary with paths to saved circuits
    """
    if not QISKIT_AVAILABLE:
        print("Warning: Qiskit not available. Cannot save benchmark circuits.")
        return {}
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store saved circuit paths
    saved_paths = {}
    
    # Process each circuit individually to limit memory usage
    circuit_keys = []
    
    # First identify all circuit keys without loading them yet
    for key, value in benchmark_result.items():
        if isinstance(value, dict) and "circuit" in value:
            circuit_keys.append((key, None))  # Direct circuit
        else:
            # Look in nested dictionaries
            for subkey, subvalue in value.items() if isinstance(value, dict) else []:
                if isinstance(subvalue, dict) and "circuit" in subvalue:
                    circuit_keys.append((key, subkey))  # Nested circuit
    
    # Now process each circuit one at a time
    for key, subkey in circuit_keys:
        try:
            if subkey is None:
                # Direct circuit
                circuit = benchmark_result[key]["circuit"]
                circuit_name = f"{key}_circuit"
            else:
                # Nested circuit
                circuit = benchmark_result[key][subkey]["circuit"]
                circuit_name = f"{key}_{subkey}_circuit"
            
            # Save the circuit with memory optimization
            if circuit is None:
                print(f"Warning: Circuit for {circuit_name} is None")
                continue
                
            path = memory_optimized_save_circuit_image(
                circuit,
                circuit_name,
                output_dir=output_dir,
                max_qubits_for_image=max_qubits_for_image
            )
            
            if path:
                saved_paths[circuit_name] = path
            
            # Force garbage collection after each circuit
            del circuit
            gc.collect()
            
        except Exception as e:
            print(f"Error saving circuit {key}/{subkey}: {e}")
    
    print(f"Saved {len(saved_paths)} benchmark circuits to {output_dir}")
    return saved_paths

def optimized_visualize_optimization(circuit, report, filename='optimization_report'):
    """
    Create a comprehensive visualization of optimization results with memory management.
    
    Args:
        circuit: The optimized quantum circuit to visualize
        report: Optimization report containing metrics and results
        filename: Base filename for the visualization outputs
        
    Returns:
        dict: Dictionary with paths to saved files
    """
    if not QISKIT_AVAILABLE:
        print("Warning: Qiskit not available. Cannot visualize optimization.")
        return {}
        
    # Create output directory
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    # Step 1: Save the circuit - use memory optimized version
    try:
        circuit_path = memory_optimized_save_circuit_image(
            circuit,
            os.path.basename(filename) + "_circuit",
            output_dir=output_dir
        )
        if circuit_path:
            saved_files["circuit"] = circuit_path
        
        # Force garbage collection after saving circuit
        gc.collect()
    except Exception as e:
        print(f"Error saving circuit visualization: {e}")
    
    # Step 2: Save the optimization report metrics
    try:
        # Create a simplified report without large objects
        simplified_report = {}
        for key, value in report.items():
            # Skip non-serializable items and large objects
            if key not in ['circuit', 'original_circuit', 'replacements']:
                simplified_report[key] = value
        
        # Save the report as JSON
        import json
        report_path = f"{filename}_report.json"
        with open(report_path, 'w') as f:
            json.dump(simplified_report, f, indent=4)
        
        saved_files["report"] = report_path
        print(f"Optimization report saved to {report_path}")
        
        # Force garbage collection
        del simplified_report
        gc.collect()
    except Exception as e:
        print(f"Error saving optimization report: {e}")
    
    # Step 3: Create a simple bar chart showing optimization results
    if MATPLOTLIB_AVAILABLE:
        try:
            # First, close any existing figures
            plt.close('all')
            
            # Extract key metrics
            original_depth = report.get('original_depth', 0)
            optimized_depth = report.get('optimized_depth', 0)
            
            # Create a simple bar chart
            plt.figure(figsize=(6, 4))
            plt.bar(['Original', 'Optimized'], [original_depth, optimized_depth])
            plt.ylabel('Circuit Depth')
            plt.title('Optimization Results')
            
            # Add percentage reduction
            reduction = report.get('depth_reduction', 0)
            if isinstance(reduction, float):
                reduction_percent = reduction * 100
            elif original_depth > 0:
                reduction_percent = (original_depth - optimized_depth) / original_depth * 100
            else:
                reduction_percent = 0
                
            plt.text(1, optimized_depth / 2, f"{reduction_percent:.1f}% reduction",
                    ha='center', va='center', fontweight='bold')
            
            # Save the plot
            plot_path = f"{filename}_depth_reduction.png"
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            
            saved_files["plot"] = plot_path
            print(f"Depth reduction plot saved to {plot_path}")
            
            # Force garbage collection
            gc.collect()
        except Exception as e:
            print(f"Error creating optimization visualization: {e}")
            # Close any partially created figures
            plt.close('all')
    
    return saved_files
