"""
IO Utilities for Quantum Circuits

This module provides I/O utilities for quantum circuits, including:
- Loading and saving circuits in various formats (QASM, pickle, JSON)
- Saving circuit visualizations as images or text files
- Loading Toffoli networks from files
"""

import os
import pickle
import json
import time
import traceback
from datetime import datetime
import gc

# Try to import Qiskit
try:
    from qiskit import QuantumCircuit
    from qiskit.qasm2 import dumps, loads
    from qiskit.visualization import circuit_drawer
    import matplotlib.pyplot as plt
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


#################################
# Circuit Loading and Saving
#################################

def save_circuit_to_qasm(circuit, filename):
    """
    Save a quantum circuit to QASM format with compatibility for different Qiskit versions.
    
    Args:
        circuit: The quantum circuit to save
        filename: The output filename
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not QISKIT_AVAILABLE:
        print("Error: Qiskit is required to save QASM files")
        return False
        
    try:
        # First try the newer method (Qiskit 0.20+)
        try:
            # Get QASM string
            qasm_str = dumps(circuit)
            
            # Write to file
            with open(filename, 'w') as f:
                f.write(qasm_str)
            
            return True
        
        # If that fails, try the older method
        except ImportError:
            # Try older method (circuit.qasm())
            if hasattr(circuit, 'qasm'):
                circuit.qasm(filename=filename)
                return True
            else:
                # Last resort - manual conversion
                try:
                    from qiskit import qasm
                    qasm_str = qasm.dumps(circuit)
                    with open(filename, 'w') as f:
                        f.write(qasm_str)
                    return True
                except:
                    raise ImportError("No QASM export method available")
    
    except Exception as e:
        print(f"Error saving circuit to QASM format: {e}")
        return False

def load_circuit_from_qasm(filename):
    """
    Load a quantum circuit from a QASM file with compatibility for different Qiskit versions.
    
    Args:
        filename: The QASM file to load
        
    Returns:
        QuantumCircuit: The loaded circuit, or None if loading failed
    """
    if not QISKIT_AVAILABLE:
        print("Error: Qiskit is required to load QASM files")
        return None
        
    try:
        # Try newer method first
        try:
            with open(filename, 'r') as f:
                qasm_str = f.read()
            
            circuit = loads(qasm_str)
            return circuit
        
        # If that fails, try the older method
        except ImportError:
            # Try older method (QuantumCircuit.from_qasm_file)
            if hasattr(QuantumCircuit, 'from_qasm_file'):
                circuit = QuantumCircuit.from_qasm_file(filename)
                return circuit
            else:
                # Last resort - manual conversion
                try:
                    from qiskit import qasm
                    with open(filename, 'r') as f:
                        qasm_str = f.read()
                    circuit = qasm.loads(qasm_str)
                    return circuit
                except:
                    raise ImportError("No QASM import method available")
    
    except Exception as e:
        print(f"Error loading circuit from QASM file: {e}")
        return None

def save_circuit_safely(circuit, name, output_dir):
    """
    Save a quantum circuit to a file for later analysis.
    
    Args:
        circuit: Quantum circuit to save
        name: Base name for the file
        output_dir: Directory to save to
        
    Returns:
        str: Path to the saved file, or None if saving failed
    """
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
    if QISKIT_AVAILABLE:
        try:
            qasm_filename = f"{name}_{timestamp}.qasm"
            qasm_filepath = os.path.join(output_dir, qasm_filename)
            
            # Export to QASM
            qasm_str = dumps(circuit)
            
            with open(qasm_filepath, 'w') as f:
                f.write(qasm_str)
            
            print(f"Circuit saved to QASM file: {qasm_filepath}")
            return qasm_filepath
            
        except Exception as e:
            print(f"Warning: Failed to save as QASM: {e}")
            # Continue to backup method
    
    # Backup method: Save circuit statistics to a readable text file
    try:
        txt_filename = f"{name}_{timestamp}.txt"
        txt_filepath = os.path.join(output_dir, txt_filename)
        
        # Get basic circuit metrics
        depth = circuit.depth() if hasattr(circuit, 'depth') else 'Unknown'
        num_qubits = circuit.num_qubits if hasattr(circuit, 'num_qubits') else 'Unknown'
        num_gates = len(circuit.data) if hasattr(circuit, 'data') else 'Unknown'
        
        # Count operations if possible
        gate_counts = {}
        if hasattr(circuit, 'count_ops'):
            try:
                gate_counts = circuit.count_ops()
            except Exception:
                gate_counts = {"Unknown": "Error counting operations"}
        
        # Get detailed circuit data
        instructions = []
        if hasattr(circuit, 'data'):
            for inst in circuit.data:
                try:
                    gate_name = inst.operation.name
                    qubits = [f"q[{q._index if hasattr(q, '_index') else q.index}]" for q in inst.qubits]
                    qubits_str = ", ".join(qubits)
                    params = []
                    if hasattr(inst.operation, 'params'):
                        params = [str(p) for p in inst.operation.params]
                    params_str = ", ".join(params)
                    if params:
                        instructions.append(f"{gate_name}({qubits_str}, {params_str})")
                    else:
                        instructions.append(f"{gate_name}({qubits_str})")
                except Exception as inst_e:
                    instructions.append(f"Error parsing instruction: {inst_e}")
        
        # Write circuit statistics and details to file
        with open(txt_filepath, 'w') as f:
            f.write(f"Circuit Details for {name}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Saved at: {timestamp}\n")
            f.write(f"Number of qubits: {num_qubits}\n")
            f.write(f"Circuit depth: {depth}\n")
            f.write(f"Total gates: {num_gates}\n")
            
            f.write("\nGate counts:\n")
            for gate, count in sorted(gate_counts.items()):
                f.write(f"  {gate}: {count}\n")
            
            f.write("\nCircuit instructions:\n")
            for i, inst in enumerate(instructions):
                f.write(f"{i+1:4d}: {inst}\n")
        
        print(f"Circuit details saved to: {txt_filepath}")
        return txt_filepath
        
    except Exception as e:
        print(f"Error saving circuit details: {e}")
        traceback.print_exc()
        return None

def save_circuit_image(circuit, filename, output_dir="circuit_images", figsize=(12, 8),
                       use_text_mode=True, dpi=72, format='png', garbage_collect=True):
    """
    Save a circuit visualization as an image or text file with memory optimization.
    
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
        print("Error: Qiskit is required to save circuit images")
        return None
        
    if circuit is None:
        print(f"Warning: Cannot save circuit image for None circuit")
        return None
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if use_text_mode:
        # Generate a unique timestamp for the filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
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
        # Generate a unique timestamp for the filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
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
        print("Error: Qiskit is required to save circuit stats")
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

def save_benchmark_circuits(benchmark_result, output_dir="benchmark_circuits", save_as_text=True):
    """
    Save circuits from benchmark results.
    
    Args:
        benchmark_result: Dictionary with benchmark results containing circuits
        output_dir: Directory to save the circuits to
        save_as_text: Whether to save as text (memory efficient) or images
        
    Returns:
        dict: Dictionary with paths to saved circuits
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store saved circuit paths
    saved_paths = {}
    
    # Try to save logical circuit
    if "logical" in benchmark_result and "circuit" in benchmark_result["logical"]:
        try:
            logical_path = save_circuit_image(
                benchmark_result["logical"]["circuit"],
                "logical_circuit",
                output_dir=output_dir,
                use_text_mode=save_as_text
            )
            saved_paths["logical"] = logical_path
        except Exception as e:
            print(f"Error saving logical circuit: {e}")
    
    # Try to save original circuit
    if "original" in benchmark_result and "circuit" in benchmark_result["original"]:
        try:
            original_path = save_circuit_image(
                benchmark_result["original"]["circuit"],
                "original_circuit",
                output_dir=output_dir,
                use_text_mode=save_as_text
            )
            saved_paths["original"] = original_path
        except Exception as e:
            print(f"Error saving original circuit: {e}")
    
    # Try to save Toffoli Optimizer circuit
    if "toffoli_optimizer" in benchmark_result:
        if "mapped" in benchmark_result["toffoli_optimizer"] and "circuit" in benchmark_result["toffoli_optimizer"]["mapped"]:
            try:
                toffoli_path = save_circuit_image(
                    benchmark_result["toffoli_optimizer"]["mapped"]["circuit"],
                    "toffoli_optimizer_circuit",
                    output_dir=output_dir,
                    use_text_mode=save_as_text
                )
                saved_paths["toffoli_optimizer"] = toffoli_path
            except Exception as e:
                print(f"Error saving Toffoli Optimizer circuit: {e}")
    
    # Try to save Qiskit transpiler circuit
    if "qiskit_transpiler" in benchmark_result and "circuit" in benchmark_result["qiskit_transpiler"]:
        try:
            qiskit_path = save_circuit_image(
                benchmark_result["qiskit_transpiler"]["circuit"],
                "qiskit_transpiler_circuit",
                output_dir=output_dir,
                use_text_mode=save_as_text
            )
            saved_paths["qiskit_transpiler"] = qiskit_path
        except Exception as e:
            print(f"Error saving Qiskit transpiler circuit: {e}")
    
    return saved_paths


#################################
# Toffoli Network Loading
#################################

class ToffoliNetworkLoader:
    """Class for loading and processing saved Toffoli networks"""
    
    @staticmethod
    def load_toffoli_network(filename, debug=False):
        """
        Load a Toffoli network from file.
        Filters to only include ccx (Toffoli) and x gates.
        
        Args:
            filename (str): The name of the file to load (without extension)
            debug (bool): Whether to print debug information
            
        Returns:
            tuple: (toffoli_gates, output_qubits, input_qubits, num_qubits) or None if loading failed
        """
        # First, try to load from JSON which is most reliable for Toffoli networks
        toffoli_gates, num_qubits = ToffoliNetworkLoader.load_json_toffoli_network(filename)
        
        # If JSON loading failed, try circuit extraction
        if not toffoli_gates:
            circuit = None
            
            # Try pickle first
            try:
                with open(f"{filename}.pickle", "rb") as f:
                    circuit = pickle.load(f)
                print(f"Circuit loaded from {filename}.pickle")
                num_qubits = circuit.num_qubits
                
                # Extract only ccx/mcx and x gates from the circuit
                toffoli_gates = ToffoliNetworkLoader.extract_toffoli_gates(circuit, debug)
                
            except Exception as e:
                print(f"Could not load circuit from {filename}.pickle: {e}")
                
                # Try QASM file
                try:
                    circuit = load_circuit_from_qasm(f"{filename}.qasm")
                    print(f"Circuit loaded from {filename}.qasm")
                    num_qubits = circuit.num_qubits
                    
                    # Extract only ccx/mcx and x gates from the circuit
                    toffoli_gates = ToffoliNetworkLoader.extract_toffoli_gates(circuit, debug)
                    
                except Exception as e:
                    print(f"Could not load circuit from {filename}.qasm: {e}")
        
        # If we still couldn't load any Toffoli gates, return None
        if not toffoli_gates or num_qubits is None:
            print("Failed to load Toffoli network: No ccx/mcx or x gates found")
            return None
        
        # Ensure we have at least 8 qubits for the benchmark system
        min_required_qubits = 8
        original_num_qubits = num_qubits
        
        if num_qubits < min_required_qubits and debug:
            print(f"Note: Circuit has {num_qubits} qubits, but benchmark system requires at least {min_required_qubits}")
            print(f"Using trivial mapping from {num_qubits}-qubit circuit to {min_required_qubits}-qubit register")
            num_qubits = min_required_qubits
        
        # Validate and fix gate indices to ensure they don't reference qubits outside the range
        valid_toffoli_gates = []
        for controls, target in toffoli_gates:
            # First check if target is within valid range
            if target < num_qubits:
                # Then check each control qubit
                valid_controls = [c for c in controls if c < num_qubits]
                
                # Only include the gate if we have valid controls and target
                if len(valid_controls) > 0 or not controls:  # Allow X gates with empty controls
                    valid_toffoli_gates.append((valid_controls, target))
                else:
                    print(f"Warning: Skipping gate with invalid controls {controls} -> {target}")
            else:
                print(f"Warning: Skipping gate with invalid target {controls} -> {target}")
        
        if len(valid_toffoli_gates) == 0:
            print("Error: No valid Toffoli gates after validation. Check if gate indices exceed qubit count.")
            return None
            
        if len(valid_toffoli_gates) < len(toffoli_gates):
            print(f"Warning: Filtered out {len(toffoli_gates) - len(valid_toffoli_gates)} invalid gates")
            print(f"Proceeding with {len(valid_toffoli_gates)} valid gates")
            toffoli_gates = valid_toffoli_gates
        
        # Try to load metadata for input/output qubits
        input_qubits = []
        output_qubits = []
        try:
            with open(f"{filename}_metadata.json", "r") as f:
                metadata = json.load(f)
                
                # If metadata has explicit input/output qubits, use them
                if "input_qubits" in metadata:
                    input_qubits = metadata["input_qubits"]
                if "output_qubits" in metadata:
                    output_qubits = metadata["output_qubits"]
        except Exception as e:
            print(f"Could not load metadata: {e}")
        
        # If we still don't have input/output qubits, infer them from the circuit
        if not input_qubits or not output_qubits:
            print("Inferring input/output qubits from Toffoli network structure...")
            output_qubits, input_qubits = ToffoliNetworkLoader.infer_io_qubits(toffoli_gates, num_qubits)
        if debug:
            print(f"Loaded Toffoli network with {len(toffoli_gates)} gates on {original_num_qubits} qubits")
            print(f"Input qubits: {input_qubits}")
            print(f"Output qubits: {output_qubits}")
        
        # Make 100% sure we have some input and output qubits
        if not input_qubits:
            input_qubits = list(range(min(3, num_qubits)))
            print(f"Warning: No input qubits identified. Using default: {input_qubits}")
            
        if not output_qubits:
            # Use the last few qubits as outputs
            output_qubits = list(range(max(0, num_qubits-3), num_qubits))
            print(f"Warning: No output qubits identified. Using default: {output_qubits}")
        
        # Ensure inputs and outputs are within bounds of the expanded register
        input_qubits = [q for q in input_qubits if q < num_qubits]
        output_qubits = [q for q in output_qubits if q < num_qubits]
        
        # Safety check to make sure we have at least some inputs and outputs
        if not input_qubits:
            input_qubits = [0, 1]
            print(f"Warning: No valid input qubits after filtering. Using default: {input_qubits}")
        
        if not output_qubits:
            output_qubits = [2, 3]
            print(f"Warning: No valid output qubits after filtering. Using default: {output_qubits}")
        
        return toffoli_gates, output_qubits, input_qubits, num_qubits
    
    @staticmethod
    def load_json_toffoli_network(filename):
        """
        Specialized function to extract Toffoli/X gates directly from the JSON representation,
        which is more reliable than extracting from circuit objects.
        
        Args:
            filename (str): The filename without extension
            
        Returns:
            tuple: (toffoli_gates, num_qubits) or (None, None) if loading failed
        """
        try:
            with open(f"{filename}.json", "r") as f:
                circuit_data = json.load(f)
            
            toffoli_gates = []
            num_qubits = circuit_data["num_qubits"]
            
            # Extract only ccx/mcx and x gates
            for instruction in circuit_data.get("instructions", []):
                name = instruction.get("name", "")
                
                # For mcx gates (multi-controlled X, including Toffoli)
                if name == "mcx":
                    if "control_qubits" in instruction and "target_qubit" in instruction:
                        control_qubits = instruction["control_qubits"]
                        target_qubit = instruction["target_qubit"]
                        toffoli_gates.append((control_qubits, target_qubit))
                    elif "qubits" in instruction and len(instruction["qubits"]) >= 2:
                        # Alternative format: last qubit is target, others are controls
                        qubits = instruction["qubits"]
                        control_qubits = qubits[:-1]
                        target_qubit = qubits[-1]
                        toffoli_gates.append((control_qubits, target_qubit))
                
                # For ccx gates (Toffoli with 2 controls)
                elif name == "ccx":
                    if "qubits" in instruction and len(instruction["qubits"]) >= 3:
                        control1 = instruction["qubits"][0]
                        control2 = instruction["qubits"][1]
                        target = instruction["qubits"][2]
                        toffoli_gates.append(([control1, control2], target))
                
                # For x gates (single-qubit NOT, represented as Toffoli with no controls)
                elif name == "x":
                    if "qubits" in instruction and len(instruction["qubits"]) >= 1:
                        target = instruction["qubits"][0]
                        toffoli_gates.append(([], target))  # X gate as a Toffoli with no controls
            
            if toffoli_gates:
                print(f"Extracted {len(toffoli_gates)} Toffoli/X gates from JSON file")
                return toffoli_gates, num_qubits
            else:
                print("No Toffoli/X gates found in JSON file")
                return None, None
                
        except Exception as e:
            print(f"Error loading Toffoli network from JSON: {e}")
            return None, None

    @staticmethod
    def extract_toffoli_gates(circuit, debug=False):
        """
        Extract Toffoli gates from a quantum circuit.
        Only extracts ccx (Toffoli) and x gates, ignoring all other gate types.
        
        Args:
            circuit: A quantum circuit
            debug: Whether to print debug information
        
        Returns:
            list: List of (control_qubits, target_qubit) tuples representing Toffoli gates
        """
        if not QISKIT_AVAILABLE:
            print("Error: Qiskit is required to extract Toffoli gates")
            return []
            
        toffoli_gates = []
        
        # Get register information to map qubits to indices
        qubit_indices = {}
        for i, qubit in enumerate(circuit.qubits):
            qubit_indices[qubit] = i
        
        # Debug the circuit structure
        if debug: print(f"DEBUG - Circuit has {circuit.num_qubits} qubits")
        
        for instruction in circuit.data:
            try:
                # Get the operation name
                gate_name = instruction.operation.name
                
                # Skip any gates that are not ccx, mcx, or x
                if gate_name not in ["ccx", "mcx", "x"]:
                    continue
                
                # Get qubit indices safely, handling different Qiskit versions
                instruction_qubits = instruction.qubits
                mapped_indices = []
                
                for qubit in instruction_qubits:
                    # Use the mapping we created
                    if qubit in qubit_indices:
                        mapped_indices.append(qubit_indices[qubit])
                    else:
                        # Fallback methods if the qubit is not in our mapping
                        try:
                            # Try different attributes
                            if hasattr(qubit, 'index'):
                                mapped_indices.append(qubit.index)
                            elif hasattr(qubit, '_index'):
                                mapped_indices.append(qubit._index)
                            else:
                                # Last resort: try to get index from string representation
                                qubit_str = str(qubit)
                                if 'q[' in qubit_str:
                                    idx_str = qubit_str.split('q[')[1].split(']')[0]
                                    mapped_indices.append(int(idx_str))
                                else:
                                    raise ValueError(f"Cannot determine index for qubit: {qubit}")
                        except Exception as e:
                            raise ValueError(f"Failed to get index for qubit {qubit}: {e}")
                
                # Process mcx (Toffoli) gates
                if gate_name == "mcx":
                    # In a Toffoli gate, the last qubit is the target
                    if len(mapped_indices) < 2:  # Need at least 1 control and 1 target
                        print(f"Warning: mcx gate with insufficient qubits: {mapped_indices}")
                        continue
                        
                    control_qubits = mapped_indices[:-1]
                    target_qubit = mapped_indices[-1]
                    
                    # Debug output for this gate
                    if debug: print(f"DEBUG - Found mcx gate: controls={control_qubits}, target={target_qubit}")
                    
                    # Add to Toffoli gates list
                    toffoli_gates.append((control_qubits, target_qubit))
                
                # Process ccx gates (special case of mcx with 2 controls)
                elif gate_name == "ccx":
                    if len(mapped_indices) < 3:  # Need exactly 2 controls and 1 target
                        print(f"Warning: ccx gate with insufficient qubits: {mapped_indices}")
                        continue
                        
                    control1 = mapped_indices[0]
                    control2 = mapped_indices[1]
                    target = mapped_indices[2]
                    
                    # Debug output for this gate
                    if debug: print(f"DEBUG - Found ccx gate: controls=[{control1},{control2}], target={target}")
                    
                    # Add to Toffoli gates list
                    toffoli_gates.append(([control1, control2], target))
                
                # Process x gates (can be represented as Toffoli gates with no controls)
                elif gate_name == "x":
                    if not mapped_indices:  # Need at least 1 target
                        print(f"Warning: x gate with no target qubit")
                        continue
                        
                    target = mapped_indices[0]
                    
                    # Debug output for this gate
                    if debug: print(f"DEBUG - Found x gate: target={target}")
                    
                    # Add as a Toffoli gate with empty control list
                    toffoli_gates.append(([], target))
            except Exception as e:
                print(f"Warning: Could not process instruction: {e}")
        
        print(f"Extracted {len(toffoli_gates)} Toffoli/X gates, filtering out all other gate types")
        
        # Print the first few extracted gates for debugging
        for i, (controls, target) in enumerate(toffoli_gates[:5]):
            if debug: print(f"  Gate {i+1}: Controls {controls} -> Target {target}")
            
        # Do a final validation to make sure the gate format is correct
        validated_gates = []
        for controls, target in toffoli_gates:
            # Ensure target is an integer and controls is a list
            if isinstance(target, int) and isinstance(controls, list):
                validated_gates.append((controls, target))
            else:
                print(f"Warning: Invalid gate format - Controls: {controls}, Target: {target}")
        
        if len(validated_gates) < len(toffoli_gates):
            print(f"Warning: {len(toffoli_gates) - len(validated_gates)} gates had invalid format")
            toffoli_gates = validated_gates
            
        return toffoli_gates
    
    @staticmethod
    def infer_io_qubits(toffoli_gates, num_qubits):
        """
        Infer the input and output qubits from the Toffoli gates.
        
        Args:
            toffoli_gates: List of (control_qubits, target_qubit) tuples
            num_qubits: Total number of qubits
            
        Returns:
            tuple: (output_qubits, input_qubits)
        """
        try:
            import numpy as np
            # Initialize counters for how often each qubit appears as control or target
            control_counts = np.zeros(num_qubits)
            target_counts = np.zeros(num_qubits)
            
            # Count occurrences
            for controls, target in toffoli_gates:
                for control in controls:
                    control_counts[control] += 1
                target_counts[target] += 1
            
            # Infer input qubits as those that appear more as controls than targets
            # or appear as controls in the first gates
            input_candidate_set = set()
            
            # Add qubits that appear more as controls
            for i in range(num_qubits):
                if control_counts[i] > target_counts[i]:
                    input_candidate_set.add(i)
            
            # Add control qubits from the first few gates if we don't have enough inputs
            if len(input_candidate_set) < num_qubits // 3:
                for controls, _ in toffoli_gates[:min(5, len(toffoli_gates))]:
                    for control in controls:
                        input_candidate_set.add(control)
            
            # Ensure we have at least some inputs (at least 2 or 25% of qubits)
            min_inputs = max(2, num_qubits // 4)
            if len(input_candidate_set) < min_inputs:
                # Add the first few qubits as inputs
                for i in range(min(min_inputs, num_qubits)):
                    input_candidate_set.add(i)
            
            input_qubits = sorted(list(input_candidate_set))
            
            # Infer output qubits as those that appear more as targets than controls
            # or appear as targets in the last gates
            output_candidate_set = set()
            
            # Add qubits that appear more as targets
            for i in range(num_qubits):
                if target_counts[i] > control_counts[i]:
                    output_candidate_set.add(i)
            
            # Add target qubits from the last few gates if we don't have enough outputs
            if len(output_candidate_set) < num_qubits // 3:
                for _, target in reversed(toffoli_gates[-min(5, len(toffoli_gates)):]):
                    output_candidate_set.add(target)
            
            # Ensure we have at least some outputs (at least 2 or 25% of qubits)
            min_outputs = max(2, num_qubits // 4)
            if len(output_candidate_set) < min_outputs:
                # Add the last few qubits as outputs
                for i in range(num_qubits - min(min_outputs, num_qubits), num_qubits):
                    output_candidate_set.add(i)
            
            output_qubits = sorted(list(output_candidate_set))
            
            return output_qubits, input_qubits
        except ImportError:
            # If numpy is not available, use a simpler approach
            # Just use the first few qubits as inputs and last few as outputs
            min_qubits = max(2, num_qubits // 4)
            input_qubits = list(range(min_qubits))
            output_qubits = list(range(num_qubits - min_qubits, num_qubits))
            return output_qubits, input_qubits

def define_loaded_toffoli_network(filename="toffoli_network_circuit", debug=False):
    """
    Define a Toffoli network loaded from a file.
    This function is compatible with the format expected by the benchmark system.
    
    Args:
        filename: The name of the file to load (without extension)
        debug: Whether to print debug information
        
    Returns:
        tuple: (toffoli_gates, output_qubits, input_qubits) or a default network if loading fails
    """
    # Try to load the network
    result = ToffoliNetworkLoader.load_toffoli_network(filename, debug)
    
    if result:
        toffoli_gates, output_qubits, input_qubits, num_qubits = result
        
        # Validate the Toffoli gates
        has_errors = False
        for i, (controls, target) in enumerate(toffoli_gates):
            # Check if control qubits and target are within range
            if target >= num_qubits:
                if debug: print(f"Error: Gate {i} has target {target} outside qubit range {num_qubits}")
                has_errors = True
                break
            if isinstance(controls, list) and any(c >= num_qubits for c in controls):
                if debug: print(f"Error: Gate {i} has controls {controls} outside qubit range {num_qubits}")
                has_errors = True
                break
            # Check if there are too many control qubits for the circuit size
            if isinstance(controls, list) and len(controls) > num_qubits - 1:
                if debug: print(f"Error: Gate {i} has too many controls ({len(controls)}) for circuit with {num_qubits} qubits")
                has_errors = True
                break
        
        # If there are no validation errors, return the original loaded network
        if not has_errors:
            print(f"Using loaded Toffoli network with {len(toffoli_gates)} gates on {num_qubits} qubits")
            return toffoli_gates, output_qubits, input_qubits
        
        # Otherwise fall back to simplified network
        print("Using a simplified Toffoli network to ensure compatibility due to validation errors")
        
    else:
        print("WARNING: Could not load network, using a simple default network instead.")
    
    # Simple default network: two Toffoli gates on 8 qubits
    simple_toffoli_gates = [
        ([0, 1], 2),  # First Toffoli with controls 0,1 and target 2
        ([2, 3], 4)   # Second Toffoli with controls 2,3 and target 4
    ]
    simple_output_qubits = [2, 4]
    simple_input_qubits = [0, 1, 3]
    
    return simple_toffoli_gates, simple_output_qubits, simple_input_qubits

# Export needed functions
load_toffoli_network = ToffoliNetworkLoader.load_toffoli_network
