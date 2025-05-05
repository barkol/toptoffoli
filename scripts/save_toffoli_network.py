#!/usr/bin/env python3
"""
Save Toffoli Network Script

This script saves Toffoli networks from ShiftExperiment_FlowControl as reusable formats.
Compatible with Qiskit 2.0 and qiskit-aer 0.17.0
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import os
import sys
#https://github.com/JedrekSt/Quantum_Synchronization_Protocol/blob/main/Computation_tools/SynchronizingCircuit.py
from SynchronizingCircuit import ShiftExperiment_FlowControl

# Check qiskit versions
import qiskit
print(f"Qiskit version: {qiskit.__version__}")

# Try different import paths for AerSimulator
aer_simulator_available = False
try:
    # Modern approach (qiskit-aer 0.17.0+)
    from qiskit_aer import AerSimulator
    aer_simulator_available = True
    print("Using qiskit_aer.AerSimulator")
except ImportError:
    try:
        # Alternative import in some versions
        from qiskit.providers.aer import AerSimulator
        aer_simulator_available = True
        print("Using qiskit.providers.aer.AerSimulator")
    except ImportError:
        print("AerSimulator not available - simulation features will be disabled")

def save_circuit_as_toffoli_network(filename, qubits=2, ancillas=4, controlling_anc=3):
    """
    Generate a circuit using ShiftExperiment_FlowControl and save it as a Toffoli network.
    
    Args:
        filename (str): The name of the file to save the circuit to (without extension)
        qubits (int): Number of qubits
        ancillas (int): Number of ancillas
        controlling_anc (int): Controlling ancilla index
    
    Returns:
        QuantumCircuit: The generated circuit
    """
    print(f"Creating ShiftExperiment_FlowControl with qubits={qubits}, ancillas={ancillas}, controlling_anc={controlling_anc}")
    
    # Create the ShiftExperiment_FlowControl object
    experiment = ShiftExperiment_FlowControl(qubits=qubits, ancillas=ancillas, controlling_anc=controlling_anc)
    
    # Get the circuit
    circuit = experiment.Part2
    print(f"Circuit created with {circuit.num_qubits} qubits and {circuit.num_clbits} classical bits")
    
    # Save the circuit in multiple formats for compatibility
    
    # 1. Save as a Qiskit pickle file (most comprehensive)
    with open(f"{filename}.pickle", "wb") as f:
        pickle.dump(circuit, f)
    print(f"Circuit saved as {filename}.pickle")
    
    # 2. Try to save QASM representation
    try:
        # Try modern Qiskit approach
        try:
            from qiskit.qasm2 import export_qasm
            qasm_str = export_qasm(circuit)
            with open(f"{filename}.qasm", "w") as f:
                f.write(qasm_str)
            print(f"Circuit saved as {filename}.qasm (using export_qasm)")
        except (ImportError, AttributeError):
            # Try legacy approach
            if hasattr(circuit, 'qasm'):
                qasm_str = circuit.qasm()
                with open(f"{filename}.qasm", "w") as f:
                    f.write(qasm_str)
                print(f"Circuit saved as {filename}.qasm (using circuit.qasm())")
            else:
                print("QASM export not available in this Qiskit version")
    except Exception as e:
        print(f"Could not save QASM representation: {e}")
    
    # 3. Try to save as an image if matplotlib is available
    try:
        plt.figure(figsize=(20, 10))
        circuit.draw('mpl', filename=f"{filename}.png")
        plt.close()
        print(f"Circuit image saved as {filename}.png")
    except Exception as e:
        print(f"Warning: Could not save circuit image: {e}")
    
    # 4. Extract and save the circuit data in JSON format
    # This includes all gates, especially focusing on Toffoli (mcx) gates
    circuit_data = {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "instructions": []
    }
    
    toffoli_count = 0
    
    for instruction in circuit.data:
        try:
            # Get the operation name and qubit indices using Qiskit 2.0 attributes
            gate_name = instruction.operation.name
            qubit_indices = [q._index for q in instruction.qubits]
            clbit_indices = [c._index for c in instruction.clbits] if instruction.clbits else []
            
            gate_data = {
                "name": gate_name,
                "qubits": qubit_indices,
                "clbits": clbit_indices,
            }
            
            # For mcx gates (Toffoli and multi-controlled X), store control and target qubits
            if gate_name == 'mcx':
                # In a Toffoli gate, the last qubit is the target
                gate_data["control_qubits"] = qubit_indices[:-1]
                gate_data["target_qubit"] = qubit_indices[-1]
                toffoli_count += 1
            
            circuit_data["instructions"].append(gate_data)
        except Exception as e:
            print(f"Warning: Could not process instruction {instruction}: {e}")
    
    with open(f"{filename}.json", "w") as f:
        json.dump(circuit_data, f, indent=2)
    print(f"Circuit details saved as {filename}.json")
    
    print(f"Total number of gates: {len(circuit.data)}")
    print(f"Number of toffoli (mcx) gates: {toffoli_count}")
    
    # Create a minimal metadata file
    metadata = {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "gate_count": len(circuit.data),
        "toffoli_count": toffoli_count,
        "creation_parameters": {
            "qubits": qubits,
            "ancillas": ancillas,
            "controlling_anc": controlling_anc
        }
    }
    
    with open(f"{filename}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Circuit metadata saved as {filename}_metadata.json")
    
    return circuit

def extract_toffoli_gates_from_circuit(circuit):
    """
    Extract Toffoli gates from a quantum circuit.
    
    Args:
        circuit: A quantum circuit
    
    Returns:
        list: List of (control_qubits, target_qubit) tuples representing Toffoli gates
    """
    toffoli_gates = []
    
    for instruction in circuit.data:
        try:
            # Get the operation name and qubit indices
            gate_name = instruction.operation.name
            qubit_indices = [q._index for q in instruction.qubits]
            
            # Process only mcx (Toffoli) gates
            if gate_name == "mcx":
                # In a Toffoli gate, the last qubit is the target
                control_qubits = qubit_indices[:-1]
                target_qubit = qubit_indices[-1]
                
                # Add to Toffoli gates list
                toffoli_gates.append((control_qubits, target_qubit))
        except Exception as e:
            print(f"Warning: Could not process instruction: {e}")
    
    return toffoli_gates

def load_toffoli_network(filename):
    """
    Load a saved Toffoli network from a file.
    
    Args:
        filename (str): The name of the file to load the circuit from (without extension)
    
    Returns:
        tuple: (circuit, toffoli_gates) where:
               - circuit is the loaded QuantumCircuit (or None if loading failed)
               - toffoli_gates is a list of (control_qubits, target_qubit) tuples
    """
    # Try loading from pickle file first (most comprehensive)
    try:
        with open(f"{filename}.pickle", "rb") as f:
            circuit = pickle.load(f)
        print(f"Circuit loaded from {filename}.pickle")
        
        # Extract Toffoli gates
        toffoli_gates = extract_toffoli_gates_from_circuit(circuit)
        print(f"Extracted {len(toffoli_gates)} Toffoli gates from circuit")
        
        return circuit, toffoli_gates
    except Exception as e:
        print(f"Could not load circuit from {filename}.pickle: {e}")
    
    # Try loading from QASM file
    try:
        circuit = QuantumCircuit.from_qasm_file(f"{filename}.qasm")
        print(f"Circuit loaded from {filename}.qasm")
        
        # Extract Toffoli gates
        toffoli_gates = extract_toffoli_gates_from_circuit(circuit)
        print(f"Extracted {len(toffoli_gates)} Toffoli gates from circuit")
        
        return circuit, toffoli_gates
    except Exception as e:
        print(f"Could not load circuit from {filename}.qasm: {e}")
    
    # Try loading from JSON file
    try:
        with open(f"{filename}.json", "r") as f:
            circuit_data = json.load(f)
        
        # Create a new circuit
        circuit = QuantumCircuit(circuit_data["num_qubits"], circuit_data["num_clbits"])
        
        # Add instructions from the JSON data
        toffoli_gates = []
        
        for instruction in circuit_data["instructions"]:
            if instruction["name"] == "measure":
                qubit = instruction["qubits"][0]
                clbit = instruction["clbits"][0]
                circuit.measure(qubit, clbit)
            elif instruction["name"] == "barrier":
                qubits = instruction["qubits"]
                circuit.barrier(qubits)
            elif instruction["name"] == "mcx":
                control_qubits = instruction["control_qubits"]
                target_qubit = instruction["target_qubit"]
                circuit.mcx(control_qubits, target_qubit)
                toffoli_gates.append((control_qubits, target_qubit))
            elif instruction["name"] == "x":
                qubit = instruction["qubits"][0]
                circuit.x(qubit)
        
        print(f"Circuit loaded from {filename}.json")
        print(f"Extracted {len(toffoli_gates)} Toffoli gates from JSON")
        
        return circuit, toffoli_gates
    except Exception as e:
        print(f"Could not load circuit from {filename}.json: {e}")
    
    # If all loading methods fail
    return None, []

def test_simulation(circuit):
    """
    Test the circuit by running a simulation if AerSimulator is available
    
    Args:
        circuit: Quantum circuit to test
        
    Returns:
        dict: Simulation counts or None if simulation not available
    """
    if not aer_simulator_available:
        print("Simulation test skipped - AerSimulator not available")
        return None
    
    try:
        # Create a simple test with initial state
        initial_state = [0] * circuit.num_qubits
        initial_state[0] = 1  # Set first qubit to |1⟩
        
        # Create a new circuit with initialization
        test_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        for i, val in enumerate(initial_state):
            if val == 1:
                test_circuit.x(i)
        
        # Compose with the loaded circuit
        test_circuit = test_circuit.compose(circuit)
        
        # Run the circuit using AerSimulator
        simulator = AerSimulator()
        compiled_circuit = transpile(test_circuit, simulator)
        job = simulator.run(compiled_circuit)
        result = job.result()
        counts = result.get_counts()
        
        print("Simulation completed successfully")
        print(f"Results: {counts}")
        return counts
    except Exception as e:
        print(f"Simulation test failed: {e}")
        return None

# Example usage
if __name__ == "__main__":
    try:
        print("=" * 50)
        print("TOFFOLI NETWORK SAVE AND LOAD UTILITY")
        print("=" * 50)
        print(f"Python version: {sys.version}")
        
        # Save the circuit
        circuit = save_circuit_as_toffoli_network("toffoli_network_circuit", qubits=2, ancillas=4, controlling_anc=3)
        
        print("\n" + "=" * 50)
        print("LOADING SAVED CIRCUIT")
        print("=" * 50)
        
        # Load the circuit
        loaded_circuit, toffoli_gates = load_toffoli_network("toffoli_network_circuit")
        
        if loaded_circuit:
            print("\nLoaded circuit summary:")
            print(f"- Number of qubits: {loaded_circuit.num_qubits}")
            print(f"- Number of classical bits: {loaded_circuit.num_clbits}")
            print(f"- Number of gates: {len(loaded_circuit.data)}")
            print(f"- Number of Toffoli gates: {len(toffoli_gates)}")
            
            if toffoli_gates:
                print("\nSample of Toffoli gates:")
                for i, (controls, target) in enumerate(toffoli_gates[:3]):
                    print(f"  Gate {i+1}: Controls {controls}, Target {target}")
                if len(toffoli_gates) > 3:
                    print(f"  ... and {len(toffoli_gates)-3} more")
            
            # Test simulation if available
            print("\n" + "=" * 50)
            print("RUNNING SIMULATION TEST")
            print("=" * 50)
            test_simulation(loaded_circuit)
        else:
            print("\nFailed to load the circuit.")
            
        print("\n" + "=" * 50)
        print("PROCESS COMPLETED")
        print("=" * 50)
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
