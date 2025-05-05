#!/usr/bin/env python3
"""
Circuit Extractor - Utility to extract Toffoli gates from quantum circuits
with Qiskit 2.0 compatibility to fix the `qubit.index` vs `qubit._index` issue
"""

import sys
import json
from qiskit import QuantumCircuit

def extract_toffoli_gates(circuit):
    """
    Extract Toffoli gates from a quantum circuit with proper handling of qubit indices
    for Qiskit 2.0 compatibility.
    
    Args:
        circuit: Qiskit QuantumCircuit
        
    Returns:
        tuple: (toffoli_gates, input_qubits, output_qubits)
    """
    toffoli_gates = []
    input_qubits = set()
    output_qubits = set()
    
    for instruction in circuit.data:
        try:
            # Get the operation name
            gate_name = instruction.operation.name
            
            # Skip any gates that are not ccx or mcx
            if gate_name not in ["ccx", "mcx"]:
                continue
            
            # Get qubit indices - using _index for Qiskit 2.0
            qubit_indices = []
            for q in instruction.qubits:
                try:
                    # For Qiskit 2.0
                    if hasattr(q, '_index'):
                        qubit_indices.append(q._index)
                    # Fallback for older versions
                    elif hasattr(q, 'index'):
                        qubit_indices.append(q.index)
                    else:
                        print(f"Cannot find index attribute on qubit")
                        continue
                except Exception as e:
                    print(f"Error getting qubit index: {e}")
                    continue
            
            # For CCX gates (Toffoli with 2 controls)
            if gate_name == "ccx" and len(qubit_indices) >= 3:
                control1, control2, target = qubit_indices[0], qubit_indices[1], qubit_indices[2]
                toffoli_gates.append(([control1, control2], target))
                
                # Add control qubits to input_qubits
                input_qubits.add(control1)
                input_qubits.add(control2)
                
                # Add target to output_qubits
                output_qubits.add(target)
            
            # For MCX gates
            elif gate_name == "mcx" and len(qubit_indices) >= 2:
                # Last qubit is target, others are controls
                controls = qubit_indices[:-1]
                target = qubit_indices[-1]
                toffoli_gates.append((controls, target))
                
                # Add control qubits to input_qubits
                for control in controls:
                    input_qubits.add(control)
                
                # Add target to output_qubits
                output_qubits.add(target)
        
        except Exception as e:
            print(f"Error processing instruction: {e}")
            continue
    
    return toffoli_gates, sorted(list(input_qubits)), sorted(list(output_qubits))

def extract_circuit_metadata(circuit):
    """
    Extract metadata from a quantum circuit.
    
    Args:
        circuit: Qiskit QuantumCircuit
        
    Returns:
        dict: Circuit metadata
    """
    # Extract Toffoli gates
    toffoli_gates, input_qubits, output_qubits = extract_toffoli_gates(circuit)
    
    # Count gates by type
    gate_counts = {}
    for instruction in circuit.data:
        try:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        except Exception as e:
            print(f"Error counting gate: {e}")
    
    # Create metadata
    metadata = {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "gate_count": len(circuit.data),
        "toffoli_count": len(toffoli_gates),
        "gate_counts": gate_counts,
        "input_qubits": input_qubits,
        "output_qubits": output_qubits,
        "depth": circuit.depth()
    }
    
    return metadata

def save_circuit_json(circuit, filename):
    """
    Save a circuit to a JSON file with proper handling of qubit indices.
    
    Args:
        circuit: Qiskit QuantumCircuit
        filename: Output filename
    """
    # Extract Toffoli and other gates
    instructions = []
    
    for inst in circuit.data:
        try:
            gate_name = inst.operation.name
            
            # Get qubit indices
            qubits = []
            for q in inst.qubits:
                try:
                    if hasattr(q, '_index'):
                        qubits.append(q._index)
                    elif hasattr(q, 'index'):
                        qubits.append(q.index)
                except:
                    continue
            
            # Create instruction data
            instruction_data = {
                "name": gate_name,
                "qubits": qubits
            }
            
            # Add extra information for specific gate types
            if gate_name == "ccx":
                instruction_data["control_qubits"] = qubits[:2]
                instruction_data["target_qubit"] = qubits[2]
            elif gate_name == "mcx":
                instruction_data["control_qubits"] = qubits[:-1]
                instruction_data["target_qubit"] = qubits[-1]
            elif gate_name == "cx":
                instruction_data["control_qubit"] = qubits[0]
                instruction_data["target_qubit"] = qubits[1]
            
            instructions.append(instruction_data)
        except Exception as e:
            print(f"Error extracting instruction: {e}")
    
    # Create circuit data
    circuit_data = {
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "instructions": instructions
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(circuit_data, f, indent=2)

def create_circuit_from_json(filename):
    """
    Create a quantum circuit from a JSON file.
    
    Args:
        filename: Input JSON filename
        
    Returns:
        QuantumCircuit: Created circuit
    """
    # Load circuit data
    with open(filename, 'r') as f:
        circuit_data = json.load(f)
    
    # Create circuit
    circuit = QuantumCircuit(circuit_data["num_qubits"], circuit_data.get("num_clbits", 0))
    
    # Add instructions
    for inst in circuit_data["instructions"]:
        gate_name = inst["name"]
        qubits = inst["qubits"]
        
        if gate_name == "h":
            circuit.h(qubits[0])
        elif gate_name == "x":
            circuit.x(qubits[0])
        elif gate_name == "y":
            circuit.y(qubits[0])
        elif gate_name == "z":
            circuit.z(qubits[0])
        elif gate_name == "s":
            circuit.s(qubits[0])
        elif gate_name == "sdg":
            circuit.sdg(qubits[0])
        elif gate_name == "t":
            circuit.t(qubits[0])
        elif gate_name == "tdg":
            circuit.tdg(qubits[0])
        elif gate_name == "cx" or gate_name == "cnot":
            circuit.cx(qubits[0], qubits[1])
        elif gate_name == "ccx" or gate_name == "toffoli":
            circuit.ccx(qubits[0], qubits[1], qubits[2])
        elif gate_name == "mcx":
            circuit.mcx(qubits[:-1], qubits[-1])
    
    return circuit

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python circuit_extractor.py <circuit_file.qasm> [output_file.json]")
        return 1
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.rsplit('.', 1)[0] + '.json'
    metadata_file = output_file.rsplit('.', 1)[0] + '_metadata.json'
    
    try:
        # Load circuit
        from qiskit import qasm2
        circuit = qasm2.load(input_file)
        
        # Extract metadata
        metadata = extract_circuit_metadata(circuit)
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save circuit
        save_circuit_json(circuit, output_file)
        
        print(f"Circuit extracted from {input_file}")
        print(f"Metadata saved to {metadata_file}")
        print(f"Circuit data saved to {output_file}")
        
        return 0
    
    except Exception as e:
        print(f"Error processing circuit: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
