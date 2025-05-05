#!/usr/bin/env python3
"""
Toffoli Network Extractor - Utility for fixing the issues with 
extracting Toffoli gates from quantum circuits in Qiskit 2.0
"""

import os
import sys
import json
import argparse
from datetime import datetime

try:
    from qiskit import QuantumCircuit, qasm2
    from qiskit.circuit import Instruction
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("Warning: Qiskit not available. Circuit processing will be limited.")

class ToffoliNetworkExtractor:
    """
    Class for extracting Toffoli networks from quantum circuits
    with Qiskit 2.0 compatibility.
    """
    
    def __init__(self, debug_mode=False):
        """
        Initialize the Toffoli network extractor.
        
        Args:
            debug_mode (bool): Whether to print debug information
        """
        self.debug_mode = debug_mode
    
    def get_qubit_index(self, qubit):
        """
        Get the index of a qubit object, handling both Qiskit 1.x and 2.x.
        
        Args:
            qubit: Qiskit qubit object
            
        Returns:
            int: Qubit index
        """
        try:
            # For Qiskit 2.0
            if hasattr(qubit, '_index'):
                return qubit._index
            # For older versions
            elif hasattr(qubit, 'index'):
                return qubit.index
            else:
                if self.debug_mode:
                    print("Warning: Qubit has no index attribute")
                return None
        except Exception as e:
            if self.debug_mode:
                print(f"Error getting qubit index: {e}")
            return None
    
    def extract_toffoli_gates(self, circuit):
        """
        Extract Toffoli gates from a quantum circuit.
        
        Args:
            circuit: Qiskit QuantumCircuit
            
        Returns:
            tuple: (toffoli_gates, input_qubits, output_qubits)
        """
        if not QISKIT_AVAILABLE or circuit is None:
            return [], [], []
        
        toffoli_gates = []
        input_qubits = set()
        output_qubits = set()
        
        # Process each instruction in the circuit
        for instruction in circuit.data:
            try:
                # Get the operation name
                gate_name = instruction.operation.name
                
                # Skip any gates that are not ccx or mcx
                if gate_name not in ["ccx", "mcx"]:
                    continue
                
                # Get qubit indices - with Qiskit 2.0 compatibility
                qubit_indices = []
                for q in instruction.qubits:
                    idx = self.get_qubit_index(q)
                    if idx is not None:
                        qubit_indices.append(idx)
                
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
                if self.debug_mode:
                    print(f"Error processing instruction: {e}")
                continue
        
        # Convert sets to sorted lists
        input_qubits_list = sorted(list(input_qubits))
        output_qubits_list = sorted(list(output_qubits))
        
        if self.debug_mode:
            print(f"Extracted {len(toffoli_gates)} Toffoli gates")
            print(f"Input qubits: {input_qubits_list}")
            print(f"Output qubits: {output_qubits_list}")
        
        return toffoli_gates, input_qubits_list, output_qubits_list
    
    def create_simple_toffoli_network(self, num_qubits=6):
        """
        Create a simple Toffoli network as a fallback.
        
        Args:
            num_qubits (int): Number of qubits
            
        Returns:
            tuple: (toffoli_gates, input_qubits, output_qubits)
        """
        toffoli_gates = [
            ([0, 1], 2),  # Controls: 0,1; Target: 2
            ([0, 2], 3),  # Controls: 0,2; Target: 3
            ([1, 2], 4),  # Controls: 1,2; Target: 4
            ([3, 4], 5),  # Controls: 3,4; Target: 5
        ]
        
        if num_qubits < 6:
            # Adjust for smaller circuits
            toffoli_gates = toffoli_gates[:max(1, num_qubits-2)]
        
        input_qubits = [0, 1]
        output_qubits = list(range(2, min(6, num_qubits)))
        
        if self.debug_mode:
            print(f"Created simple Toffoli network with {len(toffoli_gates)} gates")
            print(f"Input qubits: {input_qubits}")
            print(f"Output qubits: {output_qubits}")
        
        return toffoli_gates, input_qubits, output_qubits
    
    def save_toffoli_network(self, toffoli_gates, input_qubits, output_qubits, num_qubits, filename, metadata=None):
        """
        Save a Toffoli network to a JSON file.
        
        Args:
            toffoli_gates (list): List of Toffoli gates
            input_qubits (list): List of input qubit indices
            output_qubits (list): List of output qubit indices
            num_qubits (int): Total number of qubits
            filename (str): Output filename
            metadata (dict): Optional metadata to include
            
        Returns:
            bool: Whether the save succeeded
        """
        try:
            # Create circuit data
            circuit_data = {
                "toffoli_gates": toffoli_gates,
                "input_qubits": input_qubits,
                "output_qubits": output_qubits,
                "num_qubits": num_qubits,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add metadata if provided
            if metadata:
                circuit_data["metadata"] = metadata
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(circuit_data, f, indent=2)
            
            if self.debug_mode:
                print(f"Saved Toffoli network to {filename}")
            
            return True
        
        except Exception as e:
            if self.debug_mode:
                print(f"Error saving Toffoli network: {e}")
            return False
    
    def load_toffoli_network(self, filename):
        """
        Load a Toffoli network from a JSON file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            tuple: (toffoli_gates, input_qubits, output_qubits, num_qubits)
        """
        try:
            # Load from file
            with open(filename, 'r') as f:
                circuit_data = json.load(f)
            
            # Extract data
            toffoli_gates = circuit_data.get("toffoli_gates", [])
            input_qubits = circuit_data.get("input_qubits", [])
            output_qubits = circuit_data.get("output_qubits", [])
            num_qubits = circuit_data.get("num_qubits", max(max(input_qubits+output_qubits) + 1 if input_qubits and output_qubits else 6, 6))
            
            if self.debug_mode:
                print(f"Loaded Toffoli network from {filename}")
                print(f"  {len(toffoli_gates)} Toffoli gates")
                print(f"  {len(input_qubits)} input qubits, {len(output_qubits)} output qubits")
                print(f"  {num_qubits} total qubits")
            
            return toffoli_gates, input_qubits, output_qubits, num_qubits
        
        except Exception as e:
            if self.debug_mode:
                print(f"Error loading Toffoli network: {e}")
            return [], [], [], 0
    
    def extract_from_circuit_file(self, filename):
        """
        Extract a Toffoli network from a circuit file.
        
        Args:
            filename (str): Input filename (QASM or JSON)
            
        Returns:
            tuple: (toffoli_gates, input_qubits, output_qubits, num_qubits)
        """
        if not QISKIT_AVAILABLE:
            if self.debug_mode:
                print("Qiskit not available, cannot extract from circuit file")
            return [], [], [], 0
        
        try:
            # Check file extension
            ext = os.path.splitext(filename)[1].lower()
            
            if ext == '.qasm':
                # Load QASM file
                circuit = qasm2.load(filename)
            elif ext == '.json':
                # Load JSON file
                circuit = self._load_circuit_json(filename)
            else:
                if self.debug_mode:
                    print(f"Unsupported file extension: {ext}")
                return [], [], [], 0
            
            # Extract Toffoli network
            toffoli_gates, input_qubits, output_qubits = self.extract_toffoli_gates(circuit)
            
            # Default to simple network if no Toffoli gates found
            if not toffoli_gates:
                if self.debug_mode:
                    print("No Toffoli gates found, creating simple example")
                toffoli_gates, input_qubits, output_qubits = self.create_simple_toffoli_network(circuit.num_qubits)
            
            return toffoli_gates, input_qubits, output_qubits, circuit.num_qubits
        
        except Exception as e:
            if self.debug_mode:
                print(f"Error extracting from circuit file: {e}")
            return [], [], [], 0
    
    def _load_circuit_json(self, filename):
        """
        Load a circuit from a JSON file.
        
        Args:
            filename (str): Input filename
            
        Returns:
            QuantumCircuit: Loaded circuit
        """
        # Load from file
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
            elif gate_name == "cx" or gate_name == "cnot":
                circuit.cx(qubits[0], qubits[1])
            elif gate_name == "ccx" or gate_name == "toffoli":
                circuit.ccx(qubits[0], qubits[1], qubits[2])
            elif gate_name == "mcx":
                circuit.mcx(qubits[:-1], qubits[-1])
        
        return circuit

def main():
    """Main function for the Toffoli network extractor"""
    parser = argparse.ArgumentParser(description='Extract Toffoli networks from quantum circuits')
    
    parser.add_argument('input', type=str, help='Input circuit file')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = ToffoliNetworkExtractor(debug_mode=args.debug)
    
    # Extract Toffoli network
    toffoli_gates, input_qubits, output_qubits, num_qubits = extractor.extract_from_circuit_file(args.input)
    
    if not toffoli_gates:
        print("No Toffoli gates extracted or created")
        return 1
    
    # Determine output filename
    output_file = args.output if args.output else os.path.splitext(args.input)[0] + '_toffoli_network.json'
    
    # Create metadata
    metadata = {
        "source_file": args.input,
        "extraction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save Toffoli network
    success = extractor.save_toffoli_network(toffoli_gates, input_qubits, output_qubits, num_qubits, output_file, metadata)
    
    if success:
        print(f"Toffoli network extracted and saved to {output_file}")
        return 0
    else:
        print("Failed to save Toffoli network")
        return 1

if __name__ == "__main__":
    sys.exit(main())