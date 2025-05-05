#!/usr/bin/env python3
"""
Generate Valid Toffoli Network Circuit

This script creates a valid Toffoli network circuit file to use with the
optimization sweep script. The original toffoli_network_circuit.json file appears
to be empty or invalid.
"""

import json
import os

# Create a valid Toffoli network
toffoli_network = {
    "num_qubits": 8,
    "num_clbits": 3,
    "instructions": [
        # Toffoli gates in the format expected by the optimizer
        {
            "name": "mcx",
            "qubits": [0, 1, 2],
            "control_qubits": [0, 1],
            "target_qubit": 2
        },
        {
            "name": "mcx",
            "qubits": [0, 1, 3],
            "control_qubits": [0, 1],
            "target_qubit": 3
        },
        {
            "name": "mcx",
            "qubits": [0, 2, 4],
            "control_qubits": [0, 2],
            "target_qubit": 4
        },
        {
            "name": "mcx",
            "qubits": [1, 2, 5],
            "control_qubits": [1, 2],
            "target_qubit": 5
        },
        {
            "name": "mcx",
            "qubits": [2, 3, 6],
            "control_qubits": [2, 3],
            "target_qubit": 6
        },
        {
            "name": "mcx",
            "qubits": [4, 5, 7],
            "control_qubits": [4, 5],
            "target_qubit": 7
        }
    ]
}

# Create corresponding metadata
metadata = {
    "num_qubits": 8,
    "num_clbits": 3,
    "gate_count": len(toffoli_network["instructions"]),
    "toffoli_count": len(toffoli_network["instructions"]),
    "creation_parameters": {
        "qubits": 2,
        "ancillas": 4,
        "controlling_anc": 3
    },
    "input_qubits": [0, 1, 2],
    "output_qubits": [3, 6, 7]
}

# Save the files
with open("toffoli_network_circuit.json", "w") as f:
    json.dump(toffoli_network, f, indent=2)
    print(f"Created toffoli_network_circuit.json with {len(toffoli_network['instructions'])} Toffoli gates")

with open("toffoli_network_circuit_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
    print(f"Created toffoli_network_circuit_metadata.json")

print("\nNow you can run the optimization sweep with:")
print("python optimization-sweep-script-revised.py --input toffoli_network_circuit --output comprehensive_results --topology falcon --target-fidelities 0.85,0.90,0.95,0.99 --num-trials 5 --visualize")
