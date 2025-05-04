"""
Network Generators for Toffoli Benchmarks

This module provides functions to generate various types of Toffoli networks
for benchmarking the Toffoli Depth Optimizer.
"""

import random
import os

# Import save utilities
from ..utils.io_utils import save_circuit_to_qasm as save_circuit

def define_toffoli_network():
    """
    Define a standard 2-bit adder Toffoli network.
    
    Returns:
        tuple: (toffoli_gates, output_qubits, input_qubits)
    """
    # Define a simple 2-bit adder Toffoli network
    toffoli_gates = [
        ([0, 1], 2),  # Controls: 0,1; Target: 2
        ([0, 1], 3),  # Controls: 0,1; Target: 3
        ([0, 2], 4),  # Controls: 0,2; Target: 4
        ([1, 2], 5),  # Controls: 1,2; Target: 5
        ([2, 3], 6),  # Controls: 2,3; Target: 6
        ([4, 5], 7),  # Controls: 4,5; Target: 7
    ]
    output_qubits = [3, 6, 7]  # Sum bits and carry
    input_qubits = [0, 1, 2]   # Input bits
    
    return toffoli_gates, output_qubits, input_qubits

def define_variable_toffoli_network(num_gates=5, num_qubits=8, seed=None):
    """
    Generate a variable Toffoli network with the specified number of gates and qubits.
    
    Args:
        num_gates (int): Number of Toffoli gates to generate
        num_qubits (int): Number of qubits in the network
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (toffoli_gates, output_qubits, input_qubits)
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Generate random Toffoli gates
    toffoli_gates = []
    for _ in range(num_gates):
        # Select random control qubits (2 different qubits)
        available_qubits = list(range(num_qubits))
        control1 = random.choice(available_qubits)
        available_qubits.remove(control1)
        control2 = random.choice(available_qubits)
        available_qubits.remove(control2)
        
        # Select random target qubit (different from controls)
        if available_qubits:
            target = random.choice(available_qubits)
        else:
            # If num_qubits <= 3, we might need to reuse qubits
            # In this case, pick a random qubit that's not one of the controls
            target = random.choice([q for q in range(num_qubits) if q != control1 and q != control2])
        
        # Add the Toffoli gate
        toffoli_gates.append(([control1, control2], target))
    
    # Define input and output qubits based on gate usage
    control_qubits = set()
    target_qubits = set()
    
    for controls, target in toffoli_gates:
        for control in controls:
            control_qubits.add(control)
        target_qubits.add(target)
    
    # Input qubits = qubits that are only used as controls or used as controls first
    input_qubits = []
    for qubit in sorted(control_qubits):
        # Check if this qubit is used as a target before it's used as a control
        first_control = None
        first_target = None
        
        for i, (controls, target) in enumerate(toffoli_gates):
            if qubit in controls and (first_control is None or i < first_control):
                first_control = i
            if qubit == target and (first_target is None or i < first_target):
                first_target = i
        
        # If it's used as a control first or only used as a control, consider it an input
        if first_target is None or (first_control is not None and first_control < first_target):
            input_qubits.append(qubit)
    
    # Output qubits = target qubits that are not subsequently used as controls
    # or qubits that are used as a target in the last gates
    output_qubits = []
    for qubit in sorted(target_qubits):
        # Check if this target qubit is subsequently used as a control
        is_output = True
        
        for i, (controls, target) in enumerate(toffoli_gates):
            # Find the last occurrence as a target
            last_target = None
            for j, (_, tgt) in enumerate(toffoli_gates):
                if qubit == tgt:
                    last_target = j
            
            # Check if it's used as a control after the last target
            if qubit in controls and last_target is not None and i > last_target:
                is_output = False
        
        if is_output:
            output_qubits.append(qubit)
    
    # Ensure we have at least some inputs and outputs
    if not input_qubits:
        input_qubits = [0, 1]  # Default inputs
    
    if not output_qubits:
        output_qubits = [min(2, num_qubits-1)]  # Default output
    
    return toffoli_gates, output_qubits, input_qubits
