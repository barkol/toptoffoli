"""
Utility functions for the Toffoli Optimizer package.

This module provides various utilities for working with quantum circuits,
file I/O operations, and visualization.
"""

# Import and expose utility functions
from .circuit_utils import (
    estimate_fidelity,
    calculate_gate_fidelities,
    create_optimized_physical_mapping,
    create_naive_physical_mapping,
    get_qubit_index
)

from .io_utils import (
    save_circuit_to_qasm,
    load_circuit_from_qasm,
    load_toffoli_network,
    save_toffoli_network
)

from .visualization import (
    save_circuit_image,
    save_circuit_stats,
    save_circuit_text
)

# Define what's available directly from the utils module
__all__ = [
    # Circuit utilities
    'estimate_fidelity',
    'calculate_gate_fidelities',
    'create_optimized_physical_mapping',
    'create_naive_physical_mapping',
    'get_qubit_index',
    
    # I/O utilities
    'save_circuit_to_qasm',
    'load_circuit_from_qasm',
    'load_toffoli_network',
    'save_toffoli_network',
    
    # Visualization utilities
    'save_circuit_image',
    'save_circuit_stats',
    'save_circuit_text'
]
