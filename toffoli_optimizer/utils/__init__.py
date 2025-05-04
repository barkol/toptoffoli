"""
Quantum Circuit Utilities Package

This package provides utilities for working with quantum circuits,
including physical mapping, circuit validation, fidelity estimation,
I/O operations, and visualization tools.
"""

# Import and expose functions from circuit_utils
from .circuit_utils import (
    calculate_gate_fidelities,
    estimate_fidelity,
    create_naive_physical_mapping,
    create_optimized_physical_mapping,
    validate_physical_circuit,
    get_default_coupling_map,
    CircuitGateProcessor,
    get_qubit_index
)

# Import and expose functions from io_utils
from .io_utils import (
    save_circuit_to_qasm,
    load_circuit_from_qasm,
    save_circuit_safely,
    save_circuit_image as io_save_circuit_image,
    save_circuit_stats as io_save_circuit_stats,
    save_benchmark_circuits as io_save_benchmark_circuits,
    define_loaded_toffoli_network,
    ToffoliNetworkLoader
)

# Import and expose functions from visualization
from .visualization import (
    save_circuit_image,
    save_circuit_text,
    save_circuit_stats,
    save_circuit_safely as viz_save_circuit_safely,
    save_benchmark_circuits,
    plot_optimization_results,
    visualize_optimization
)

# Try to import Qiskit to check availability
try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Package version
__version__ = "0.1.0"

# Package metadata
__author__ = "Quantum Optimization Team"
__email__ = "quantum@example.com"
__description__ = "Utilities for working with quantum circuits"

# Public API
__all__ = [
    # circuit_utils exports
    'calculate_gate_fidelities',
    'estimate_fidelity',
    'create_naive_physical_mapping',
    'create_optimized_physical_mapping',
    'validate_physical_circuit',
    'get_default_coupling_map',
    'CircuitGateProcessor',
    'get_qubit_index',
    
    # io_utils exports
    'save_circuit_to_qasm',
    'load_circuit_from_qasm',
    'save_circuit_safely',
    'io_save_circuit_image',
    'io_save_circuit_stats',
    'io_save_benchmark_circuits',
    'define_loaded_toffoli_network',
    'ToffoliNetworkLoader',
    
    # visualization exports
    'save_circuit_image',
    'save_circuit_text',
    'save_circuit_stats',
    'viz_save_circuit_safely',
    'save_benchmark_circuits',
    'plot_optimization_results',
    'visualize_optimization',
    
    # Package info
    'QISKIT_AVAILABLE'
]

# Provide global QISKIT_AVAILABLE flag for consistent behavior across modules
#import circuit_utils
#import io_utils
#import visualization

circuit_utils.QISKIT_AVAILABLE = QISKIT_AVAILABLE
io_utils.QISKIT_AVAILABLE = QISKIT_AVAILABLE
visualization.QISKIT_AVAILABLE = QISKIT_AVAILABLE
