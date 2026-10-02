"""
Quantum Circuit Utilities Package

This package provides utilities for working with quantum circuits,
including physical mapping, circuit validation, fidelity estimation,
I/O operations, and visualization tools.
"""

from toffoli_optimizer._legacy import warn_legacy as _warn_legacy
_warn_legacy("toffoli_optimizer.utils", stacklevel=3)


# Try to import Qiskit to check availability
try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

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
    save_circuit_image,
    save_circuit_stats,
    save_benchmark_circuits,
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

# Package version - use the same as main package
__version__ = "1.0.0"

# Package metadata
__author__ = "Karol Bartkiewicz and Patrycja Tulewicz"
__email__ = "karol.bartkiewicz@amu.edu.pl"
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
    'save_circuit_image',
    'save_circuit_stats',
    'save_benchmark_circuits',
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

# Ensure QISKIT_AVAILABLE is properly propagated to the submodules
import sys
from . import circuit_utils
from . import io_utils
from . import visualization

circuit_utils.QISKIT_AVAILABLE = QISKIT_AVAILABLE
io_utils.QISKIT_AVAILABLE = QISKIT_AVAILABLE
visualization.QISKIT_AVAILABLE = QISKIT_AVAILABLE
