"""
Toffoli Optimizer Library

A comprehensive library for optimizing Toffoli gate networks on quantum hardware
with limited connectivity constraints.

Main components:
- Compiler: Convert Toffoli networks to quantum circuits
- Optimizer: Reduce circuit depth while respecting hardware constraints
- Pattern Library: Identify and replace common Toffoli gate patterns
- Utils: Circuit manipulation, I/O, and visualization utilities
- Benchmark: Tools for evaluating optimizer performance
"""

from .core import (
    ToffoliCompiler, 
    ToffoliType,
    ToffoliDepthOptimizer, 
    EnhancedToffoliDepthOptimizer,
    OptimizationStrategy,
    validate_physical_circuit
)

from .utils import (
    QISKIT_AVAILABLE,
    create_optimized_physical_mapping,
    validate_physical_circuit,
    estimate_fidelity,
    save_circuit_safely,
    load_circuit_from_qasm
)

# Define version
__version__ = '1.0.0'

# Define all exportable components
__all__ = [
    # Version
    '__version__',
    
    # Core components
    'ToffoliCompiler',
    'ToffoliType',
    'ToffoliDepthOptimizer',
    'EnhancedToffoliDepthOptimizer',
    'OptimizationStrategy',
    'validate_physical_circuit',
    
    # Utility components
    'QISKIT_AVAILABLE',
    'create_optimized_physical_mapping',
    'validate_physical_circuit',
    'estimate_fidelity',
    'save_circuit_safely',
    'load_circuit_from_qasm'
]
