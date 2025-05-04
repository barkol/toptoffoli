"""
Toffoli Optimizer Library

A comprehensive library for optimizing Toffoli gate networks on quantum hardware
with limited connectivity constraints.

Main components:
- Compiler: Convert Toffoli networks to quantum circuits
- Optimizer: Reduce circuit depth while respecting hardware constraints
- Pattern Library: Identify and replace common Toffoli gate patterns
- ZX-Calculus: Optimize circuits using ZX-calculus techniques
- RL-Optimizer: Use reinforcement learning to find optimal transformations
- Utils: Circuit manipulation, I/O, and visualization utilities
- Benchmark: Tools for evaluating optimizer performance
"""

from .core import (
    ToffoliCompiler, 
    ToffoliType,
    ToffoliDepthOptimizer, 
    EnhancedToffoliDepthOptimizer,
    OptimizationStrategy,
    ZXOptimizer,
    validate_physical_circuit
)

# Import RL components if available
try:
    from .core import (
        RLToffoliOptimizer,
        RLOptimizationWrapper,
        RL_OPTIMIZER_AVAILABLE
    )
except ImportError:
    RL_OPTIMIZER_AVAILABLE = False

from .utils import (
    QISKIT_AVAILABLE,
    create_optimized_physical_mapping,
    validate_physical_circuit,
    estimate_fidelity,
    save_circuit_safely,
    load_circuit_from_qasm
)

# Define version
__version__ = '1.1.0'

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
    'ZXOptimizer',
    'validate_physical_circuit',
    
    # Utility components
    'QISKIT_AVAILABLE',
    'create_optimized_physical_mapping',
    'validate_physical_circuit',
    'estimate_fidelity',
    'save_circuit_safely',
    'load_circuit_from_qasm',
]

# Add RL components if available
if RL_OPTIMIZER_AVAILABLE:
    __all__.extend([
        'RLToffoliOptimizer',
        'RLOptimizationWrapper',
        'RL_OPTIMIZER_AVAILABLE'
    ])
else:
    __all__.append('RL_OPTIMIZER_AVAILABLE')
