"""
Toffoli Optimizer Package

This package provides tools for optimizing quantum circuits with Toffoli gates,
including depth optimization, gate count reduction, and benchmark utilities.
"""

__version__ = '1.0.0'
__author__ = 'Karol Bartkiewicz and Patrycja Tulewicz'

# Import core components for easier access
from toffoli_optimizer.core import (
    ToffoliCompiler, 
    ToffoliType,
    ToffoliDepthOptimizer, 
    OptimizationStrategy
)

# Import utility components
from toffoli_optimizer.utils import (
    estimate_fidelity,
    create_optimized_physical_mapping,
    get_default_coupling_map,
    save_circuit_to_qasm,
    load_circuit_from_qasm
)

# Import scripts/command classes
from toffoli_optimizer.scripts.optimize import Optimize
from toffoli_optimizer.scripts.base_optimizer import BaseOptimizer
