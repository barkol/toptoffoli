"""
Core components for Toffoli circuit optimization.

This module provides the main components for optimizing quantum circuits containing Toffoli gates,
with a focus on minimizing circuit depth while respecting hardware connectivity constraints.
"""

# Import and expose the main components
from .compiler import ToffoliCompiler, ToffoliType
from .optimizer import (
    ToffoliDepthOptimizer,
    EnhancedToffoliDepthOptimizer,
    OptimizationStrategy,
    validate_physical_circuit
)
from .pattern_library import (
    ToffoliPatternGenerator, 
    ToffoliPatternOptimizer
)

# Import ZX-calculus optimizer
from .zx_optimizer import ZXOptimizer

# Import RL-based optimizer if available
try:
    from .rl_optimizer import (
        RLToffoliOptimizer,
        GateTransformation,
        QLearningState
    )
    from .rl_integration import RLOptimizationWrapper
    RL_OPTIMIZER_AVAILABLE = True
except ImportError:
    RL_OPTIMIZER_AVAILABLE = False

# Version information
__version__ = '1.1.0'
__author__ = 'ToffoliOptimizer Contributors'

# Define what's available directly from the core module
__all__ = [
    # From compiler
    'ToffoliCompiler',
    'ToffoliType',
    
    # From optimizer
    'ToffoliDepthOptimizer',
    'EnhancedToffoliDepthOptimizer',
    'OptimizationStrategy',
    'validate_physical_circuit',
    
    # From pattern library
    'ToffoliPatternGenerator',
    'ToffoliPatternOptimizer',
    
    # From ZX-calculus optimizer
    'ZXOptimizer',
]

# Add RL components if available
if RL_OPTIMIZER_AVAILABLE:
    __all__.extend([
        'RLToffoliOptimizer',
        'GateTransformation', 
        'QLearningState',
        'RLOptimizationWrapper',
        'RL_OPTIMIZER_AVAILABLE'
    ])
else:
    __all__.append('RL_OPTIMIZER_AVAILABLE')
