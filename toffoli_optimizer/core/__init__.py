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
)
from .pattern_library import (
    ToffoliPatternGenerator,
    ToffoliPatternOptimizer
)

# Add function for validating physical circuits
from .optimizer import validate_physical_circuit

# Version information
__version__ = '1.0.0'
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
    'ToffoliPatternOptimizer'
]
