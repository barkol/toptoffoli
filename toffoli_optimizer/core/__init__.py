"""
Core components for Toffoli circuit optimization.

This module provides the main components for optimizing quantum circuits containing Toffoli gates,
with a focus on minimizing circuit depth while respecting hardware connectivity constraints.
"""

# Import and expose the main components
from .compiler import ToffoliCompiler, ToffoliType
from .optimizer import (
    ToffoliDepthOptimizer,
    OptimizationStrategy,
)
from .pattern_library_module import ToffoliPatternLibrary

__version__ = '1.0.0'
__author__ = 'Karol Bartkiewicz and Patrycja Tulewicz'

__all__ = [
    'ToffoliCompiler',
    'ToffoliType',
    'ToffoliDepthOptimizer',
    'OptimizationStrategy',
    'ToffoliPatternLibrary',
]
