"""
Example Scripts for Toffoli Optimizer

This module provides example scripts demonstrating the usage of the
Toffoli Optimizer library for circuit optimization and benchmarking.
"""

# Import useful functions for examples
from .optimize_network import (
    optimize_toffoli_network_example,
    load_and_optimize_network
)

from .run_benchmark import (
    run_simple_benchmark,
    run_comparison_benchmark
)

__all__ = [
    'optimize_toffoli_network_example',
    'load_and_optimize_network',
    'run_simple_benchmark',
    'run_comparison_benchmark'
]
