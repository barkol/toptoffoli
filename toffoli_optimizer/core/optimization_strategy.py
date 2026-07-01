"""
Optimization Strategy Module

This module defines the OptimizationStrategy enum which represents different strategies
for optimizing Toffoli circuits.
"""

from enum import Enum, auto

class OptimizationStrategy(Enum):
    """Enumeration of optimization strategies for Toffoli circuits."""
    STANDARD = auto()                # Prioritize depth, then gates, then fidelity
    DEPTH_REDUCTION = auto()         # Only care about depth
    GATE_REDUCTION = auto()          # Only care about gate count
    FIDELITY = auto()                # Only care about fidelity
    HYBRID = auto()                  # Use a weighted score of all metrics
    ULTRA_DEPTH_REDUCTION = auto()   # Aggressively reduce depth with fidelity trade-offs
    DEPTH_FIDELITY_BALANCE = auto()  # Explicitly balance depth and fidelity with configurable weights
    TRANSPILER = auto()              # Use Qiskit's transpiler directly (Level 3)
    TRANSPILER_L1 = auto()           # Use Qiskit's transpiler with optimization level 1
    TRANSPILER_L2 = auto()           # Use Qiskit's transpiler with optimization level 2
    TRANSPILER_L3 = auto()           # Use Qiskit's transpiler with optimization level 3
    TOFFOLI_COUNT_REDUCTION = auto() # Reduce the NUMBER of atomic CCX/MCX gates (kept atomic, not decomposed)
