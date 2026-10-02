"""
Toffoli Optimizer Package

This package provides tools for optimizing quantum circuits with Toffoli gates,
including depth optimization, gate count reduction, and benchmark utilities.
"""

__version__ = '1.2.1'
__author__ = 'Karol Bartkiewicz and Patrycja Tulewicz'

# Maintained, certified pass (the paper's method).
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector  # noqa: F401
from toffoli_optimizer.core.error_model import HardwareErrorModel  # noqa: F401

# Deprecated legacy API (v1.0): loaded lazily and with a DeprecationWarning.
_LEGACY = {
    "ToffoliCompiler": ("toffoli_optimizer.core.compiler", "ToffoliCompiler"),
    "ToffoliType": ("toffoli_optimizer.core.compiler", "ToffoliType"),
    "ToffoliDepthOptimizer": ("toffoli_optimizer.core.toffoli_depth_optimizer", "ToffoliDepthOptimizer"),
    "OptimizationStrategy": ("toffoli_optimizer.core.optimization_strategy", "OptimizationStrategy"),
    "estimate_fidelity": ("toffoli_optimizer.utils", "estimate_fidelity"),
    "create_optimized_physical_mapping": ("toffoli_optimizer.utils", "create_optimized_physical_mapping"),
    "get_default_coupling_map": ("toffoli_optimizer.utils", "get_default_coupling_map"),
    "save_circuit_to_qasm": ("toffoli_optimizer.utils", "save_circuit_to_qasm"),
    "load_circuit_from_qasm": ("toffoli_optimizer.utils", "load_circuit_from_qasm"),
    "Optimize": ("toffoli_optimizer.scripts.optimize", "Optimize"),
    "BaseOptimizer": ("toffoli_optimizer.scripts.base_optimizer", "BaseOptimizer"),
}


def __getattr__(name):
    if name in _LEGACY:
        import importlib
        from toffoli_optimizer._legacy import warn_legacy
        warn_legacy(name)
        mod, attr = _LEGACY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'toffoli_optimizer' has no attribute {name!r}")
