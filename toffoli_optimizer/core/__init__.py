"""
Core components for Toffoli circuit optimization.

This module provides the main components for optimizing quantum circuits containing Toffoli gates,
with a focus on minimizing circuit depth while respecting hardware connectivity constraints.
"""

# Import and expose the main components
# Maintained, certified pass.
from .decomposition_selector import ErrorBudgetSelector  # noqa: F401
from .error_model import HardwareErrorModel  # noqa: F401

__version__ = '1.2.1'
__author__ = 'Karol Bartkiewicz and Patrycja Tulewicz'

# Deprecated legacy API (v1.0): lazy, with a DeprecationWarning.
_LEGACY = {
    'ToffoliCompiler': ('.compiler', 'ToffoliCompiler'),
    'ToffoliType': ('.compiler', 'ToffoliType'),
    'ToffoliDepthOptimizer': ('.toffoli_depth_optimizer', 'ToffoliDepthOptimizer'),
    'OptimizationStrategy': ('.optimization_strategy', 'OptimizationStrategy'),
    'ToffoliPatternLibrary': ('.pattern_library_module', 'ToffoliPatternLibrary'),
}

__all__ = ['ErrorBudgetSelector', 'HardwareErrorModel'] + list(_LEGACY)


def __getattr__(name):
    if name in _LEGACY:
        import importlib
        from toffoli_optimizer._legacy import warn_legacy
        warn_legacy(name)
        mod, attr = _LEGACY[name]
        return getattr(importlib.import_module(mod, __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
