"""Deprecation of the legacy part of the package (release v1.0, May 2025).

The maintained part is the certified decomposition pass of the paper:
``toffoli_optimizer.core`` modules decomposition_selector, subspace_check,
reach_local, reachable_subspace, window_pairs, orientation, context_analysis,
phase_observability, error_model, equivalence_verifier, scalable_verification and
toffoli_count_reducer.

The legacy modules (core.compiler, core.toffoli_depth_optimizer,
core.circuit_gate_processor, core.pattern_library_module, the utils package and
the command-line scripts) are kept only for reproducibility of the first release.
The audit of 2026-10-02 found confirmed correctness bugs in them, including a
default Toffoli decomposition that is not a CCX, pattern rewrites that are not
equivalent to CCX out of context, gates silently dropped or moved to wrong qubits,
and command-line tools that write circuits not equivalent to their input. The bugs
are pinned as strict xfail tests in tests/unit/. Do not use these modules for
results that must be correct.
"""
import warnings

LEGACY_MSG = ("{name} belongs to the deprecated legacy part of toptoffoli (v1.0). It has "
              "known correctness bugs (see toffoli_optimizer/_legacy.py and tests/unit/); "
              "use toffoli_optimizer.core.ErrorBudgetSelector for certified results.")


def warn_legacy(name: str, stacklevel: int = 3) -> None:
    warnings.warn(LEGACY_MSG.format(name=name), DeprecationWarning, stacklevel=stacklevel)
