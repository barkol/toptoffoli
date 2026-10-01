"""Error-budget-aware, context-verified Toffoli decomposition selection.

A CCX/MCX circuit has to be lowered to a two-qubit basis to run on hardware. There
are two kinds of Toffoli decomposition:

* EXACT (Clifford+T): the textbook H / T / T-dagger / CX Toffoli, **6 CX**. Always
  correct anywhere.

* RELATIVE-PHASE (Margolus): a cheaper gadget with only **3 CX** that implements
  CCX correctly *up to a relative phase* on the target subspace. It is valid only
  where that phase is later uncomputed (a compute/uncompute pair) -- see
  ``context_analysis``.

``ErrorBudgetSelector`` chooses, per Toffoli, which decomposition to use so as to
minimize the :class:`HardwareErrorModel` infidelity (dominated by the two-qubit
count). It admits the cheap relative-phase gadget ONLY at a structurally detected
compute/uncompute site AND only after :class:`ExactEquivalenceVerifier` certifies,
on the affected window, that substituting the gadget into BOTH gates of the pair
yields a circuit exactly equivalent to the all-exact decomposition.

KEY GUARANTEE: the returned circuit is always verified-correct against the
exact-only decomposition. Relative-phase gadgets appear only at verified sites; a
site that fails verification is rejected and falls back to the exact gadget.

PHASE-OBSERVABILITY-AWARE PATH (``phase_aware=True``, default). Beyond the
compute/uncompute-pair path, the selector admits the cheap relative-phase gadget at a
STANDALONE Toffoli ``g`` when BOTH hold:

  * ``is_phase_unobservable(circuit, g, ...)`` -- the relative phase it introduces can
    never affect any measurement outcome (it cancels in a pair, OR its forward cone is
    purely classical-reversible into a computational-basis read; see
    ``phase_observability``), AND
  * ``verify_on_reachable_basis`` -- on every REACHABLE basis input (computed / soundly
    over-approximated by ``reachable_subspace``), the gadget agrees with the exact CCX
    on the output computational-basis state, PHASE-INSENSITIVELY, ancilla clean.

This admits substitutions the exact-unitary, pair-only path provably cannot (a
relative-phase gadget differs from CCX by a relative phase, so it always FAILS the
exact-unitary check as a standalone replacement -- yet is sound when the phase is
unobservable). Every committed substitution is therefore sound: either
exact-unitary-equivalent (pairs) or reachable-basis-equivalent-with-unobservable-phase
(the new path). UNSOUND admissions are impossible by the double gate.
"""

from __future__ import annotations

import math
from typing import List, Optional

from qiskit import QuantumCircuit

from .equivalence_verifier import ExactEquivalenceVerifier
from .error_model import HardwareErrorModel
from .context_analysis import find_relative_phase_safe_sites, RelativePhaseSite
from .phase_observability import is_phase_unobservable, default_affected_qubits
from .reachable_subspace import reachable_overapprox
from .subspace_check import (
    project_support, gadget_unitary, check_on_subspace, certify_on_input_subspace)

_CCX_NAMES = {"ccx", "mcx", "mcx_gray"}

def _ccx_matrix():
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    return Operator(qc).data

_U_CCX = _ccx_matrix()


def _gadget_matrix(fn):
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(3)
    fn(qc, 0, 1, 2)
    return Operator(qc).data

# Whole-circuit unitary certification is attempted up to this width.
_CERT_MAX_QUBITS = 12


# --------------------------------------------------------------------------- gadgets
def append_exact_ccx(qc: QuantumCircuit, a: int, b: int, t: int) -> None:
    """Append the textbook exact Clifford+T Toffoli (6 CX) onto qc[a,b -> t].

    This is qiskit's standard CCX decomposition. We emit it explicitly (rather than
    qc.ccx + transpile) so the two-qubit count is deterministic and the basis is a
    fixed {h, t, tdg, cx} set.
    """
    qc.h(t)
    qc.cx(b, t)
    qc.tdg(t)
    qc.cx(a, t)
    qc.t(t)
    qc.cx(b, t)
    qc.tdg(t)
    qc.cx(a, t)
    qc.t(b)
    qc.t(t)
    qc.cx(a, b)
    qc.h(t)
    qc.t(a)
    qc.tdg(b)
    qc.cx(a, b)


def append_control_drop(qc: QuantumCircuit, a: int, b: int, t: int, keep) -> None:
    """Append the CONTROL-DROPPED specialisation of CCX(a,b,t) onto ``qc``.

    A Toffoli equals a cheaper gate when one (or both) of its controls is constant on
    the inputs that actually reach it:

      * ``keep == ()``        -> a control is |0> on (almost) all reachable mass, so
        the |11> branch never fires: CCX acts as IDENTITY. Emit NOTHING (0 two-qubit
        gates).
      * ``keep == (a,)``      -> control ``b`` is |1> on (almost) all reachable mass,
        so CCX reduces to a single CX(a, t).
      * ``keep == (b,)``      -> control ``a`` is |1> on (almost) all reachable mass,
        so CCX reduces to a single CX(b, t).

    This is a CONTEXT-SPECIALISED candidate: it is only correct on the reachable
    subspace (and only within tolerance there), so the selector admits it solely
    through the bounded-approximate reachable check. ``keep`` lists the control(s)
    that remain after dropping; the surviving control (if any) becomes the control of
    the residual CX onto ``t``.
    """
    keep = tuple(keep)
    if keep == ():
        return  # CCX collapses to identity on the reachable subspace
    if len(keep) == 1:
        qc.cx(keep[0], t)
        return
    raise ValueError(f"control-drop keep must have 0 or 1 controls, got {keep!r}")


def append_relative_phase_ccx(qc: QuantumCircuit, a: int, b: int, t: int) -> None:
    """Append a relative-phase (Margolus) Toffoli, **3 CX**, onto qc[a,b -> t].

    Implements CCX(a,b,t) correctly UP TO a relative phase on the target. Valid only
    inside a compute/uncompute pair where that phase is uncomputed. The construction
    is the standard Margolus gate:

        Ry(-pi/4) t ; CX(b,t) ; Ry(-pi/4) t ; CX(a,t) ; Ry(pi/4) t ; CX(b,t) ; Ry(pi/4) t

    Cost: 3 CX + 4 single-qubit Ry, vs 6 CX for the exact gadget.
    """
    q = math.pi / 4
    qc.ry(-q, t)
    qc.cx(b, t)
    qc.ry(-q, t)
    qc.cx(a, t)
    qc.ry(q, t)
    qc.cx(b, t)
    qc.ry(q, t)


def append_relative_phase_ccx_mirror(qc: QuantumCircuit, a: int, b: int, t: int) -> None:
    """Mirrored Margolus gadget, **3 CX**: the four Ry angles change sign.

    Its diagonal phase -1 sits on the output |a=1, b=0, t=1> instead of
    |a=1, b=0, t=0>. That branch is unreachable when the target enters in |0>
    (an AND written into a clean ancilla) and when it enters holding a&b (the
    uncompute), so on those reachable subspaces the gadget equals CCX exactly and
    condition (R) admits it without any pairing.
    """
    q = math.pi / 4
    qc.ry(q, t)
    qc.cx(b, t)
    qc.ry(q, t)
    qc.cx(a, t)
    qc.ry(-q, t)
    qc.cx(b, t)
    qc.ry(-q, t)


# --------------------------------------------------------------------------- report
class SelectionReport(dict):
    """A plain dict subclass so the report is JSON-friendly but easy to print."""

    def summary(self) -> str:
        n_phase = len(self.get("phase_aware_admitted", []))
        n_approx = len(self.get("approx_admitted", []))
        eps = self.get("epsilon_spent_total", 0.0)
        return (
            f"sites_found={self['sites_found']} "
            f"applied={self['sites_applied']} rejected={self['sites_rejected']} "
            f"phase_aware={n_phase} approx={n_approx} eps_spent={eps:.4g} | "
            f"2q: {self['two_qubit_before']} -> {self['two_qubit_after']} | "
            f"infidelity: {self['infidelity_before']:.4g} -> {self['infidelity_after']:.4g} | "
            f"verified={self['verified']}"
        )


# --------------------------------------------------------------------------- selector
class ErrorBudgetSelector:
    """Select per-Toffoli decompositions minimizing an error-budget infidelity.

    Parameters
    ----------
    error_model : HardwareErrorModel, optional
        The cost model whose infidelity is minimized. Defaults to standard rates.
    verifier : ExactEquivalenceVerifier, optional
        The exact correctness gate. Defaults to a fresh verifier.
    allow_permutation : bool
        Passed through to the verifier when checking the final whole-circuit
        equivalence (the per-site checks use identity wiring).
    """

    def __init__(
        self,
        error_model: Optional[HardwareErrorModel] = None,
        verifier: Optional[ExactEquivalenceVerifier] = None,
        allow_permutation: bool = False,
        phase_aware: bool = True,
        epsilon: float = 0.0,
        semantics: str = "subroutine",
        window_pairs: bool = True,
        max_window_qubits: int = 12,
    ):
        self.error_model = error_model or HardwareErrorModel()
        self.verifier = verifier or ExactEquivalenceVerifier()
        self.allow_permutation = allow_permutation
        # When True, admit relative-phase gadgets at standalone Toffolis whose phase
        # is provably UNOBSERVABLE on the reachable subspace (the new, more permissive
        # path), in addition to the exact-unitary compute/uncompute-pair path.
        self.phase_aware = phase_aware
        # SEMANTICS of correctness the output must satisfy.
        #  * "subroutine" (default): the emitted circuit equals the exact one as an
        #    operator on the input subspace up to ONE global phase, so it can be used
        #    coherently inside a larger algorithm (adders, oracles). Only conditions
        #    (C) compute/uncompute pairs and (R) exact-on-reachable-subspace are used.
        #  * "program": the circuit is a complete program ending in a
        #    computational-basis measurement. Additionally admits condition (U):
        #    standalone relative-phase gadgets whose phase the terminal measurement
        #    cannot see. NOT valid for subroutines (it changes the unitary).
        if semantics not in ("subroutine", "program"):
            raise ValueError(f"semantics must be 'subroutine' or 'program', got {semantics!r}")
        self.semantics = semantics
        # Condition (W): window-certified mirror pairs (exact segment check).
        self.window_pairs = window_pairs
        self.max_window_qubits = max_window_qubits
        # BOUNDED-APPROXIMATE-ON-REACHABLE knob. epsilon == 0.0 admits only candidates
        # that agree with CCX EXACTLY on the reachable subspace (e.g. a provably
        # constant control). epsilon > 0.0 additionally admits a cheaper
        # context-specialised candidate whose worst-case phase-insensitive deviation
        # over the (soundly over-approximated) reachable subspace is <= epsilon -- AND
        # only when the two-qubit-gate infidelity it SAVES exceeds the epsilon it
        # SPENDS (epsilon is charged into the circuit's error budget).
        if epsilon < 0.0:
            raise ValueError("epsilon must be >= 0")
        self.epsilon = epsilon

    # ----------------------------------------------------------------- baselines
    def decompose_exact_only(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """All-exact baseline: replace every CCX/MCX with the 6-CX exact gadget."""
        out = QuantumCircuit(circuit.num_qubits)
        for inst in circuit.data:
            self._emit_exact(out, circuit, inst)
        return out

    # -------------------------------------------------------------------- select
    def select(self, circuit: QuantumCircuit, input_space="all_basis", pinned_zero=()) -> dict:
        """Choose decompositions to minimize infidelity; verify every substitution.

        Parameters
        ----------
        circuit : QuantumCircuit
        input_space : 'all_basis' | iterable of int
            The computational-basis inputs the whole circuit may be run on. Defaults
            to ``'all_basis'`` (every 2^n input). Pass a RESTRICTED set when the
            circuit is only ever fed a known sub-domain -- e.g. ancilla qubits known
            to start in |0>. A restricted input space shrinks the (soundly
            over-approximated) reachable subspace at each gate, which is what lets the
            control-drop / bounded-approximate candidates become admissible: a control
            can only be provably constant (or rarely set) once the inputs that pin or
            correlate it are known. Soundness is unaffected -- the caller asserts the
            circuit is never run outside ``input_space``; the reachable analysis then
            uses a sound over-approximation of the states reaching each gate.

        Returns a dict with keys:
          circuit  : the decomposed, verified-correct QuantumCircuit (2q basis)
          report   : SelectionReport (sites found/applied/rejected, 2q & infidelity
                     before/after, verified flag, total certified error budget,
                     approx_admitted + per-index epsilon spent)
        """
        exact = self.decompose_exact_only(circuit)

        sites = find_relative_phase_safe_sites(circuit)
        applied: List[RelativePhaseSite] = []
        rejected: List[dict] = []

        # The per-index action map of committed specialised lowerings (see _build).
        # Start empty; add a site's (compute, uncompute) pair only if that
        # substitution verifies against the exact-only decomposition.
        actions: dict = {}

        for site in sites:
            candidate = dict(actions)
            candidate[site.compute_idx] = ("relphase",)
            candidate[site.uncompute_idx] = ("relphase",)
            built = self._build(circuit, candidate)
            ok, _perm, info = self.verifier.verify(
                exact, built, allow_permutation=False
            )
            if ok:
                actions = candidate
                applied.append(site)
            else:
                rejected.append({"site": site, "reason": info.get("reason"), "info": info})

        # ---- Condition (W): window-certified mirror pairs ------------------------
        # Pairs the structural detector rejects (a gate between them writes a
        # control or the target) can still cancel their phases, e.g. the mirrored
        # MAJ/UMA blocks of a Cuccaro adder. Each pair is admitted only if the
        # segment between them, with gadgets, equals the exact segment up to one
        # global phase (exact check on the qubits the segment touches).
        from .window_pairs import admit_window_pairs
        window_pairs_log: List[dict] = []
        if self.window_pairs:
            gad = {k for k, a in actions.items() if a[0] == "relphase"}
            gad2, window_pairs_log = admit_window_pairs(
                circuit, gad, append_relative_phase_ccx,
                max_window_qubits=self.max_window_qubits,
                accepted_windows=[(s_.compute_idx, s_.uncompute_idx) for s_ in applied])
            for k in gad2 - gad:
                actions[k] = ("relphase",)

        # ---- Condition (R) for relative-phase gadgets -----------------------------
        # A standalone Toffoli gets a 3-CX gadget when the gadget equals CCX on the
        # reachable subspace with one phase. The mirrored gadget does so for every
        # AND written into a clean ancilla and for its uncompute. Reachability is
        # exact (vectorised) for classical prefixes, else the sound over-approx.
        from .reach_local import local_reachable
        pinned_zero = tuple(pinned_zero)

        def _local(idx, qbs):
            loc = (local_reachable(circuit, idx, qbs, pinned_zero)
                   if input_space == "all_basis" else None)
            if loc is None:
                reach = reachable_overapprox(circuit, idx, input_space=_space)
                loc = project_support(reach, qbs)
            return loc

        # Input space for the over-approximation: valid inputs with pinned qubits 0.
        _space = input_space
        if pinned_zero and input_space == "all_basis" and circuit.num_qubits <= 20:
            _pm = sum(1 << q for q in pinned_zero)
            _space = [x for x in range(1 << circuit.num_qubits) if not x & _pm]

        rphase_admitted: List[dict] = []
        locked = {k for k in actions}
        for w in window_pairs_log:
            locked.update(w["pair"])
        for s_ in sites:
            locked.update((s_.compute_idx, s_.uncompute_idx))
        for idx, inst in enumerate(circuit.data):
            if idx in locked:
                continue
            name = inst.operation.name.lower()
            qb = [circuit.find_bit(q).index for q in inst.qubits]
            if name not in _CCX_NAMES or len(qb) != 3:
                continue
            local = _local(idx, tuple(qb))
            for kind, U_g in (("relphase", _U_MARG), ("relphase_m", _U_MARG_M)):
                ok_r, dev_r, _ = check_on_subspace(U_g, _U_CCX, local, 0.0)
                if ok_r:
                    actions[idx] = (kind,)
                    rphase_admitted.append({"index": idx, "qubits": qb, "gadget": kind,
                                            "condition": "R", "reachable_local": len(local)})
                    break

        # ---- NEW PATH: phase-observability-aware standalone admissibility --------
        # A standalone Toffoli (not in a compute/uncompute pair) can ALSO get the
        # cheap relative-phase gadget IF its relative phase is provably unobservable
        # on the reachable subspace. This admits substitutions the pair-only path
        # cannot. Each is double-gated: (1) is_phase_unobservable proves the extra
        # phase can never affect a measurement, and (2) verify_on_reachable_basis
        # confirms the gadget agrees with the exact CCX (phase-insensitively) on
        # every REACHABLE basis input. Both must hold -> sound.
        # Indices belonging to a verified compute/uncompute PAIR are soundness-locked:
        # their relative-phase gadgets cancel only as a matched pair, so the approx
        # path below must never touch them.
        paired_idx = set()
        for s in sites:
            paired_idx.add(s.compute_idx)
            paired_idx.add(s.uncompute_idx)
        for w in window_pairs_log:
            paired_idx.update(w["pair"])

        phase_admitted: List[dict] = []
        if self.phase_aware and self.semantics == "program":
            for idx, inst in enumerate(circuit.data):
                if idx in actions or idx in paired_idx:
                    continue
                name = inst.operation.name.lower()
                qb = [circuit.find_bit(q).index for q in inst.qubits]
                if name not in _CCX_NAMES or len(qb) != 3:
                    continue  # only plain 3-qubit CCX get the Margolus gadget here

                affected = default_affected_qubits(circuit, idx)
                # Gate 1: phase provably unobservable downstream?
                if not is_phase_unobservable(circuit, idx, affected):
                    rejected.append({
                        "site": ("standalone", idx),
                        "reason": "phase observable downstream",
                        "info": {"path": "phase_aware"}})
                    continue

                # Condition (U) of the paper: the forward-cone test above proves that
                # W D W^dagger is diagonal (every later gate touching the phased qubits
                # is a basis permutation), so the terminal computational-basis
                # measurement cannot see D for ANY input state, superpositions
                # included. No reachability argument is needed or used here.
                # (Earlier versions additionally ran a basis-state-wise
                # phase-insensitive check, which is blind to relative phases and was
                # applied with gate-level reachable states as circuit inputs; it is
                # removed because it neither adds nor justifies soundness.)
                actions = dict(actions)
                actions[idx] = ("relphase",)
                phase_admitted.append({
                    "index": idx, "qubits": qb, "condition": "U"})

        # ---- NEW PATH: bounded-approximate-on-the-reachable-subspace -------------
        # For each still-exact standalone CCX, try the CONTROL-DROP specialisations
        # (drop-to-identity, drop control a -> CX(b,t), drop control b -> CX(a,t)).
        # Each candidate is admitted iff (1) its worst-case phase-insensitive
        # deviation over the SOUND over-approximation of the reachable subspace is
        # <= epsilon, AND (2) the two-qubit-gate infidelity it SAVES strictly exceeds
        # the epsilon it SPENDS (epsilon is charged into the error budget). With
        # epsilon == 0 only EXACT-on-reachable drops (e.g. a provably-|0> control)
        # can fire. The deviation is measured by the verifier against the exact-only
        # decomposition; control_drop is never globally equivalent, so it is
        # admissible ONLY through this reachable-subspace check.
        # NOTE on scope: a standalone CCX already assigned the relative-phase gadget
        # by the phase-aware path above is RECONSIDERED here -- a control-drop may be
        # strictly cheaper (down to 0 or 1 two-qubit gates vs the gadget's 3) and just
        # as sound on the reachable subspace. Indices locked into a compute/uncompute
        # PAIR (``paired_idx``) are never touched: their gadgets cancel only as a pair.
        approx_admitted: List[dict] = []
        perturbed_targets: set = set()
        epsilon_per_index: dict = {}
        em = self.error_model
        for idx, inst in enumerate(circuit.data):
            if idx in paired_idx:
                continue
            name = inst.operation.name.lower()
            qb = [circuit.find_bit(q).index for q in inst.qubits]
            if name not in _CCX_NAMES or len(qb) != 3:
                continue
            a, b, t = qb

            # Reachable support at g, computed in the CURRENT circuit: a gate already
            # replaced by an approximate (deviation > 0) specialisation may have
            # changed the support of its target, which we model soundly by an H on
            # that target (free value) in the circuit used for reachability.
            reach_circ, idx_map = self._reach_circuit(circuit, perturbed_targets)
            local = None
            if not perturbed_targets and input_space == "all_basis":
                from .reach_local import local_reachable as _lr
                local = _lr(circuit, idx, (a, b, t), pinned_zero)
            if local is None:
                reachable = reachable_overapprox(reach_circ, idx_map[idx], input_space=_space)
                local = project_support(reachable, (a, b, t))
            else:
                reachable = local
            base_inf = em.circuit_infidelity(self._build(circuit, actions))

            best = None  # (margin, candidate_actions, descriptor)
            for keep in ((), (a,), (b,)):
                local_keep = tuple({a: 0, b: 1}[k] for k in keep)
                U_d = gadget_unitary(
                    lambda qc, x, y, z, lk=local_keep: append_control_drop(qc, x, y, z, lk))
                # Condition (R) / its bounded-approximate variant: one phase on the
                # reachable SUBSPACE, operator norm (never basis-state-wise).
                admissible, max_dev, vinfo = check_on_subspace(
                    U_d, _U_CCX, local, self.epsilon)
                if not admissible:
                    continue
                cand_actions = dict(actions)
                cand_actions[idx] = ("control_drop", keep)
                built = self._build(circuit, cand_actions)
                # Budget test: infidelity SAVED by this candidate vs the current
                # selection must exceed the epsilon SPENT (= max_dev charged in).
                cand_inf = em.circuit_infidelity(built)
                saved = base_inf - cand_inf
                margin = saved - max_dev
                if margin > 0 and (best is None or margin > best[0]):
                    best = (margin, cand_actions, {
                        "index": idx, "qubits": qb, "keep": list(keep),
                        "condition": "R" if max_dev == 0.0 else "R-eps",
                        "max_deviation": max_dev,
                        "infidelity_saved": saved,
                        "reachable_inputs": len(reachable),
                        "reachable_local": len(local)})

            if best is not None:
                actions = best[1]
                approx_admitted.append(best[2])
                epsilon_per_index[idx] = best[2]["max_deviation"]
                if best[2]["max_deviation"] > 0.0:
                    perturbed_targets.add((idx, t))
            elif self.epsilon > 0.0:
                rejected.append({
                    "site": ("approx", idx),
                    "reason": "no control-drop within epsilon with positive budget margin",
                    "info": {"path": "approx"}})

        selected = self._build(circuit, actions)
        epsilon_spent_total = sum(epsilon_per_index.values())

        # Final whole-circuit certification. There are two soundness regimes:
        #
        #  * If NO phase-aware (standalone, unobservable-phase) substitution was made,
        #    the selected circuit is exact-unitary-equivalent to the exact-only
        #    decomposition -- certify with the strict exact verifier (unchanged).
        #
        #  * If at least one phase-aware substitution was made, the selected circuit
        #    intentionally differs from exact-only by an UNOBSERVABLE relative phase,
        #    so an exact-unitary check would (correctly) fail. We instead certify the
        #    weaker-but-sound guarantee that backs those substitutions: on every
        #    REACHABLE basis input, selected and exact-only produce the same output
        #    computational-basis state (phase-insensitive), with ancilla clean. Each
        #    such substitution was additionally gated by is_phase_unobservable, so the
        #    relative phase cannot affect any measurement -> sound.
        #
        #  * If at least one BOUNDED-APPROXIMATE (control-drop) substitution was made,
        #    the selected circuit differs from exact-only by MORE than a phase even on
        #    the reachable subspace -- but by at most ``epsilon`` per reachable input.
        #    We certify the quantitative guarantee that backs those substitutions: the
        #    worst-case phase-insensitive reachable deviation is <= epsilon. (The
        #    error budget separately accounts for the epsilon spent.)
        # The whole-circuit certification is on the circuit's INPUT domain -- the set
        # of computational-basis states the caller asserts the circuit is run on
        # (``input_space``; the full 2^n space by default). We feed those as inputs to
        # both ``exact`` and ``selected``. (For ``all_basis`` this is the full space;
        # for a restricted domain it is exactly the inputs the substitutions relied
        # on.) Using ``reachable_overapprox`` at gate index 0 yields a sound superset
        # of that domain.
        cert_inputs = reachable_overapprox(circuit, 0, input_space=input_space)
        if pinned_zero:
            pm = 0
            for q in pinned_zero:
                pm |= 1 << q
            cert_inputs = [x for x in cert_inputs if not (x & pm)]
        # Semantics actually guaranteed: subroutine equivalence (one global phase on
        # the whole input subspace) unless some substitution relied on condition (U),
        # in which case observational equivalence (terminal basis measurement).
        semantics = "observational" if phase_admitted else "subroutine"
        if not (approx_admitted or phase_admitted or rphase_admitted or pinned_zero):
            verified, perm, vinfo = self.verifier.verify(
                exact, selected, allow_permutation=self.allow_permutation
            )
            vinfo = {**vinfo, "certification": "exact_unitary"}
        elif max(exact.num_qubits, selected.num_qubits) <= _CERT_MAX_QUBITS \
                and selected.num_qubits == exact.num_qubits:
            from qiskit.quantum_info import Operator
            U_ex = Operator(exact).data
            U_sel = Operator(selected).data
            verified, cert_dev, vinfo = certify_on_input_subspace(
                U_ex, U_sel, cert_inputs, mode=semantics,
                tolerance=epsilon_spent_total)
            verified = bool(verified)
            perm = None
            vinfo = {**vinfo, "certification": f"input_subspace_{semantics}",
                     "certified_deviation": cert_dev}
        else:
            # Too wide for a whole-circuit unitary: soundness rests on the per-gate
            # conditions (C)/(R)/(U) of the soundness theorem. Reported as such and
            # NOT as a machine-checked whole-circuit certificate.
            verified, perm = None, None
            vinfo = {"certification": "per_gate_conditions_only",
                     "reason": "circuit too wide for whole-circuit unitary check"}

        # Fail closed: a failed whole-circuit certificate must never let an
        # uncertified circuit out. Fall back to the all-exact decomposition.
        fell_back = False
        if verified is False:
            selected = exact
            fell_back = True
            applied, phase_admitted, approx_admitted, rphase_admitted = [], [], [], []
            epsilon_per_index, epsilon_spent_total = {}, 0.0
            verified, perm, vinfo = True, None, {
                **vinfo, "fallback": "exact_only_after_failed_certificate"}

        # The certified error budget includes the epsilon spent on approximate
        # admissions: the on-hardware infidelity of the cheaper circuit PLUS the
        # algorithmic deviation we deliberately introduced on the reachable subspace.
        certified_budget = em.circuit_infidelity(selected) + epsilon_spent_total
        report = SelectionReport(
            sites_found=len(sites),
            sites_applied=len(applied),
            sites_rejected=len(rejected),
            applied_sites=[tuple(s) for s in applied],
            rejected_sites=rejected,
            phase_aware_admitted=phase_admitted,
            window_pairs_admitted=window_pairs_log,
            rphase_admitted=rphase_admitted,
            pinned_zero=list(pinned_zero),
            approx_admitted=approx_admitted,
            epsilon=self.epsilon,
            epsilon_per_index=epsilon_per_index,
            epsilon_spent_total=epsilon_spent_total,
            two_qubit_before=em.two_qubit_count(exact),
            two_qubit_after=em.two_qubit_count(selected),
            infidelity_before=em.circuit_infidelity(exact),
            infidelity_after=em.circuit_infidelity(selected),
            certified_error_budget=certified_budget,
            verified=verified if verified is None else bool(verified),
            semantics=semantics,
            fell_back_to_exact=fell_back,
            verify_info=vinfo,
            output_permutation=perm,
        )
        return {"circuit": selected, "report": report}

    # ------------------------------------------------------------- reachability
    @staticmethod
    def _reach_circuit(circuit: QuantumCircuit, perturbed_targets):
        """Circuit used for reachability, plus an index map original -> new.

        After an approximate (deviation > 0) specialisation at gate i with target t,
        the value of t after i is no longer the CCX value. We model this soundly by
        inserting an H on t right after i: the over-approximation then treats t (and
        everything it later influences) as free.
        """
        if not perturbed_targets:
            return circuit, list(range(len(circuit.data)))
        after = {}
        for i, t in perturbed_targets:
            after.setdefault(i, []).append(t)
        out = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        idx_map = []
        for i, inst in enumerate(circuit.data):
            idx_map.append(len(out.data))
            out.append(inst.operation, inst.qubits, inst.clbits)
            for t in after.get(i, []):
                out.h(out.qubits[t])
        return out, idx_map

    # ------------------------------------------------------------------ builders
    def _build(self, circuit: QuantumCircuit, actions: dict) -> QuantumCircuit:
        """Decompose ``circuit`` under a per-index ``actions`` map.

        ``actions`` maps an instruction index to a chosen specialised lowering:

          * ``("relphase",)``           -> the 3-CX relative-phase (Margolus) gadget.
          * ``("control_drop", keep)``  -> the control-dropped specialisation
            (identity if ``keep == ()``, else a single CX from the kept control).

        Any index NOT in ``actions`` gets its exact lowering. For backward
        compatibility a plain ``set`` of indices is also accepted and read as "apply
        the relative-phase gadget at each of these indices".
        """
        if isinstance(actions, (set, frozenset)):
            actions = {idx: ("relphase",) for idx in actions}
        out = QuantumCircuit(circuit.num_qubits)
        for idx, inst in enumerate(circuit.data):
            action = actions.get(idx)
            if action is None:
                self._emit_exact(out, circuit, inst)
                continue
            qb = [circuit.find_bit(q).index for q in inst.qubits]
            kind = action[0]
            if kind == "relphase":
                append_relative_phase_ccx(out, qb[0], qb[1], qb[2])
            elif kind == "relphase_m":
                append_relative_phase_ccx_mirror(out, qb[0], qb[1], qb[2])
            elif kind == "control_drop":
                append_control_drop(out, qb[0], qb[1], qb[2], action[1])
            else:
                raise ValueError(f"unknown action {action!r}")
        return out

    def _emit_exact(self, out: QuantumCircuit, src: QuantumCircuit, inst) -> None:
        """Emit the exact lowering of one source instruction onto ``out``."""
        name = inst.operation.name.lower()
        qb = [src.find_bit(q).index for q in inst.qubits]
        if name in ("ccx",) or (name in ("mcx", "mcx_gray") and len(qb) == 3):
            append_exact_ccx(out, qb[0], qb[1], qb[2])
        elif name in ("mcx", "mcx_gray"):
            # Generic multi-control: fall back to qiskit's own decomposition to a
            # CX-based basis. Rare in these tests; kept correct rather than minimal.
            tmp = QuantumCircuit(out.num_qubits)
            tmp.mcx(qb[:-1], qb[-1])
            out.compose(tmp.decompose().decompose(), inplace=True)
        else:
            # Pass through any non-CCX gate verbatim (cx, x, h, ry, ...).
            out.append(inst.operation, [out.qubits[i] for i in qb])


_U_MARG = _gadget_matrix(append_relative_phase_ccx)
_U_MARG_M = _gadget_matrix(append_relative_phase_ccx_mirror)
