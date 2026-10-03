# Lower bounds of Proposition 1 (CNOT cost of a Toffoli on a reachable support)

## Setting
Three qubits a, b (controls), t (target). Basis strings are written abt. CCX maps |a b t> to |a b (t xor ab)>.
For a nonempty L ⊆ {0,1}^3 let c*(L) be the smallest k such that there is a circuit U consisting of k CNOT gates
(any ordered pair of distinct qubits, any direction) interleaved with ARBITRARY single-qubit unitaries (before, between, after)
with   U|x> = e^{iθ} CCX|x>   for all x in L, with ONE common phase θ.
(Equivalently U P_L = e^{iθ} CCX P_L with P_L the projector onto span(L).)

Let Π = projection of L onto the controls ab, and Q = {100, 101, 010, 011}.
Claimed values (numerically confirmed for all 159 supports up to exchange a<->b; data: paper/teoria/koszt.jsonl):
  c* = 0  if 11 ∉ Π or Π = {11}
  c* = 1  otherwise, if 01 ∉ Π or 10 ∉ Π
  c* = 2  if Π = {01,10,11}
  c* = 3  if Π = {00,01,10,11} and Q ⊄ L
  c* = 4  if Π = {00,01,10,11}, Q ⊆ L, L ≠ {0,1}^3
  c* = 6  if L = {0,1}^3            (lower bound: Shende & Markov, "On the CNOT-cost of TOFFOLI gates", QIC 2009)
Upper bounds are constructive (class 2: X_t CX(a,t) CX(b,t); class 3: Margolus gadget or mirror/swapped; class 4: CX(a,t)CX(b,t)CX(a,t)CX(b,t) with single-qubit gates, explicit numeric circuit in paper/teoria/klasa4_przyklad.qpy).

## What must be PROVED (lower bounds)
Monotonicity: L ⊆ L' implies c*(L) ≤ c*(L'). So it suffices to prove each bound on the MINIMAL supports of the class:
 (B1) c* ≥ 1 when 11 ∈ Π and Π ≠ {11}: minimal L = {11t, uvs} with uv ≠ 11.
 (B2) c* ≥ 2 when Π = {01,10,11}: minimal L has three strings, one per control value in Π.
 (B3) c* ≥ 3 when Π = {00,01,10,11}: minimal L has four strings, one per control value (any targets).
 (B4) c* ≥ 4 when L ⊇ Q ∪ {00x, 11y} for some x, y (minimal L = Q ∪ {00x, 11y}, 4 choices of (x,y), up to symmetry).
Note: a k-CNOT circuit can be padded to k+2 (CX·CX = I), so "impossible with k" does NOT imply "impossible with k-1"; each k below the claimed value must be excluded (or use a parity-free argument).

## Useful facts
- CCX restricted to L maps basis states to basis states. Any valid U must send each |x>, x∈L, to a phase times a basis state.
- With k ≤ 1 CNOT, some qubit is never touched by a CNOT, so U = V ⊗ W (W on that qubit).
- With 2 CNOTs, either both act on the same pair (third qubit untouched) or they form a chain i–j, j–k sharing one qubit j.
- Song & Klappenecker (2004) proved 3 CNOTs optimal for "simplified Toffoli" (CCX up to a diagonal on the WHOLE space); our condition is weaker (only on span(L)), so their result does not directly apply, but their technique may.
- Single-qubit gates are arbitrary; global phase is free; only ONE common phase on L.

## Rules
- Python: ~/anaconda3/envs/ml/bin/python (numpy, sympy, scipy, torch available). Use OMP_NUM_THREADS<=4.
- A proof must be complete and checkable by a mathematician: state lemmas, give every step, list every case. Computer-assisted steps are allowed only if EXACT (symbolic/rational/exhaustive finite enumeration) or rigorously bounded (interval arithmetic with proven enclosures); plain floating-point optimization is NOT a proof.
- If you cannot complete a proof, say exactly which case remains open and why; do not claim more than you proved.
- Deliverable: <id>/DOWOD.md (statement, proof, case list, what is proven / open), plus any scripts with their output. Reply with 10-15 lines: what is proven, what is open, how confident, how a referee would check it.
